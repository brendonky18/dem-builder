import asyncio
import cmath
import logging
import logging.config
import math
import numbers
import os
import sys
import time
import warnings
from contextlib import ExitStack
from contextlib import asynccontextmanager
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import aiofiles
import httpx
import numpy as np
import rasterio
import rasterio.errors
import rasterio.merge
import rasterio.warp
import tqdm
from affine import Affine
from rasterio import windows
from rasterio.enums import Resampling
from rasterio.errors import MergeError
from rasterio.errors import RasterioDeprecationWarning
from rasterio.errors import RasterioError
from rasterio.errors import WindowError
from rasterio.io import DatasetWriter
from rasterio.transform import Affine
from rasterio.windows import subdivide


logger = logging.getLogger(__name__)


def merge_with_progress(
    sources,
    bounds=None,
    res=None,
    nodata=None,
    dtype=None,
    precision=None,
    indexes=None,
    output_count=None,
    resampling=Resampling.nearest,
    method="first",
    target_aligned_pixels=False,
    mem_limit=64,
    use_highest_res=False,
    masked=False,
    dst_path=None,
    dst_kwds=None,
):
    """Copy valid pixels from input files to an output file.

    All files must have the same number of bands, data type, and
    coordinate reference system. Rotated, flipped, or upside-down
    rasters cannot be merged.

    Input files are merged in their listed order using the reverse
    painter's algorithm (default) or another method. If the output file
    exists, its values will be overwritten by input values.

    Geospatial bounds and resolution of a new output file in the units
    of the input file coordinate reference system may be provided and
    are otherwise taken from the first input file.

    Parameters
    ----------
    sources : list
        A sequence of dataset objects opened in 'r' mode or Path-like
        objects.
    bounds: tuple, optional
        Bounds of the output image (left, bottom, right, top).
        If not set, bounds are determined from bounds of input rasters.
    res: tuple, optional
        Output resolution in units of coordinate reference system. If
        not set, a source resolution will be used. If a single value is
        passed, output pixels will be square.
    use_highest_res: bool, optional. Default: False.
        If True, the highest resolution of all sources will be used. If
        False, the first source's resolution will be used.
    nodata: float, optional
        nodata value to use in output file. If not set, uses the nodata
        value in the first input raster.
    masked: bool, optional. Default: False.
        If True, return a masked array. Note: nodata is always set in
        the case of file output.
    dtype: numpy.dtype or string
        dtype to use in outputfile. If not set, uses the dtype value in
        the first input raster.
    precision: int, optional
        This parameters is unused, deprecated in rasterio 1.3.0, and
        will be removed in version 2.0.0.
    indexes : list of ints or a single int, optional
        bands to read and merge
    output_count: int, optional
        If using callable it may be useful to have additional bands in
        the output in addition to the indexes specified for read
    resampling : Resampling, optional
        Resampling algorithm used when reading input files.
        Default: `Resampling.nearest`.
    method : str or callable
        pre-defined method:

            * first: reverse painting
            * last: paint valid new on top of existing
            * min: pixel-wise min of existing and new
            * max: pixel-wise max of existing and new
            * sum: pixel-wise sum of existing and new
            * count: pixel-wise count of valid pixels

        or custom callable with signature:
            merged_data : array_like
                array to update with new_data
            new_data : array_like
                data to merge
                same shape as merged_data
            merged_mask, new_mask : array_like
                boolean masks where merged/new data pixels are invalid
                same shape as merged_data
            index: int
                index of the current dataset within the merged dataset
                collection
            roff: int
                row offset in base array
            coff: int
                column offset in base array

    target_aligned_pixels : bool, optional
        Whether to adjust output image bounds so that pixel coordinates
        are integer multiples of pixel size, matching the ``-tap``
        options of GDAL utilities.  Default: False.
    mem_limit : int, optional
        Process merge output in chunks of mem_limit MB in size.
    dst_path : str or PathLike, optional
        Path of output dataset
    dst_kwds : dict, optional
        Dictionary of creation options and other parameters that will be
        overlaid on the profile of the output dataset.

    Returns
    -------
    tuple
        Two elements:
            dest: numpy.ndarray
                Contents of all input rasters in single array
            out_transform: affine.Affine()
                Information for mapping pixel coordinates in `dest` to
                another coordinate system

    Raises
    ------
    rasterio.errors.MergeError
        When sources cannot be merged due to incompatibility between
        them or limitations of the tool.
    """
    if method in rasterio.merge.MERGE_METHODS:
        copyto = rasterio.merge.MERGE_METHODS[method]
    elif callable(method):
        copyto = method
    else:
        raise ValueError(
            "Unknown method {}, must be one of {} or callable".format(
                method, list(rasterio.merge.MERGE_METHODS.keys())
            )
        )

    # Create a dataset_opener object to use in several places in this function.
    if isinstance(sources[0], (str, os.PathLike)):
        dataset_opener = rasterio.open
    else:

        @contextmanager
        def nullcontext(obj):
            try:
                yield obj
            finally:
                pass

        dataset_opener = nullcontext

    dst = None

    with ExitStack() as exit_stack:
        with dataset_opener(sources[0]) as first:
            first_profile = first.profile
            first_crs = first.crs
            best_res = first.res
            first_nodataval = first.nodatavals[0]
            nodataval = first_nodataval
            dt = first.dtypes[0]

            if indexes is None:
                src_count = first.count
            elif isinstance(indexes, int):
                src_count = indexes
            else:
                src_count = len(indexes)

            try:
                first_colormap = first.colormap(1)
            except ValueError:
                first_colormap = None

        if not output_count:
            output_count = src_count

        # Extent from option or extent of all inputs
        if bounds:
            dst_w, dst_s, dst_e, dst_n = bounds
        else:
            # scan input files
            xs = []
            ys = []

            for i, dataset in enumerate(sources):
                with dataset_opener(dataset) as src:
                    src_transform = src.transform

                    if use_highest_res:
                        best_res = min(
                            best_res,
                            src.res,
                            key=lambda x: (
                                x
                                if isinstance(x, numbers.Number)
                                else math.sqrt(x[0] ** 2 + x[1] ** 2)
                            ),
                        )

                    # The merge tool requires non-rotated rasters with origins at their
                    # upper left corner. This limitation may be lifted in the future.
                    if not src_transform.is_rectilinear:
                        raise MergeError(
                            "Rotated, non-rectilinear rasters cannot be merged."
                        )
                    if src_transform.a < 0:
                        raise MergeError(
                            'Rasters with negative pixel width ("flipped" rasters) cannot be merged.'
                        )
                    if src_transform.e > 0:
                        raise MergeError(
                            'Rasters with negative pixel height ("upside down" rasters) cannot be merged.'
                        )

                    left, bottom, right, top = src.bounds

                xs.extend([left, right])
                ys.extend([bottom, top])

            dst_w, dst_s, dst_e, dst_n = min(xs), min(ys), max(xs), max(ys)

        # Resolution/pixel size
        if not res:
            res = best_res
        elif isinstance(res, numbers.Number):
            res = (res, res)
        elif len(res) == 1:
            res = (res[0], res[0])

        if target_aligned_pixels:
            dst_w = math.floor(dst_w / res[0]) * res[0]
            dst_e = math.ceil(dst_e / res[0]) * res[0]
            dst_s = math.floor(dst_s / res[1]) * res[1]
            dst_n = math.ceil(dst_n / res[1]) * res[1]

        # Compute output array shape. We guarantee it will cover the output
        # bounds completely
        output_width = int(round((dst_e - dst_w) / res[0]))
        output_height = int(round((dst_n - dst_s) / res[1]))

        output_transform = Affine.translation(dst_w, dst_n) * Affine.scale(
            res[0], -res[1]
        )

        if dtype is not None:
            dt = dtype
            logger.debug("Set dtype: %s", dt)

        if nodata is not None:
            nodataval = nodata
            logger.debug("Set nodataval: %r", nodataval)

        inrange = False
        if nodataval is not None:
            # Only fill if the nodataval is within dtype's range
            if np.issubdtype(dt, np.integer):
                info = np.iinfo(dt)
                inrange = info.min <= nodataval <= info.max
            else:
                if cmath.isfinite(nodataval):
                    info = np.finfo(dt)
                    inrange = info.min <= nodataval <= info.max
                    nodata_dt = np.min_scalar_type(nodataval)
                    inrange = inrange & np.can_cast(nodata_dt, dt)
                else:
                    inrange = True

            if not inrange:
                warnings.warn(
                    f"Ignoring nodata value. The nodata value, {nodataval}, cannot safely be represented "
                    f"in the chosen data type, {dt}. Consider overriding it "
                    "using the --nodata option for better results. "
                    "Falling back to first source's nodata value."
                )
                nodataval = first_nodataval
        else:
            logger.debug("Set nodataval to 0")
            nodataval = 0

        # Round the width and height to the nearest multiple of 700
        output_width = round(output_width / 700) * 700
        output_height = round(output_height / 700) * 700

        center_x = round((output_width) / 2)
        center_y = round((output_height) / 2)
        x_offset = round(output_width / 2)
        y_offset = round(output_height / 2)

        # When dataset output is selected, we might need to create one
        # and will also provide the option of merging by chunks.
        dout_window = windows.Window(
            center_x - x_offset,
            center_y - y_offset,
            center_x + x_offset,
            center_y + y_offset,
        )
        if dst_path is not None:
            if isinstance(dst_path, DatasetWriter):
                dst = dst_path
            else:
                out_profile = first_profile
                out_profile.update(**(dst_kwds or {}))
                out_profile["transform"] = output_transform
                out_profile["height"] = output_height
                out_profile["width"] = output_width
                out_profile["count"] = output_count
                out_profile["dtype"] = dt
                if nodata is not None:
                    out_profile["nodata"] = nodata
                dst = rasterio.open(dst_path, "w", **out_profile)
                exit_stack.enter_context(dst)

            max_pixels = mem_limit * 1.0e6 / (np.dtype(dt).itemsize * output_count)

            if output_width * output_height < max_pixels:
                chunks = [dout_window]
            else:
                n = math.floor(math.sqrt(max_pixels))
                chunks = subdivide(dout_window, n, n)
        else:
            chunks = [dout_window]

        def _intersect_bounds(bounds1, bounds2, transform):
            """Based on gdal_merge.py."""
            int_w = max(bounds1[0], bounds2[0])
            int_e = min(bounds1[2], bounds2[2])

            if int_w >= int_e:
                raise ValueError

            if transform.e < 0:
                # north up
                int_s = max(bounds1[1], bounds2[1])
                int_n = min(bounds1[3], bounds2[3])
                if int_s >= int_n:
                    raise ValueError
            else:
                int_s = min(bounds1[1], bounds2[1])
                int_n = max(bounds1[3], bounds2[3])
                if int_n >= int_s:
                    raise ValueError

            return int_w, int_s, int_e, int_n

        with tqdm.tqdm(
            desc="Merging", total=len(chunks) * len(sources)
        ) as merge_progress:
            for chunk in chunks:
                dst_w, dst_s, dst_e, dst_n = windows.bounds(chunk, output_transform)
                dest = np.zeros((output_count, chunk.height, chunk.width), dtype=dt)
                if inrange:
                    dest.fill(nodataval)

                # From gh-2221
                chunk_bounds = windows.bounds(chunk, output_transform)
                chunk_transform = windows.transform(chunk, output_transform)

                def win_align(window):
                    """Equivalent to rounding both offsets and lengths.

                    This method computes offsets, width, and height that are
                    useful for compositing arrays into larger arrays and
                    datasets without seams. It is used by Rasterio's merge
                    tool and is based on the logic in gdal_merge.py.

                    Returns
                    -------
                    Window
                    """
                    row_off = math.floor(window.row_off + 0.1)
                    col_off = math.floor(window.col_off + 0.1)
                    height = math.floor(window.height + 0.5)
                    width = math.floor(window.width + 0.5)
                    return windows.Window(col_off, row_off, width, height)

                for idx, dataset in enumerate(sources):
                    merge_progress.update()
                    with dataset_opener(dataset) as src:

                        # Intersect source bounds and tile bounds
                        if first_crs != src.crs:
                            raise RasterioError(f"CRS mismatch with source: {dataset}")

                        try:
                            ibounds = _intersect_bounds(
                                src.bounds, chunk_bounds, chunk_transform
                            )
                            sw = windows.from_bounds(*ibounds, src.transform)
                            cw = windows.from_bounds(*ibounds, chunk_transform)
                        except (ValueError, WindowError):
                            continue

                        cw = win_align(cw)
                        rows, cols = cw.toslices()
                        region = dest[:, rows, cols]

                        if cmath.isnan(nodataval):
                            region_mask = np.isnan(region)
                        elif not np.issubdtype(region.dtype, np.integer):
                            region_mask = np.isclose(region, nodataval)
                        else:
                            region_mask = region == nodataval

                        data = src.read(
                            out_shape=(src_count, cw.height, cw.width),
                            indexes=indexes,
                            masked=True,
                            window=sw,
                            resampling=resampling,
                        )

                        copyto(
                            region,
                            data,
                            region_mask,
                            data.mask,
                            index=idx,
                            roff=cw.row_off,
                            coff=cw.col_off,
                        )

                if dst:
                    dw = windows.from_bounds(*chunk_bounds, output_transform)
                    dw = win_align(dw)
                    dst.write(dest, window=dw)

        if dst is None:
            if masked:
                dest = np.ma.masked_equal(dest, nodataval, copy=False)
            return dest, output_transform
        else:
            if first_colormap:
                dst.write_colormap(1, first_colormap)
            dst.close()


def is_valid_geotiff(file: Path, crs: rasterio.CRS | None = None) -> bool:
    try:
        with rasterio.open(file) as r:
            if crs is not None and r.crs != crs:
                return False
            for i in r.indexes:
                r.checksum(i)
    except rasterio.errors.RasterioError as e:
        logger.warning(f"Error validating {file}: {e}")
        return False
    else:
        logger.debug(f"Successfully validated {file}")
        return True


class MemoryManager(asyncio.Event):
    def __init__(self, max_memory: int):
        super().__init__()
        self.max_memory = max_memory
        self.current_memory = 0
        self.set()  # initially, memory is below limit

    @asynccontextmanager
    async def acquire(self, size: int, name: str):
        if size > self.max_memory:
            raise ValueError(
                f"Requested memory size {size/1024**2:4.2f} MB exceeds maximum memory {self.max_memory/1024**3:4.2f} GB"
            )
        while self.current_memory + size > self.max_memory:
            await self.wait()
            self.clear()
        self.current_memory += size

        try:
            yield
        finally:
            self.current_memory -= size
            self.set()


async def download(url: str, dest_dir: Path, memory_manager: MemoryManager) -> Path:
    dest_path = dest_dir / url.split("/")[-1]
    if dest_path.is_file() and is_valid_geotiff(dest_path):
        logger.info(f"Skipping {url}, already downloaded")
        return dest_path

    async with httpx.AsyncClient() as client, client.stream("GET", url) as response:
        response.raise_for_status()
        download_size = int(response.headers["content-length"])
        encoding = response.headers.get("Content-Encoding")
        async with memory_manager.acquire(download_size, f"download {url}"):
            logger.info(f"Downloading {url}")
            # contents = await response.aread()
            async with aiofiles.open(dest_path, "ab") as f:
                async for chunk in response.aiter_bytes():
                    await f.write(chunk)
            # async with aiofiles.open(dest_path, "wb") as f:
            # await f.write(contents)
            logger.info(f"Saved {url} to {dest_path}")
    return dest_path


async def reproject_to_crs(
    src_tif: Path, dst_crs: rasterio.CRS, mem_limit: int = 0
) -> Path:
    if not is_valid_geotiff(src_tif):
        raise ValueError(f"Source file {src_tif} is not a valid GeoTIFF")

    with rasterio.open(src_tif) as src_dataset:
        if src_dataset.crs == dst_crs:
            logger.info(f"{src_tif} is already in {dst_crs}, skipping reprojection")
            return src_tif

        dest_dir = src_tif.parent.parent / str(dst_crs).replace(":", "-")
        dest_dir.mkdir(parents=True, exist_ok=True)
        reprojected_tif = dest_dir / (src_tif.name)
        if reprojected_tif.is_file() and is_valid_geotiff(reprojected_tif, dst_crs):
            logger.info(f"Skipping reprojection for {src_tif}, already reprojected")
            return reprojected_tif

        transform, width, height = rasterio.warp.calculate_default_transform(
            src_dataset.crs,
            dst_crs,
            src_dataset.width,
            src_dataset.height,
            *src_dataset.bounds,
        )
        kwargs = src_dataset.meta.copy()
        kwargs["crs"] = dst_crs
        kwargs["transform"] = transform
        kwargs["width"] = width
        kwargs["height"] = height

        with rasterio.open(reprojected_tif, "w", **kwargs) as reprojected_dataset:
            logger.info(f"Reprojecting {src_tif} from {src_dataset.crs} to {dst_crs}")
            for i in range(1, src_dataset.count + 1):
                try:
                    rasterio.warp.reproject(
                        source=rasterio.band(src_dataset, i),
                        destination=rasterio.band(reprojected_dataset, i),
                        src_transform=src_dataset.transform,
                        src_crs=src_dataset.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=rasterio.warp.Resampling.cubic,
                        warp_mem_limit=mem_limit // 1024**2,
                    )
                except rasterio.errors.RasterioError as e:
                    e.add_note(f"Error reprojecting band {i} of {src_tif}")
                    raise
        return reprojected_tif


def convert_data_types(
    src_tif: Path, min_elevation: int = 0, max_elevation: int = 4096
) -> Path:
    dest_dir = src_tif.parent.parent / "converted"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_tif = dest_dir / src_tif.name
    elevation_range = max_elevation - min_elevation
    if dest_tif.is_file():
        with rasterio.open(dest_tif) as dest:
            if dest.meta["dtype"] == "uint16":
                logger.info(f"Skipping conversion for {src_tif}, already converted")
                return dest_tif

    with rasterio.open(src_tif, "r") as src:
        src_data = src.read()
        src_meta: dict = src.meta.copy()
    src_meta.update(dtype=rasterio.uint16, driver="GTiff", nodata=0)
    # convert array to png
    src_data[src_data == src.nodata] = np.nan
    # remap the elevation values to the specified range
    src_data -= min_elevation
    src_data *= 65535 / elevation_range
    src_data = np.clip(src_data, 0, 65535)
    np.nan_to_num(src_data, copy=False, nan=0)
    src_data = src_data.astype(np.uint16)
    with rasterio.open(dest_tif, "w", **src_meta) as dest:
        dest.write(src_data)

    return dest_tif


type seconds = float
type nanoseconds = int


@dataclass
class RateLimiter:
    """Allows at most `frequency` operations every `time` seconds."""

    time: seconds
    frequency: int = 1

    _next_reset: nanoseconds = 0
    _burst_count: int = 0

    async def wait(self):
        if self._burst_count < self.frequency:
            self._burst_count += 1
            return
        else:
            sleep_time = (self._next_reset - time.perf_counter_ns()) / 1e9
            if sleep_time > 0:
                logger.info(
                    f"Rate limit reached, sleeping for {sleep_time:.2f} seconds"
                )
                await asyncio.sleep(sleep_time)
            self._next_reset += int(self.time * 1e9)
            self._burst_count = 1
            return
        # if we have reached the burst limit, wait until the next reset time
        # and reset the burst count and the next reset time
        # otherwise, increment the burst count


async def main(
    src: Path,
    dst: Path,
    crs: rasterio.CRS,
    output: Path,
    mem_limit: int = 500 * 1024**2,
    elevation_min: int = 0,
    elevation_max: int = 4096,
):
    elevation_range = elevation_max - elevation_min
    if elevation_range <= 0:
        raise ValueError(
            f"Minimum elevation {elevation_min} is not less than maximum elevation {elevation_max}"
        )
    elif elevation_range > 65536:
        raise ValueError(
            f"Elevation range {elevation_range} is too large to fit in a 16-bit PNG"
        )

    memory_manager = MemoryManager(int(mem_limit * 1e6))
    limiter = RateLimiter(frequency=5, time=1)

    async with aiofiles.open(src, "r") as src_file:
        urls = [line.strip() for line in (await src_file.readlines()) if line.strip()]

    download_bar = tqdm.tqdm(desc="Downloading", total=len(urls))

    async def download_and_reproject(url: str) -> Path:
        # nonlocal current_memory_usage
        reprojected_file = None
        for i in range(5):
            try:
                await limiter.wait()
                downloaded_file = await download(url, downloads_dir, memory_manager)
            except Exception as e:
                logger.warning(f"Error downloading {url}: {e!r}. Retrying ({i+1}/5)...")
                continue
            try:
                reprojected_file = await reproject_to_crs(
                    downloaded_file, crs, mem_limit
                )
            except Exception as e:
                downloaded_file.unlink(missing_ok=True)
                logger.warning(
                    f"Error reprojecting {downloaded_file}: {e!r}. Retrying ({i+1}/5)..."
                )
            if reprojected_file is not None:
                download_bar.update()
                return reprojected_file

        raise RuntimeError(f"Failed to download and reproject {url} after 5 attempts")

    # Download all the files
    with download_bar:
        async with asyncio.TaskGroup() as tg:
            downloads_dir = dst / "downloads"
            downloads_dir.mkdir(parents=True, exist_ok=True)
            results = [tg.create_task(download_and_reproject(url)) for url in urls]

        # Build a VRT from all the reprojected files
        reprojected_tifs = [task.result() for task in results]
    converted_tifs = [
        convert_data_types(tif, elevation_min, elevation_max)
        for tif in tqdm.tqdm(reprojected_tifs, desc="Converting")
    ]
    merge_with_progress(
        converted_tifs,
        dst_path=output,
        mem_limit=mem_limit,
    )
    logger.info(f"Saved merged output to {output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path, help="Path to the source file with URLs")
    parser.add_argument("dst", type=Path, help="Path to the destination directory")
    parser.add_argument(
        "crs", type=rasterio.CRS.from_string, help="Target CRS in EPSG format"
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="Output file path", default=Path("output.png")
    )
    parser.add_argument(
        "-l",
        "--mem-limit",
        type=float,
        help="Memory limit in MB (default: 500 MB)",
        default=500,
    )
    parser.add_argument(
        "-m",
        "--elevation-min",
        type=int,
        help="Minimum elevation value (default: 0)",
        default=0,
    )
    parser.add_argument(
        "-M",
        "--elevation-max",
        type=int,
        help="Maximum elevation value (default: 4096)",
        default=4096,
    )
    parser.add_argument(
        "-v", "--verbose", action="count", help="Set the logging verbosity", default=0
    )

    class TQDMHandler(logging.Handler):
        def emit(self, record):
            if record.name != "__main__":
                return

            symbol_map = {
                logging.DEBUG: "~",
                logging.INFO: "+",
                logging.WARNING: "!",
                logging.ERROR: "x",
                logging.CRITICAL: "x",
            }
            record.symbol = symbol_map[record.levelno]
            if record.levelno >= logging.WARNING:
                file = sys.stderr
            else:
                file = sys.stdout
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg, file=file)
                self.flush()
            except Exception:
                self.handleError(record)

    args = parser.parse_args()

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "log_file": {
                    "format": "[%(levelname)s|%(name)s:L%(lineno)d] %(message)s",
                },
                "simple": {
                    "format": "[%(symbol)s] %(message)s",
                },
            },
            "handlers": {
                "tqdm": {
                    "()": TQDMHandler,
                    "formatter": "simple",
                    "level": logging.WARNING - args.verbose * 10,
                },
                "file": {
                    "class": "logging.FileHandler",
                    "filename": "dem_builder.log",
                    "mode": "w",
                    "formatter": "log_file",
                    "level": logging.DEBUG,
                },
            },
            "loggers": {
                "root": {
                    "handlers": ["tqdm", "file"],
                    "level": 0,
                    "propagate": False,
                }
            },
        }
    )

    asyncio.run(
        main(
            args.src,
            args.dst,
            args.crs,
            output=args.output,
            mem_limit=args.mem_limit,
            elevation_min=args.elevation_min,
            elevation_max=args.elevation_max,
        )
    )
