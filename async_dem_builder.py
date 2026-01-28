import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import aiofiles
import httpx
import numpy as np
import rasterio
import rasterio.errors
import rasterio.merge
import rasterio.warp


def is_valid_geotiff(file: Path, crs: rasterio.CRS | None = None) -> bool:
    try:
        with rasterio.open(file) as r:
            if crs is not None:
                return r.crs == crs
            return True
    except rasterio.errors.RasterioError:
        return False


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
        print(f"Skipping {url}, already downloaded")
        return dest_path

    async with httpx.AsyncClient() as client, client.stream("GET", url) as response:
        response.raise_for_status()
        download_size = int(response.headers["content-length"])
        encoding = response.headers.get("Content-Encoding")
        async with memory_manager.acquire(download_size, f"download {url}"):
            print(f"Downloading {url}")
            # contents = await response.aread()
            async with aiofiles.open(dest_path, "ab") as f:
                async for chunk in response.aiter_bytes():
                    await f.write(chunk)
            # async with aiofiles.open(dest_path, "wb") as f:
            # await f.write(contents)
            print(f"Saved {url} to {dest_path}")
    return dest_path


async def reproject_to_crs(
    src_tif: Path, dst_crs: rasterio.CRS, mem_limit: int = 0
) -> Path:
    with rasterio.open(src_tif) as src_dataset:
        if src_dataset.crs == dst_crs:
            print(f"{src_tif} is already in {dst_crs}, skipping reprojection")
            return src_tif

        dest_dir = src_tif.parent.parent / str(dst_crs).replace(":", "-")
        dest_dir.mkdir(parents=True, exist_ok=True)
        reprojected_tif = dest_dir / (src_tif.name)
        if reprojected_tif.is_file() and is_valid_geotiff(reprojected_tif, dst_crs):
            print(f"Skipping reprojection for {src_tif}, already reprojected")
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
            print(f"Reprojecting {src_tif} to {dst_crs}")
            for i in range(1, src_dataset.count + 1):
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
                print(f"Skipping conversion for {src_tif}, already converted")
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
                print(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
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
    mem_limit: int = 1 * 1024**3,
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

    memory_manager = MemoryManager(mem_limit)
    limiter = RateLimiter(frequency=5, time=1)

    async def download_and_reproject(url: str) -> Path:
        # nonlocal current_memory_usage
        downloaded_file = None
        while downloaded_file is None:
            try:
                await limiter.wait()
                downloaded_file = await download(url, downloads_dir, memory_manager)
            except Exception as e:
                print(f"Error downloading {url}: {e!s}. Retrying...")
                await asyncio.sleep(1)

        reprojected_file = await reproject_to_crs(downloaded_file, crs, mem_limit)
        return reprojected_file

    # Download all the files
    async with asyncio.TaskGroup() as tg:
        async with aiofiles.open(src, "r") as src_file:
            urls = [
                line.strip() for line in (await src_file.readlines()) if line.strip()
            ]
        downloads_dir = dst / "downloads"
        downloads_dir.mkdir(parents=True, exist_ok=True)

        results = [tg.create_task(download_and_reproject(url)) for url in urls]

    # Build a VRT from all the reprojected files
    reprojected_tifs = [task.result() for task in results]
    converted_tifs = [
        convert_data_types(tif, elevation_min, elevation_max)
        for tif in reprojected_tifs
    ]
    print(f"Merging {len(converted_tifs)} rasters...")
    mosaic, mosaic_transform = rasterio.merge.merge(converted_tifs)
    with rasterio.open(converted_tifs[0]) as src0:
        kwargs = src0.meta.copy()
    kwargs["height"] = mosaic.shape[1]
    kwargs["width"] = mosaic.shape[2]
    kwargs["transform"] = mosaic_transform

    print(f"Saving to {output}...")
    with rasterio.open(output, "w", **kwargs) as dest:
        dest.write(mosaic)


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
        help="Memory limit in bytes (default: 1 GB)",
        default=1,
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

    args = parser.parse_args()

    asyncio.run(
        main(
            args.src,
            args.dst,
            args.crs,
            output=args.output,
            mem_limit=args.mem_limit * 1024**3,
            elevation_min=args.elevation_min,
            elevation_max=args.elevation_max,
        )
    )
