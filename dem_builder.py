import asyncio
import math
from collections import defaultdict
from pathlib import Path

import aiopath
import httpx
import rasterio
import rasterio.errors
import requests
from osgeo import gdal


def is_valid_geotiff(file: Path, crs: rasterio.CRS | None = None) -> bool:
    try:
        with rasterio.open(file) as r:
            if crs is not None:
                return r.crs == crs
            return True
    except rasterio.errors.RasterioError:
        return False


async def download(url: str, dest_dir: aiopath.AsyncPath):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download {url}: {response.status_code}")
        return response.content


async def download_all(urls: list[str], dest_dir: Path) -> list[Path]:
    count = len(urls)
    _display_width = math.ceil(math.log(count + 1))
    downloaded = []
    for i, url in enumerate(urls):
        dest = dest_dir / url.split("/")[-1]
        downloaded.append(dest)
        if dest.is_file() and is_valid_geotiff(dest):
            print(
                f"[{i + 1: {_display_width}}/{count: {_display_width}}] Skipping {dest}, already downloaded"
            )
            continue

        print(
            f"[{i + 1: {_display_width}}/{count: {_display_width}}] Downloading from {url}"
        )
        with requests.get(url, stream=True) as resp:
            if not resp.ok:
                print(
                    f"{' ' * (_display_width * 2 + 3)} Failed to download {dest}, {resp.status_code}: {resp.reason}"
                )
                continue
            print(f"{' ' * (_display_width * 2 + 3)} Saving to {dest}")
            with dest.open("wb") as d:
                for chunk in resp.iter_content(chunk_size=8192):
                    d.write(chunk)

    return downloaded


async def amain(download_list: Path, dest_dir: aiopath.AsyncPath):
    download_urls = download_list.read_text().splitlines()
    async with asyncio.TaskGroup() as tg:
        for url in download_urls:
            tg.create_task(download(url, dest_dir))


async def main(download_list: Path, dest_dir: Path):
    print("Downloading src rasters")
    downloads = download_all(Path(download_list).read_text().splitlines(), dest_dir)
    crs_count: defaultdict[rasterio.CRS, int] = defaultdict(lambda: 0)
    for downloaded_tif in downloads:
        with rasterio.open(downloaded_tif) as src_raster:
            crs_count[src_raster.crs] += 1

    # get the most common CRS
    mode_crs = max(crs_count, key=crs_count.__getitem__)
    crs_unit, _ = rasterio.CRS(mode_crs).linear_units_factor
    # print(f"{mode_crs}: {rasterio.CRS(mode_crs).linear_units_factor}")
    if crs_unit != "metre":
        raise ValueError(f"Cannot use {mode_crs}, because it does not use meters")
    min_x = None
    min_y = None
    max_x = None
    max_y = None
    print(f"Reprojecting rasters to same CRS ({mode_crs})")
    rasters = []
    for downloaded_tif in downloads:
        reprojected_tif = downloaded_tif.with_stem(
            f"{downloaded_tif.stem}_{str(mode_crs).replace(":", "")}"
        )
        with rasterio.open(downloaded_tif) as src_raster:
            if src_raster.count != 1:
                raise ValueError(
                    f"Raster '{src_raster.files[0]}' has {src_raster.count} bands, 1 expected"
                )
            # reproject raster if it doesn't have the correct CRS
            if src_raster.crs != mode_crs:
                if not is_valid_geotiff(reprojected_tif, crs=mode_crs):
                    print(
                        f"=> Reprojecting from {downloaded_tif} ({src_raster.crs}) to {reprojected_tif} ({mode_crs})"
                    )
                    warp = gdal.Warp(
                        str(reprojected_tif),
                        str(downloaded_tif),
                        options=gdal.WarpOptions(dstSRS=mode_crs.to_string()),
                    )
                    warp = None
                else:
                    print(
                        f"=> {downloaded_tif} ({src_raster.crs}) has already been reprojected to {reprojected_tif} ({mode_crs})"
                    )
                raster_tif = reprojected_tif
            else:
                raster_tif = downloaded_tif
            rasters.append(str(raster_tif))
            # print(dir(raster))
            # print(raster.files)
            with rasterio.open(raster_tif) as raster:
                bl = (raster.bounds.left, raster.bounds.bottom)
                tr = (raster.bounds.right, raster.bounds.top)

                if min_x is None or raster.bounds.left < min_x:
                    min_x = raster.bounds.left
                if min_y is None or raster.bounds.bottom < min_y:
                    min_y = raster.bounds.bottom

                if max_x is None or raster.bounds.right > max_x:
                    max_x = raster.bounds.right
                if max_y is None or raster.bounds.top > max_y:
                    max_y = raster.bounds.top
    # Construct new VRT
    output_vrt = dest_dir / "output.vrt"
    print(f"Building VRT from {len(rasters)} rasters")
    vrt_dataset = gdal.BuildVRT(
        str(output_vrt),
        rasters,
        options=gdal.BuildVRTOptions(resampleAlg="cubic"),
    )

    # Clip the VRT to the new size
    # calculate new bounding box:
    original_width = max_x - min_x
    clipped_width = (original_width // 700) * 700
    original_height = max_y - min_y
    clipped_height = (original_height // 700) * 700
    center = (round((min_x + max_x) / 2), round((min_y + max_y) / 2))

    print(
        f"{original_width=} {clipped_width=} {original_height=} {clipped_height=} {center=} {min_x=} {min_y=} {max_x=} {max_y=}"
    )
    clipped_png = dest_dir / "clipped.png"

    print("Clipping VRT, converting to PNG")
    clipped_dataset = gdal.Translate(
        str(clipped_png),
        vrt_dataset,
        options=gdal.TranslateOptions(
            format="PNG",
            projWin=(
                center[0] - (clipped_width // 2),
                center[1] + (clipped_height // 2),
                center[0] + (clipped_width // 2),
                center[1] - (clipped_height // 2),
            ),
            scaleParams=[
                (-12, 500, 0, 65535),
            ],
            width=clipped_width / 3.5,
            height=clipped_height / 3.5,
            resampleAlg="cubic",
            outputType=gdal.GDT_UInt16,
        ),
    )
    print(f"{clipped_dataset=}")
    if clipped_dataset is None:
        print("Failed to generate clipped PNG")
    else:
        clipped_dataset = None
        print(f"Clipped PNG saved to {clipped_png}")
    print("Done!")


if __name__ == "__main__":
    import sys

    asyncio.run(main(Path(sys.argv[1]), Path(sys.argv[2])))
