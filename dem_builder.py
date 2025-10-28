from collections import defaultdict
from pathlib import Path
import rasterio
import rasterio.errors
import rasterio.warp
import math
import requests
from osgeo import gdal


def is_valid_geotiff(file: Path) -> bool:
    try:
        with rasterio.open(file) as r:
            return True
    except rasterio.errors.RasterioError:
        return False


def download(urls: list[str]) -> list[Path]:
    c = len(urls)
    w = math.ceil(math.log(c + 1))
    downloaded = []
    for i, url in enumerate(urls):
        dest = Path(url.split("/")[-1])
        if dest.is_file() and is_valid_geotiff(dest):
            print(f"[{i: w}/{c: w}] Skipping {dest}, already downloaded")
            continue

        print(f"[{i: w}/{c: w}] Downloading from {url}")
        with requests.get(url, stream=True) as resp:
            if not resp.ok:
                print(
                    f"{' ' * (w * 2 + 3)} Failed to download {dest}, {resp.status_code}: {resp.reason}"
                )
                continue
            print(f"{' ' * (w * 2 + 3)} Saving to {dest}")
            with dest.open("wb") as d:
                for chunk in resp.iter_content(chunk_size=8192):
                    d.write(chunk)

            downloaded.append(dest)
    return downloaded


def main(download_list):
    print("Downloading src rasters")
    downloaded = download(Path(download_list).read_text().splitlines())
    crs_count: defaultdict[str, int] = defaultdict(lambda: 0)
    for d in downloaded:
        with rasterio.open(d) as src_raster:
            crs_count[src_raster.crs] += 1

    # get the most common CRS
    mode_crs = max(crs_count, crs_count.__getitem__)
    crs_unit, _ = mode_crs.linear_units_and_factors()
    if crs_unit != "m":
        raise ValueError(f"Cannot use {mode_crs}, because it does not use meters")
    min_coords = None
    max_coords = None
    print(f"Reprojecting rasters to same CRS ({mode_crs})")
    rasters = []
    for d in downloaded:
        with rasterio.open(d) as src_raster:
            # reproject raster if it doesn't have the correct CRS
            if src_raster.crs != mode_crs:
                reprojected = d.with_name(f"{d.name}_{mode_crs}")
                raster, _ = rasterio.warp.reproject(src_raster, reprojected)
            else:
                raster = src_raster
            rasters.append(raster.path)

            bl = (raster.bounds.left, raster.bounds.bottom)
            tr = (raster.bounds.right, raster.bounds.top)
            if min_coords is None or bl < min_coords:
                min_coords = bl
            if max_coords is None or tr > max_coords:
                max_coords = tr

    # Construct new VRT
    vrt = Path("output.vrt")
    print(f"Building VRT from {len(rasters)} rasters")
    vrt_dataset = gdal.BuildVRT(
        vrt, rasters, options=gdal.BuildVRTOptions(resampleAlg="cubic", addAlpha=True)
    )

    # Clip the VRT to the new size
    # calculate new bounding box:
    original_width = max_coords[0] - min_coords[0]
    clipped_width = (original_width // 700) * 700
    original_height = max_coords[1] - min_coords[1]
    clipped_height = (original_height // 700) * 700
    center = (min_coords[0] + (clipped_width / 2), min_coords[1] + {clipped_width / 2})
    clipped_png = Path("clipped.png")

    print("Clipping VRT and converting to PNG")
    clipped_dataset = gdal.Translate(
        clipped_png,
        vrt_dataset,
        options=gdal.TranslateOptions(
            format="PNG",
            projWin=(
                center[0] - (clipped_width / 2),
                center[1] - (clipped_height / 2),
                center[0] + (clipped_width / 2),
                center[1] + (clipped_height / 2),
            ),
            scaleParams=(-12, 500),
            width=clipped_width / 3.5,
            height=clipped_height / 3.5,
            resampleAlg="cubic",
        ),
    )
    clipped_dataset = None
    print(f"Clipped PNG saved to {clipped_png}")
    print("Done!")


if __name__ == "__main__":
    import sys

    main(sys.argv[1])
