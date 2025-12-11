import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from contextlib import contextmanager
from pathlib import Path

import aiofiles
import httpx
import rasterio
import rasterio.errors
from osgeo import gdal


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
        print(f"current memory={self.current_memory/1024**3:4.2f} {self.is_set()=}")
        while self.current_memory + size > self.max_memory:
            print(
                f"Waiting for memory for {name}: current memory={self.current_memory/1024**3: 4.2f} GB size={size/1024**2: 4.2f} MB"
            )
            await self.wait()
        self.current_memory += size
        print(
            f"Acquired memory for {name}: current memory={self.current_memory/1024**3: 4.2f} GB size={size/1024**2: 4.2f} MB"
        )
        if self.current_memory > self.max_memory:
            self.clear()

        try:
            yield
        finally:
            self.release(size, name)

    def release(self, size: int, name: str):
        self.current_memory -= size
        print(
            f"Releasing memory for {name}: current memory={self.current_memory/1024**3: 4.2f} GB size={size/1024**2: 4.2f} MB"
        )
        if self.current_memory < self.max_memory:
            self.set()


async def download(url: str, dest_dir: Path, memory_manager: MemoryManager) -> Path:
    dest_path = dest_dir / url.split("/")[-1]
    if dest_path.is_file() and is_valid_geotiff(dest_path):
        print(f"Skipping {url}, already downloaded")
        return dest_path

    async with httpx.AsyncClient() as client, client.stream("GET", url) as response:
        # if response.status_code != 200:
        #     raise RuntimeError(f"Failed to download {url}: {response.status_code}")
        response.raise_for_status()
        print(f"Headers: {response.headers}")
        download_size = int(response.headers["content-length"])
        print(f"{download_size=}")
        encoding = response.headers.get("Content-Encoding")
        print(f"{encoding=}")
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


async def get_crs(file: Path) -> rasterio.CRS | None:
    async with aiofiles.open(file, "rb") as f:
        content = await f.read()
    with rasterio.MemoryFile(content) as memfile:
        with memfile.open() as dataset:
            return dataset.crs


@contextmanager
def gdal_vsimem_file(file_path: Path, data: bytes):
    gdal.FileFromMemBuffer(str(file_path), data)
    result = gdal.VSIStatL(str(file_path), gdal.VSI_STAT_SIZE_FLAG)
    print(f"{result.size=} {file_path=}")

    try:
        yield
    except Exception as e:
        print(f"Got exception {e!s}")
        raise
    finally:
        print(f"Deleting {file_path} from /vsimem")
        gdal.Unlink(str(file_path))


async def reproject_to_crs(src_tif: Path, dst_crs: rasterio.CRS) -> Path:
    print(f"Reprojecting {src_tif} to {dst_crs}")
    with rasterio.open(src_tif) as src_dataset:
        if src_dataset.crs == dst_crs:
            print(f"{src_tif} is already in {dst_crs}, skipping reprojection")
            return src_tif
        reprojected_tif = src_tif.parent / (
            f"{src_tif.stem}_{str(dst_crs).replace(":", "-")}{src_tif.suffix}"
        )
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
            for i in range(1, src_dataset.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src_dataset, i),
                    destination=rasterio.band(reprojected_dataset, i),
                    src_transform=src_dataset.transform,
                    src_crs=src_dataset.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=rasterio.warp.Resampling.nearest,
                    warp_mem_limit=1 * 1024**3,
                )
        return Path(reprojected_tif)


async def main(src: Path, dst: Path, crs: rasterio.CRS, MAX_MEMORY: int = 2 * 1024**3):
    # current_memory_usage = 0
    # memory_usage_below_limit = asyncio.Event()

    memory_manager = MemoryManager(MAX_MEMORY)

    async def download_and_reproject(url: str) -> Path:
        # nonlocal current_memory_usage
        downloaded_file = None
        while downloaded_file is None:
            try:
                downloaded_file = await download(url, dest_dir, memory_manager)
            except Exception as e:
                print(f"Error downloading {url}: {e!s}. Retrying...")
                await asyncio.sleep(1)
        # file_size = downloaded_file.stat().st_size
        # while current_memory_usage > MAX_MEMORY:
        #     print(f"waiting: {current_memory_usage=}")
        #     await memory_usage_below_limit.wait()
        # current_memory_usage += file_size
        async with memory_manager.acquire(1 * 1024**3, f"reproject {url}"):
            reprojected_file = await reproject_to_crs(downloaded_file, crs)
        # current_memory_usage -= file_size
        # memory_usage_below_limit.set()
        return reprojected_file

    # Download all the files
    src_tifs = []
    async with asyncio.TaskGroup() as tg:
        async with aiofiles.open(src, "r") as src_file:
            urls = [
                line.strip() for line in (await src_file.readlines()) if line.strip()
            ]
        dest_dir = dst / "downloads"
        dest_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for url in urls:
            cur_tif = dest_dir / url.split("/")[-1]
            src_tifs.append(cur_tif)
            results.append(tg.create_task(download_and_reproject(url)))

            # throttle download rate
            await asyncio.sleep(0.2)

    # Build a VRT from all the reprojected files


if __name__ == "__main__":
    import sys

    # gdal.WarpOptions(options=["--version"])
    # gdal.Warp("", "", options=["--version"])
    # print(gdal.VersionInfo())
    asyncio.run(
        main(
            Path(sys.argv[1]), Path(sys.argv[2]), rasterio.CRS.from_string(sys.argv[3])
        )
    )
