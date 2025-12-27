import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import aiofiles
import httpx
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
        print(f"current memory={self.current_memory/1024**3:4.2f} {self.is_set()=}")
        while self.current_memory + size > self.max_memory:
            print(
                f"Waiting for memory for {name}: current memory={self.current_memory/1024**3: 4.2f} GB size={size/1024**2: 4.2f} MB"
            )
            await self.wait()
            self.clear()
        self.current_memory += size
        print(
            f"Acquired memory for {name}: current memory={self.current_memory/1024**3: 4.2f} GB size={size/1024**2: 4.2f} MB"
        )

        try:
            yield
        finally:
            self.current_memory -= size
            print(
                f"Releasing memory for {name}: current memory={self.current_memory/1024**3: 4.2f} GB size={size/1024**2: 4.2f} MB"
            )
            self.set()


async def download(url: str, dest_dir: Path, memory_manager: MemoryManager) -> Path:
    dest_path = dest_dir / url.split("/")[-1]
    if dest_path.is_file() and is_valid_geotiff(dest_path):
        print(f"Skipping {url}, already downloaded")
        return dest_path

    async with httpx.AsyncClient() as client, client.stream("GET", url) as response:
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


async def reproject_to_crs(
    src_tif: Path, dst_crs: rasterio.CRS, mem_limit: int = 0
) -> Path:
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


@dataclass
class RateLimiter:
    """Allows at most `frequency` operations every `time` seconds."""

    time: float
    frequency: int = 1

    _next_reset: int = 0
    _burst_count: int = 0

    async def wait(self):
        if self._burst_count < self.frequency:
            self._burst_count += 1
            return
        else:
            sleep_time = self._next_reset - time.perf_counter_ns() / 1e9
            if sleep_time > 0:
                print(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
            self._next_reset += int(self.time * 1e9)
            self._burst_count = 1
            return
        # if we have reached the burst limit, wait until the next reset time
        # and reset the burst count and the next reset time
        # otherwise, increment the burst count


async def main(src: Path, dst: Path, crs: rasterio.CRS, mem_limit: int = 2 * 1024**3):

    memory_manager = MemoryManager(mem_limit)
    limiter = RateLimiter(frequency=5, time=1)

    async def download_and_reproject(url: str) -> Path:
        # nonlocal current_memory_usage
        downloaded_file = None
        while downloaded_file is None:
            try:
                print("waiting for rate limit")
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

    mosaic, mosaic_transform = rasterio.merge.merge(
        reprojected_tifs, dtype="uint16", mem_limit=mem_limit // 1024**2
    )
    with rasterio.open(reprojected_tifs[0]) as src0:
        kwargs = src0.meta.copy()
    kwargs["driver"] = "GTiff"
    kwargs["height"] = mosaic.shape[1]
    kwargs["width"] = mosaic.shape[2]
    kwargs["transform"] = mosaic_transform

    output_tif = dst / "mosaic.tif"
    print(f"Writing merged raster to {output_tif}")
    with rasterio.open(output_tif, "w", **kwargs) as dest:
        dest.write(mosaic)


if __name__ == "__main__":
    import sys

    asyncio.run(
        main(
            Path(sys.argv[1]), Path(sys.argv[2]), rasterio.CRS.from_string(sys.argv[3])
        )
    )
