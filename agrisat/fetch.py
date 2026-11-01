"""Download Sentinel-2 L2A bands for an area of interest from Microsoft Planetary Computer.

Every scene is warped onto one fixed 10 m UTM grid, so all dates line up pixel-for-pixel.
Results are cached to disk as a compressed .npz so re-running a report is instant.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
REFLECTANCE_BANDS = ["B02", "B03", "B04", "B05", "B08", "B8A", "B11"]
# Scene Classification Layer values we trust: 4 vegetation, 5 bare soil, 6 water.
SCL_CLEAR = (4, 5, 6)

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_MULTIRANGE="YES",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    GDAL_HTTP_MAX_RETRY="4",
    GDAL_HTTP_RETRY_DELAY="2",
    VSI_CACHE="TRUE",
)


@dataclass
class Grid:
    crs: str
    transform: rasterio.Affine
    width: int
    height: int


@dataclass
class Stack:
    """Time stack of surface reflectance on a common grid."""
    grid: Grid
    dates: list[str]                 # ISO dates, sorted
    bands: dict[str, np.ndarray]     # band -> float32 [T, H, W] reflectance (0-1), NaN where invalid
    clear: np.ndarray                # bool [T, H, W] cloud/shadow-free


def make_grid(bbox: list[float], epsg: int, res: float = 10.0) -> Grid:
    """Snap a lon/lat bbox to a res-metre grid in the given UTM CRS."""
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    xs, ys = to_utm.transform([bbox[0], bbox[2], bbox[0], bbox[2]], [bbox[1], bbox[1], bbox[3], bbox[3]])
    left, right = np.floor(min(xs) / res) * res, np.ceil(max(xs) / res) * res
    bottom, top = np.floor(min(ys) / res) * res, np.ceil(max(ys) / res) * res
    return Grid(f"EPSG:{epsg}", from_origin(left, top, res, res),
                int(round((right - left) / res)), int(round((top - bottom) / res)))


def search(bbox, start, end, max_cloud):
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    items = catalog.search(
        collections=["sentinel-2-l2a"], bbox=bbox, datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    ).item_collection()
    return sorted(items, key=lambda it: it.datetime)


def _boa_offset(item) -> float:
    # Processing baseline 04.00+ (Jan 2022 onward) stores reflectance with a +1000 DN offset.
    baseline = item.properties.get("s2:processing_baseline", "00.00")
    return -1000.0 if float(baseline) >= 4.0 else 0.0


def _read_asset(href: str, grid: Grid, resampling) -> np.ndarray:
    with rasterio.Env(**GDAL_ENV), rasterio.open(href) as src, WarpedVRT(
        src, crs=grid.crs, transform=grid.transform, width=grid.width, height=grid.height,
        resampling=resampling, src_nodata=0, nodata=0,
    ) as vrt:
        return vrt.read(1)


def _read_item(item, grid: Grid):
    item = planetary_computer.sign(item)
    offset = _boa_offset(item)
    out = {}
    for band in REFLECTANCE_BANDS:
        dn = _read_asset(item.assets[band].href, grid, Resampling.bilinear).astype(np.float32)
        refl = (dn + offset) / 10000.0
        refl[dn == 0] = np.nan
        out[band] = refl
    scl = _read_asset(item.assets["SCL"].href, grid, Resampling.nearest)
    out["clear"] = np.isin(scl, SCL_CLEAR)
    return item.datetime.date().isoformat(), out


def _cache_key(cfg: dict) -> str:
    keys = {k: cfg[k] for k in ("aoi", "season_start", "season_end", "max_cloud")}
    return hashlib.sha1(json.dumps(keys, sort_keys=True).encode()).hexdigest()[:12]


def load_stack(cfg: dict, cache_dir: Path, workers: int = 6, log=print) -> Stack:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"stack_{_cache_key(cfg)}.npz"
    if cache.exists():
        log(f"Using cached imagery: {cache}")
        z = np.load(cache, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        grid = Grid(meta["crs"], rasterio.Affine(*meta["transform"]), meta["width"], meta["height"])
        bands = {b: z[b].astype(np.float32) / 10000.0 for b in REFLECTANCE_BANDS}
        for b in bands.values():
            b[b < -0.5] = np.nan
        return Stack(grid, meta["dates"], bands, z["clear"])

    items = search(cfg["aoi"], cfg["season_start"], cfg["season_end"], cfg["max_cloud"])
    if not items:
        raise SystemExit("No Sentinel-2 scenes found for this area/season. Try a larger max_cloud or another season.")
    epsg = int(str(items[0].properties.get("proj:code") or f"EPSG:{items[0].properties['proj:epsg']}").split(":")[1])
    grid = make_grid(cfg["aoi"], epsg)
    log(f"Found {len(items)} scenes; grid {grid.width}x{grid.height} px at 10 m ({grid.crs}). Downloading…")

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, res in enumerate(pool.map(lambda it: _read_item(it, grid), items), 1):
            results.append(res)
            log(f"  [{i}/{len(items)}] {res[0]}")

    # Merge scenes captured on the same day (tile overlaps / reprocessed duplicates).
    by_date: dict[str, dict] = {}
    for date, data in results:
        valid = data["clear"] & ~np.isnan(data["B08"])
        if date not in by_date:
            by_date[date] = {k: v.copy() for k, v in data.items()}
            by_date[date]["clear"] = valid
            continue
        merged = by_date[date]
        fill = valid & ~merged["clear"]
        for b in REFLECTANCE_BANDS:
            merged[b][fill] = data[b][fill]
        merged["clear"] |= fill

    dates = sorted(by_date)
    bands = {b: np.stack([by_date[d][b] for d in dates]) for b in REFLECTANCE_BANDS}
    clear = np.stack([by_date[d]["clear"] for d in dates])
    for b in bands.values():
        b[~clear] = np.nan

    meta = dict(crs=grid.crs, transform=list(grid.transform)[:6], width=grid.width, height=grid.height, dates=dates)
    np.savez_compressed(
        cache, meta=json.dumps(meta), clear=clear,
        **{b: np.nan_to_num(v * 10000.0, nan=-32768).astype(np.int16) for b, v in bands.items()},
    )
    log(f"Cached imagery to {cache}")
    return Stack(grid, dates, bands, clear)
