"""Per-field statistics and the over-watering / excess-fertiliser indicators.

Core idea: compare each field with the *other cropland in the same image on the same day*.
Same day ⇒ same weather, same sun angle, same atmosphere. Similar canopy (NDVI) ⇒ similar
expected leaf moisture (NDMI) and similar expected chlorophyll (NDRE). A field that is
consistently wetter, or consistently greener-per-leaf, than its look-alike neighbours is
the signal. These are screening indicators, not measurements of applied water or fertiliser.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np
from pyproj import Transformer
from rasterio.features import geometry_mask
from shapely.geometry import box, shape, mapping
from shapely.ops import transform as shp_transform

from .fetch import Grid

# ---- thresholds (tune per region / crop after ground-truthing) -------------------------
CROP_PRESENT_NDVI = 0.30     # field counts as "crop on ground" on a date
ACTIVE_GROWTH_NDVI = 0.50    # dates used for the nitrogen indicator
FULL_CANOPY_NDVI = 0.80      # NDVI near saturation: more N rarely adds biomass
MIN_FIELD_CLEAR = 0.70       # share of field pixels that must be cloud-free on a date
STANDING_WATER_SHARE = 0.10  # ≥10 % of field showing open water (NDWI > 0)
MIN_OBS = 4


@dataclass
class Field:
    id: str
    name: str
    geom: object            # shapely geometry in grid CRS
    geom_wgs84: object
    mask: np.ndarray        # bool [H, W]
    area_ha: float
    meta: dict = dc_field(default_factory=dict)


@dataclass
class Flag:
    level: str              # "high" | "watch" | "ok" | "insufficient"
    title: str
    detail: str
    action: str
    evidence: dict


@dataclass
class FieldResult:
    field: Field
    series: dict            # name -> np.ndarray[T] (NaN where field not observed)
    water: Flag
    nitrogen: Flag
    zones: np.ndarray | None  # int8 [H, W] 0 none, 1 low, 2 mid, 3 high vigour
    zone_spread: float


# ---- fields ----------------------------------------------------------------------------
def _to_grid(geom, grid: Grid):
    t = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    return shp_transform(t.transform, geom)


def _mask(geom, grid: Grid) -> np.ndarray:
    return geometry_mask([mapping(geom)], out_shape=(grid.height, grid.width),
                         transform=grid.transform, invert=True, all_touched=False)


def load_fields(path: Path, grid: Grid) -> list[Field]:
    fc = json.loads(Path(path).read_text())
    out = []
    for i, feat in enumerate(fc["features"], 1):
        props = feat.get("properties") or {}
        g84 = shape(feat["geometry"])
        g = _to_grid(g84, grid)
        m = _mask(g, grid)
        if m.sum() < 9:
            print(f"  skipping {props.get('name', i)}: smaller than ~9 Sentinel-2 pixels or outside the AOI")
            continue
        fid = str(props.get("id", i))
        out.append(Field(fid, str(props.get("name", f"Field {fid}")), g, g84, m, g.area / 1e4, props))
    return out


def cropland_mask(idx: dict, clear: np.ndarray) -> np.ndarray:
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        n_obs = clear.sum(0)
        ndvi_hi = np.nanpercentile(idx["NDVI"], 90, axis=0)
        ndwi_med = np.nanmedian(idx["NDWI"], axis=0)
    return (n_obs >= MIN_OBS) & (ndvi_hi >= 0.5) & (ndwi_med < 0)


def auto_plots(crop: np.ndarray, grid: Grid, size_m: float) -> list[Field]:
    """Demo mode: tile the AOI into square plots and keep the ones that are cropland."""
    res = grid.transform.a
    step = max(1, int(round(size_m / res)))
    to84 = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    plots = []
    for r0 in range(0, grid.height - step + 1, step):
        for c0 in range(0, grid.width - step + 1, step):
            if crop[r0:r0 + step, c0:c0 + step].mean() < 0.9:
                continue
            x0, y0 = grid.transform * (c0, r0 + step)
            x1, y1 = grid.transform * (c0 + step, r0)
            g = box(x0, y0, x1, y1)
            m = np.zeros(crop.shape, bool)
            m[r0:r0 + step, c0:c0 + step] = True
            pid = f"R{r0 // step:02d}C{c0 // step:02d}"
            plots.append(Field(pid, f"Plot {pid}", g, shp_transform(to84.transform, g), m, g.area / 1e4))
    return plots


# ---- peer reference --------------------------------------------------------------------
def peer_zscores(target: np.ndarray, ndvi: np.ndarray, crop: np.ndarray, min_px: int = 30):
    """Per date: expected `target` given NDVI (median by 0.05-NDVI bin over cropland),
    then a robust z-score of each pixel's departure from that expectation."""
    T = target.shape[0]
    z = np.full(target.shape, np.nan, np.float32)
    edges = np.arange(0.0, 1.0001, 0.05)
    centers = (edges[:-1] + edges[1:]) / 2
    for t in range(T):
        sel = crop & np.isfinite(target[t]) & np.isfinite(ndvi[t])
        if sel.sum() < 200:
            continue
        x, y = ndvi[t][sel], target[t][sel]
        b = np.digitize(x, edges) - 1
        med = np.array([np.median(y[b == k]) if (b == k).sum() >= min_px else np.nan
                        for k in range(len(centers))])
        ok = np.isfinite(med)
        if ok.sum() < 3:
            continue
        expected = np.interp(ndvi[t], centers[ok], med[ok]).astype(np.float32)
        resid = target[t] - expected
        r = resid[sel]
        scale = 1.4826 * np.median(np.abs(r - np.median(r))) or 1e-3
        z[t] = resid / scale
    return z


# ---- crop-calendar groups ---------------------------------------------------------------
def crop_groups(ndvi: np.ndarray, crop: np.ndarray, k: int = 6, iters: int = 25, seed: int = 0) -> np.ndarray:
    """Cluster cropland pixels by their season NDVI curve (k-means). Pixels in the same group share a
    crop calendar — a rough stand-in for crop type (e.g. sugarcane vs. short Rabi crops). -1 = not cropland."""
    T = ndvi.shape[0]
    X = ndvi[:, crop].T.astype(np.float32)                       # [N, T]
    good_dates = np.mean(np.isfinite(X), axis=0) > 0.8
    X = X[:, good_dates]
    col_mean = np.nanmean(X, axis=0)
    X = np.where(np.isfinite(X), X, col_mean)
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), 1)]
    for _ in range(1, k):                                          # k-means++ init
        d = np.min(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        C = np.vstack([C, X[rng.choice(len(X), p=d / d.sum())]])
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        C = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    out = np.full(crop.shape, -1, np.int8)
    out[crop] = lab
    return out


def group_zscores(target, ndvi, crop, groups, min_group_px=2000):
    """z-score against the pixel's own crop-calendar group (falls back to all cropland)."""
    z = peer_zscores(target, ndvi, crop)
    for g in np.unique(groups[groups >= 0]):
        sel = groups == g
        if sel.sum() < min_group_px:
            continue
        zg = peer_zscores(target, ndvi, sel)
        z[:, sel] = zg[:, sel]
    return z


# ---- per-field -------------------------------------------------------------------------
def _field_series(f: Field, idx, zm, zn, clear):
    m = f.mask
    T = clear.shape[0]
    s = {k: np.full(T, np.nan) for k in ("NDVI", "NDMI", "NDRE", "z_moist", "z_n", "water_share", "clear_share")}
    for t in range(T):
        ok = m & clear[t] & np.isfinite(idx["NDVI"][t])
        share = ok.sum() / m.sum()
        s["clear_share"][t] = share
        if share < MIN_FIELD_CLEAR:
            continue
        for k in ("NDVI", "NDMI", "NDRE"):
            s[k][t] = np.nanmean(idx[k][t][ok])
        s["z_moist"][t] = np.nanmean(zm[t][ok])
        s["z_n"][t] = np.nanmean(zn[t][ok])
        s["water_share"][t] = np.mean(idx["NDWI"][t][ok] > 0)
    return s


def _water_flag(s, dates) -> Flag:
    present = np.isfinite(s["NDVI"]) & (s["NDVI"] >= CROP_PRESENT_NDVI)
    n = int(present.sum())
    if n < MIN_OBS:
        return Flag("insufficient", "Not enough clear images", f"Only {n} usable cloud-free dates with a crop on the field.",
                    "Check again next season or add a cloud-tolerant source (Sentinel-1 radar).", {"n_obs": n})
    z = s["z_moist"][present]
    med_z = float(np.nanmedian(z))
    share_wet = float(np.mean(z > 1.0))
    wet_days = [d for d, w in zip(np.array(dates)[present], s["water_share"][present]) if w >= STANDING_WATER_SHARE]
    ev = {"n_obs": n, "median_z": round(med_z, 2), "share_wetter": round(share_wet, 2), "standing_water_dates": wet_days}

    if (med_z >= 1.0 and share_wet >= 0.5) or len(wet_days) >= 2:
        level, title = "high", "Likely over-watering"
    elif med_z >= 0.5 or share_wet >= 0.3 or len(wet_days) == 1:
        level, title = "watch", "Possibly wetter than needed"
    else:
        return Flag("ok", "Moisture in normal range",
                    f"Leaf moisture is in line with similar crops nearby on {n} clear dates.",
                    "No change needed.", ev)

    parts = [f"On {share_wet:.0%} of {n} clear dates this field held more moisture than nearby fields "
             f"with the same crop calendar and canopy (typical gap: {med_z:+.1f} standard units)."]
    if wet_days:
        parts.append(f"Standing water was visible on {len(wet_days)} date(s): {', '.join(wet_days[:4])}"
                     f"{'…' if len(wet_days) > 4 else ''}.")
    action = ("Try lengthening the interval between irrigations by 1–2 days, or shortening each irrigation, "
              "and check the field drains well. If on flood irrigation, consider furrow/alternate-furrow or drip. "
              "Confirm with a hand soil-feel test (root zone, 15–30 cm) before each irrigation.")
    return Flag(level, title, " ".join(parts), action, ev)


def _nitrogen_flag(s) -> Flag:
    grow = np.isfinite(s["NDVI"]) & (s["NDVI"] >= ACTIVE_GROWTH_NDVI)
    n = int(grow.sum())
    if n < MIN_OBS:
        return Flag("insufficient", "Not enough active-growth images",
                    f"Only {n} clear dates while the crop was actively growing.", "—", {"n_obs": n})
    peak = float(np.nanpercentile(s["NDVI"][grow], 90))
    zn = float(np.nanmedian(s["z_n"][grow]))
    share_hi = float(np.mean(s["z_n"][grow] > 1.0))
    ev = {"n_obs": n, "peak_ndvi": round(peak, 2), "median_ndre_z": round(zn, 2), "share_greener": round(share_hi, 2)}
    full = peak >= FULL_CANOPY_NDVI

    if full and zn >= 1.0:
        level, title = "high", "Fertiliser (nitrogen) may be more than the crop can use"
    elif (full and zn >= 0.5) or zn >= 1.0:
        level, title = "watch", "Nitrogen looks generous"
    else:
        return Flag("ok", "No sign of excess nitrogen",
                    f"Leaf chlorophyll is in line with similar crops nearby (peak NDVI {peak:.2f}).",
                    "Keep following soil-test based doses.", ev)
    detail = (f"Peak canopy greenness (NDVI {peak:.2f}) is {'already at full cover' if full else 'good'}, and leaf "
              f"chlorophyll (red-edge, NDRE) was above similar crops nearby on {share_hi:.0%} of {n} growth-stage "
              f"dates (typical gap {zn:+.1f} standard units). " +
              ("Very green leaves on a full canopy usually mean nitrogen is not the limiting factor."
               if full else "Leaves greener than look-alike neighbours suggest nitrogen is not in short supply."))
    action = ("Before the next top-dressing, check a leaf colour chart or get a Soil Health Card test. If leaves "
              "already match the darkest panels, skip or reduce the urea dose (e.g. by 25%). Excess nitrogen "
              "costs money, attracts pests, causes lodging, and leaches into groundwater.")
    return Flag(level, title, detail, action, ev)


def _zones(f: Field, ndvi, s):
    grow = np.isfinite(s["NDVI"]) & (s["NDVI"] >= ACTIVE_GROWTH_NDVI)
    if grow.sum() < 2 or f.mask.sum() < 30:
        return None, 0.0
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_ndvi = np.nanmean(ndvi[grow], axis=0)
    vals = mean_ndvi[f.mask & np.isfinite(mean_ndvi)]
    if vals.size < 30:
        return None, 0.0
    lo, hi = np.percentile(vals, [33.3, 66.7])
    zones = np.zeros(ndvi.shape[1:], np.int8)
    inside = f.mask & np.isfinite(mean_ndvi)
    zones[inside & (mean_ndvi < lo)] = 1
    zones[inside & (mean_ndvi >= lo) & (mean_ndvi < hi)] = 2
    zones[inside & (mean_ndvi >= hi)] = 3
    spread = float(np.mean(vals[vals >= hi]) - np.mean(vals[vals < lo]))
    return zones, spread


@dataclass
class Analysis:
    dates: list[str]
    idx: dict
    crop: np.ndarray
    z_moist: np.ndarray
    z_n: np.ndarray
    peer: dict              # group id (or "all") -> per-date percentiles for charts
    groups: np.ndarray
    results: list[FieldResult]


def drop_hazy_dates(stack, idx, log=print, drop=0.08):
    """Thin haze often slips past the cloud mask and pulls NDVI down across the whole scene.
    Drop dates whose cropland-median NDVI dips well below the surrounding dates."""
    crop = cropland_mask(idx, stack.clear)
    with np.errstate(all="ignore"):
        med = np.array([np.nanmedian(idx["NDVI"][t][crop]) if (crop & stack.clear[t]).mean() > 0.02 else np.nan
                        for t in range(len(stack.dates))])
    bad = []
    for t in range(len(med)):
        if not np.isfinite(med[t]):
            continue
        nb = [med[j] for j in range(max(0, t - 2), min(len(med), t + 3)) if j != t and np.isfinite(med[j])]
        if len(nb) >= 2 and med[t] < np.median(nb) - drop:
            bad.append(t)
    for t in bad:
        stack.clear[t] = False
        for k in idx:
            idx[k][t] = np.nan
    if bad:
        log(f"Dropped {len(bad)} hazy date(s): {', '.join(stack.dates[t] for t in bad)}")


def analyse(stack, idx, fields_path: Path | None, plot_size_m: float, log=print) -> Analysis:
    drop_hazy_dates(stack, idx, log)
    crop = cropland_mask(idx, stack.clear)
    log(f"Cropland pixels: {crop.sum():,} ({crop.mean():.0%} of area)")
    groups = crop_groups(idx["NDVI"], crop)
    sizes = {int(g): int((groups == g).sum()) for g in np.unique(groups[groups >= 0])}
    log(f"Crop-calendar groups (pixels): {sizes}")
    zm = group_zscores(idx["NDMI"], idx["NDVI"], crop, groups)
    zn = group_zscores(idx["NDRE"], idx["NDVI"], crop, groups)

    fields = load_fields(fields_path, stack.grid) if fields_path else auto_plots(crop, stack.grid, plot_size_m)
    log(f"Analysing {len(fields)} {'fields' if fields_path else 'auto-generated plots'}")

    def percentiles(sel):
        out = {}
        for k in ("NDVI", "NDMI"):
            a = idx[k][:, sel]
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[k] = np.nanpercentile(a, [25, 50, 75], axis=1)
        out["n_px"] = int(sel.sum())
        return out

    peer = {"all": percentiles(crop)}
    for g, n in sizes.items():
        if n >= 2000:
            peer[g] = percentiles(groups == g)

    results = []
    for f in fields:
        g = groups[f.mask]
        g = g[g >= 0]
        f.meta["group"] = int(np.bincount(g).argmax()) if g.size else None
        s = _field_series(f, idx, zm, zn, stack.clear)
        zones, spread = _zones(f, idx["NDVI"], s)
        results.append(FieldResult(f, s, _water_flag(s, stack.dates), _nitrogen_flag(s), zones, spread))
    return Analysis(stack.dates, idx, crop, zm, zn, peer, groups, results)
