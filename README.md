# Farm water & fertiliser check (Sentinel-2, Maharashtra)

Uses free Sentinel-2 satellite imagery to compute vegetation (NDVI), moisture (NDMI) and red-edge
chlorophyll (NDRE) indices over farmland, and produces plain-language reports that point farmers to
fields that are probably **over-watered** or getting **more nitrogen than the crop can use**.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run.py configs/baramati_demo.toml                       # demo: auto 1-ha plots
.venv/bin/python run.py configs/baramati_demo.toml \
    --fields examples/sample_fields.geojson --out reports/example_fields  # real field boundaries
open reports/baramati_demo_rabi_2025_26/index.html
```

The first run downloads imagery (~1–2 min, no account needed: Microsoft Planetary Computer) and caches it in `cache/`.

## Outputs (`reports/<name>/`)

| File | What |
|---|---|
| `index.html` | Area summary: counts, overview map, table of flagged fields |
| `fields/<id>.html` | One-page farmer report: flags, what to try, maps, season charts. Self-contained, so it can be printed or sent on WhatsApp |
| `fields.csv` | Every field's flags and numbers |
| `fields_flagged.geojson` | Field outlines with flags (QGIS / Google Earth) |

## Your own area / farms

Copy `configs/baramati_demo.toml`, change `name`, `aoi` (lon/lat box) and the season dates.
For real farms, draw boundaries on [geojson.io](https://geojson.io) (or use cadastral / 7/12 maps), save as
GeoJSON with `id` and `name` properties, and pass `--fields`. Fields smaller than ~0.1 ha (about 9 pixels) are skipped.

## How it works

1. **Imagery**: Sentinel-2 L2A bands B02–B05, B08, B8A, B11 + scene classification (cloud/shadow mask), warped to one
   10 m grid. Scenes with thin haze that the mask misses are detected and dropped.
2. **Indices**: NDVI (B08/B04) for canopy vigour, NDMI (B08/B11) for leaf water, NDRE (B8A/B05) for chlorophyll/nitrogen, NDWI (B03/B08) for standing water.
3. **Cropland & crop groups**: pixels are clustered (k-means) by their season NDVI curve, a rough stand-in for crop
   type, so sugarcane is compared with sugarcane-like fields and short Rabi crops with Rabi crops.
4. **Peer comparison**: for each date and crop group, expected NDMI and NDRE are estimated *given NDVI* from all
   cropland in the scene. Each pixel's departure is a robust z-score. Same day means same weather and atmosphere.
5. **Flags** (`agrisat/analysis.py`, thresholds at the top):
   - *Over-watering*: field is consistently wetter than look-alike neighbours (median z ≥ 1 on ≥ 50 % of dates), or
     standing water appears on ≥ 2 dates. "Watch" at weaker evidence.
   - *Excess nitrogen*: full canopy (NDVI ≥ 0.8) **and** chlorophyll above look-alike neighbours (median NDRE z ≥ 1).
   - *Uneven field*: within-field vigour zones. Suggests variable-rate fertiliser instead of uniform spreading.

## Limits — read before showing farmers

- These are **screening indicators**, not measurements of water or fertiliser applied. Variety, sowing date,
  soil, salinity, intercropping and rain can all produce the same signals. Ground-truth before advising.
- Fertiliser *overuse* can't be seen directly from space. The tool flags where nitrogen is very likely **not
  limiting**, which is where extra urea is most likely wasted.
- Thresholds are uncalibrated. Validate against a handful of farms with known irrigation schedules and soil tests,
  then tune the constants in `analysis.py`.
- 10 m pixels: field edges mix with bunds, roads and neighbours. Small strip fields are noisy.

## Ideas for next steps

- Marathi translation of the farmer reports.
- Sentinel-1 radar soil moisture (sees through clouds, so it works in Kharif).
- Rainfall (IMD / CHIRPS) and evapotranspiration to tell rain apart from irrigation.
- Crop-type labels from farmers to replace the k-means proxy.
