"""Self-contained HTML reports (images embedded) — one summary page plus one page per field."""
from __future__ import annotations

import base64
import csv
import html
import io
import json
import warnings
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from shapely.geometry import mapping

# ---- palette (dataviz reference palette) -----------------------------------------------
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"
FIELD_C = "#2a78d6"
STATUS = {"high": "#d03b3b", "watch": "#fab219", "ok": "#0ca30c", "insufficient": "#898781"}
RANK = {"high": 3, "watch": 2, "ok": 1, "insufficient": 0}
CHIP = {"high": ("▲", "Act now"), "watch": ("!", "Watch"), "ok": ("✓", "OK"), "insufficient": ("?", "Not enough data")}

MOIST_CMAP = LinearSegmentedColormap.from_list("moist", ["#c23b3a", "#e88a85", "#f0efec", "#86b6ef", "#1c5cab"])
VIGOUR_CMAP = LinearSegmentedColormap.from_list("vig", ["#f3f1e6", "#b9d99a", "#5fae4a", "#1f7a1f", "#0b4a12"])
ZONE_CMAP = ListedColormap(["#d7e9bf", "#7dbb5c", "#2a7a2a"])

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _rgb(stack, t, rows=slice(None), cols=slice(None)):
    rgb = np.dstack([stack.bands[b][t][rows, cols] for b in ("B04", "B03", "B02")])
    lo, hi = np.nanpercentile(rgb, 2), np.nanpercentile(rgb, 98)
    return np.clip(np.nan_to_num((rgb - lo) / (hi - lo + 1e-6)), 0, 1) ** 0.8


def best_photo_date(stack, A) -> int:
    clear_share = stack.clear.mean((1, 2))
    score = np.where(clear_share > 0.95, np.nan_to_num(A.peer["all"]["NDVI"][1]), -1)
    return int(np.argmax(score))


def _outline(ax, geom, grid, color, lw=1.2, r0=0, c0=0):
    inv = ~grid.transform
    polys = geom.geoms if hasattr(geom, "geoms") else [geom]
    for p in polys:
        xs, ys = p.exterior.xy
        cr = [inv * (x, y) for x, y in zip(xs, ys)]
        ax.plot([c - 0.5 - c0 for c, _ in cr], [r - 0.5 - r0 for _, r in cr], color=color, lw=lw)


def _season_layers(A):
    grow = A.idx["NDVI"] >= 0.3
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ndvi_peak = np.nanpercentile(A.idx["NDVI"], 90, axis=0)
        z_moist = np.nanmedian(np.where(grow, A.z_moist, np.nan), axis=0)
    return ndvi_peak, z_moist


def _peer(res, A):
    return A.peer.get(res.field.meta.get("group"), A.peer["all"])


# ---- charts ----------------------------------------------------------------------------
def chart_timeseries(res, A) -> str:
    d = [date.fromisoformat(x) for x in A.dates]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    specs = [("NDVI", "Crop greenness (NDVI)", (0, 1)), ("NDMI", "Leaf moisture (NDMI)", None)]
    for ax, (k, title, ylim) in zip(axes, specs):
        p25, p50, p75 = _peer(res, A)[k]
        ok = np.isfinite(p50)
        dd = [x for x, o in zip(d, ok) if o]
        ax.fill_between(dd, p25[ok], p75[ok], color=GRID, lw=0, label="Nearby farms (middle half)")
        ax.plot(dd, p50[ok], color=MUTED, lw=1.5, ls=(0, (4, 2)), label="Nearby farms (typical)")
        s = res.series[k]
        f = np.isfinite(s)
        ax.plot([x for x, o in zip(d, f) if o], s[f], color=FIELD_C, lw=2, marker="o", ms=4,
                markeredgecolor=SURFACE, markeredgewidth=1, label="This field")
        if k == "NDMI":
            w = f & (res.series["water_share"] >= 0.10)
            if w.any():
                ax.scatter([x for x, o in zip(d, w) if o], s[w], s=70, marker="v", color=STATUS["high"],
                           zorder=5, edgecolor=SURFACE, lw=1, label="Standing water seen")
        ax.set_title(title, loc="left", color=INK, fontsize=10, fontweight="bold")
        ax.grid(axis="y", color=GRID, lw=0.6)
        if ylim:
            ax.set_ylim(*ylim)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    handles, labels = axes[1].get_legend_handles_labels()
    order = [labels.index(l) for l in ("This field", "Nearby farms (typical)", "Nearby farms (middle half)") if l in labels]
    order += [i for i, l in enumerate(labels) if i not in order]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=8.5, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _png(fig)


def chart_field_maps(res, A, stack, photo_t, ndvi_peak, z_moist) -> str:
    grid = stack.grid
    rr, cc = np.where(res.field.mask)
    pad = max(4, int(0.4 * max(np.ptp(rr), np.ptp(cc))))
    r0, r1 = max(0, rr.min() - pad), min(grid.height, rr.max() + pad + 1)
    c0, c1 = max(0, cc.min() - pad), min(grid.width, cc.max() + pad + 1)
    rs, cs = slice(r0, r1), slice(c0, c1)

    panels = [("Satellite photo\n" + A.dates[photo_t], "rgb"),
              ("Crop vigour (peak NDVI)", "ndvi"),
              ("Moisture vs. nearby farms", "moist")]
    if res.zones is not None:
        panels.append(("Vigour zones in this field", "zones"))
    fig, axes = plt.subplots(1, len(panels), figsize=(2.3 * len(panels), 2.9))
    for ax, (title, kind) in zip(axes, panels):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if kind == "rgb":
            ax.imshow(_rgb(stack, photo_t, rs, cs), interpolation="nearest")
            oc = "#ffffff"
        elif kind == "ndvi":
            im = ax.imshow(ndvi_peak[rs, cs], cmap=VIGOUR_CMAP, vmin=0.2, vmax=0.9, interpolation="nearest")
            cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04, ticks=[0.2, 0.55, 0.9])
            cb.ax.set_xticklabels(["low", "", "high"]); oc = INK
        elif kind == "moist":
            im = ax.imshow(z_moist[rs, cs], cmap=MOIST_CMAP, vmin=-2.5, vmax=2.5, interpolation="nearest")
            cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04, ticks=[-2.5, 0, 2.5])
            cb.ax.set_xticklabels(["drier", "same", "wetter"]); oc = INK
        else:
            z = np.where(res.zones[rs, cs] > 0, res.zones[rs, cs], np.nan)
            im = ax.imshow(z, cmap=ZONE_CMAP, vmin=0.5, vmax=3.5, interpolation="nearest")
            cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04, ticks=[1, 2, 3])
            cb.ax.set_xticklabels(["weaker", "middle", "stronger"]); oc = INK
        if kind != "rgb":
            cb.ax.tick_params(labelsize=7.5, length=0); cb.outline.set_visible(False)
        _outline(ax, res.field.geom, grid, oc, 1.4, r0, c0)
        ax.set_xlim(-0.5, c1 - c0 - 0.5); ax.set_ylim(r1 - r0 - 0.5, -0.5)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    fig.tight_layout()
    return _png(fig)


def chart_overview(A, stack, photo_t, z_moist) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    ax = axes[0]
    ax.imshow(_rgb(stack, photo_t), interpolation="nearest")
    for level in ("ok", "watch", "high"):
        for r in A.results:
            if max(RANK[r.water.level], RANK[r.nitrogen.level]) == RANK[level] and level != "ok":
                _outline(ax, r.field.geom, stack.grid, STATUS[level], 1.3)
    ax.set_title(f"Flagged fields on {A.dates[photo_t]} photo", loc="left", fontsize=10, color=INK)
    ax.plot([], [], color=STATUS["high"], lw=2, label="▲ Act now")
    ax.plot([], [], color=STATUS["watch"], lw=2, label="! Watch")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    ax2 = axes[1]
    show = np.where(A.crop, z_moist, np.nan)
    im = ax2.imshow(show, cmap=MOIST_CMAP, vmin=-2.5, vmax=2.5, interpolation="nearest")
    ax2.set_facecolor("#e9e8e3")
    ax2.set_title("Season moisture vs. similar crops (cropland only)", loc="left", fontsize=10, color=INK)
    cb = fig.colorbar(im, ax=ax2, fraction=0.035, pad=0.02, ticks=[-2.5, 0, 2.5])
    cb.ax.set_yticklabels(["drier", "same", "wetter"]); cb.outline.set_visible(False)
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    return _png(fig)


# ---- HTML ------------------------------------------------------------------------------
CSS = """
:root{color-scheme:light;--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--line:#e1e0d9;--accent:#2a78d6;--high:#d03b3b;--watch:#fab219;--ok:#0ca30c;--na:#898781}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#0d0d0d;--card:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--line:#2c2c2a;--accent:#3987e5}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#0d0d0d;--card:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#2c2c2a;--accent:#3987e5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:900px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 10px}.sub{color:var(--ink2);margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:12px 0}
.card h3{margin:0 0 6px;font-size:16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.chip{display:inline-flex;gap:6px;align-items:center;font-size:12.5px;font-weight:600;padding:2px 10px;border-radius:99px;
color:#fff;white-space:nowrap}.chip.high{background:var(--high)}.chip.watch{background:var(--watch);color:#0b0b0b}
.chip.ok{background:var(--ok)}.chip.insufficient{background:var(--na)}
.action{border-left:3px solid var(--accent);padding:6px 12px;margin-top:10px;color:var(--ink)}
.action b{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--ink2)}
img{width:100%;height:auto;border-radius:8px;background:#fcfcfb}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}
th{color:var(--ink2);font-weight:600;font-size:12.5px}td.num{font-variant-numeric:tabular-nums}
.tablewrap{overflow-x:auto}a{color:var(--accent)}.note{color:var(--ink2);font-size:13.5px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.stat .v{font-size:26px;font-weight:700}.stat .l{color:var(--ink2);font-size:13px}
@media print{body{background:#fff}.card{break-inside:avoid}}
"""


def _chip(level):
    icon, label = CHIP[level]
    return f'<span class="chip {level}">{icon} {label}</span>'


def _page(title, body):
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>'
            f"<style>{CSS}</style></head><body><main>{body}</main></body></html>")


def _flag_card(heading, flag):
    e = lambda s: html.escape(s)
    action = "" if flag.level in ("ok", "insufficient") else f'<div class="action"><b>What to try</b>{e(flag.action)}</div>'
    return (f'<div class="card"><h3>{e(heading)} {_chip(flag.level)}</h3>'
            f'<div><b>{e(flag.title)}.</b> {e(flag.detail)}</div>{action}</div>')


CAVEAT = ("These are satellite-based <b>indicators</b>, not direct measurements of how much water or fertiliser was "
          "applied. A field is compared with other fields in the same image on the same day that follow the same "
          "season-long greenness pattern (a stand-in for crop type) and have similar leaf cover. Crop variety, "
          "sowing date, soil type, intercropping and rainfall can still explain differences. Please confirm "
          "on the ground (soil feel test, leaf colour chart, Soil Health Card) before changing practice. Sentinel-2 "
          "pixels are 10 m, so very small or narrow fields are less reliable.")

def field_page(res, A, stack, cfg, photo_t, ndvi_peak, z_moist) -> str:
    f = res.field
    c = f.geom_wgs84.centroid
    zones_html = ""
    if res.zones is not None and res.zone_spread >= 0.08:
        zones_html = (f'<div class="card"><h3>Uneven growth inside the field</h3>The strongest third of this field is '
                      f'{res.zone_spread:.2f} NDVI points greener than the weakest third (see “Vigour zones” map). '
                      f'<div class="action"><b>What to try</b>Don\'t spread fertiliser evenly: give the dark-green '
                      f'“stronger” zone less and check the pale “weaker” zone for waterlogging, salinity, pests or poor '
                      f'germination first, since more fertiliser won\'t fix those.</div></div>')
    body = f"""
<h1>{html.escape(f.name)}</h1>
<p class="sub">{html.escape(cfg['name'])} · {f.area_ha:.2f} ha · centre {c.y:.5f}° N, {c.x:.5f}° E ·
season {cfg['season_start']} to {cfg['season_end']}</p>
{_flag_card("Water use", res.water)}
{_flag_card("Fertiliser (nitrogen)", res.nitrogen)}
{zones_html}
<h2>Maps</h2>
<img alt="Maps of field: satellite photo, crop vigour, moisture compared to neighbours, vigour zones"
 src="{chart_field_maps(res, A, stack, photo_t, ndvi_peak, z_moist)}">
<h2>Through the season</h2>
<img alt="Time series of NDVI and NDMI for this field compared with nearby farms"
 src="{chart_timeseries(res, A)}">
<p class="note">Blue line: this field. Grey: the {_peer(res, A)["n_px"] // 100:,} or so hectares of nearby cropland
whose greenness rose and fell on the same calendar as this field (most likely the same crop).
NDVI shows how green and dense the crop is; NDMI shows how much water the leaves hold.</p>
<h2>About this report</h2><p class="note">{CAVEAT}</p>
<p class="note">Imagery: Copernicus Sentinel-2 L2A ({len(A.dates)} dates), via Microsoft Planetary Computer.
Generated {date.today().isoformat()}.</p>"""
    return _page(f.name, body)


def write_reports(A, stack, cfg, out_dir: Path, max_pages: int, user_fields: bool, log=print):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fields").mkdir(exist_ok=True)
    photo_t = best_photo_date(stack, A)
    ndvi_peak, z_moist = _season_layers(A)

    def sev(r):
        return (max(RANK[r.water.level], RANK[r.nitrogen.level]), RANK[r.water.level] + RANK[r.nitrogen.level],
                r.water.evidence.get("median_z", 0))
    ranked = sorted(A.results, key=sev, reverse=True)
    if user_fields:
        page_for = ranked
    else:
        page_for = [r for r in ranked if sev(r)[0] >= RANK["watch"]][:max_pages]

    pages = {}
    for i, r in enumerate(page_for, 1):
        fn = f"fields/{r.field.id}.html"
        (out_dir / fn).write_text(field_page(r, A, stack, cfg, photo_t, ndvi_peak, z_moist))
        pages[r.field.id] = fn
        log(f"  report {i}/{len(page_for)}: {fn}")

    # machine-readable outputs
    with open(out_dir / "fields.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "name", "area_ha", "water_flag", "water_median_z", "water_share_wetter", "standing_water_dates",
                    "nitrogen_flag", "peak_ndvi", "ndre_median_z", "zone_spread", "lat", "lon"])
        for r in ranked:
            we, ne, c = r.water.evidence, r.nitrogen.evidence, r.field.geom_wgs84.centroid
            w.writerow([r.field.id, r.field.name, f"{r.field.area_ha:.2f}", r.water.level, we.get("median_z"),
                        we.get("share_wetter"), ";".join(we.get("standing_water_dates", [])), r.nitrogen.level,
                        ne.get("peak_ndvi"), ne.get("median_ndre_z"), f"{r.zone_spread:.3f}", f"{c.y:.6f}", f"{c.x:.6f}"])
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(r.field.geom_wgs84),
         "properties": {"id": r.field.id, "name": r.field.name, "water": r.water.level, "nitrogen": r.nitrogen.level}}
        for r in ranked]}
    (out_dir / "fields_flagged.geojson").write_text(json.dumps(gj))

    # summary page
    count = lambda attr, lvl: sum(getattr(r, attr).level == lvl for r in A.results)
    rows = []
    for r in ranked:
        if r.field.id not in pages:
            continue
        rows.append(f'<tr><td><a href="{pages[r.field.id]}">{html.escape(r.field.name)}</a></td>'
                    f'<td class="num">{r.field.area_ha:.2f}</td><td>{_chip(r.water.level)}</td>'
                    f'<td>{_chip(r.nitrogen.level)}</td>'
                    f'<td class="num">{r.water.evidence.get("median_z", "–")}</td>'
                    f'<td class="num">{r.nitrogen.evidence.get("median_ndre_z", "–")}</td></tr>')
    kind = "fields" if user_fields else "1-ha plots"
    body = f"""
<h1>{html.escape(cfg['name'])}: water &amp; fertiliser check</h1>
<p class="sub">Sentinel-2, {cfg['season_start']} to {cfg['season_end']} · {len(A.dates)} image dates ·
{len(A.results)} {kind} analysed</p>
<div class="stats">
<div class="stat"><div class="v">{count('water', 'high')}</div><div class="l">▲ likely over-watering</div></div>
<div class="stat"><div class="v">{count('water', 'watch')}</div><div class="l">! wetter than needed (watch)</div></div>
<div class="stat"><div class="v">{count('nitrogen', 'high')}</div><div class="l">▲ likely excess nitrogen</div></div>
<div class="stat"><div class="v">{count('nitrogen', 'watch')}</div><div class="l">! generous nitrogen (watch)</div></div>
</div>
<h2>Area overview</h2>
<img alt="Overview map of flagged fields and season moisture anomaly" src="{chart_overview(A, stack, photo_t, z_moist)}">
<h2>{'Fields' if user_fields else f'Flagged plots (top {len(rows)})'}</h2>
<div class="tablewrap"><table><thead><tr><th>Field</th><th>ha</th><th>Water</th><th>Nitrogen</th>
<th>Moisture gap</th><th>Chlorophyll gap</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="note">“Gap” columns are the season median difference from similar crops nearby, in standard units
(0 = typical, +1 = clearly above). Full results for every {kind[:-1]}: <code>fields.csv</code> and
<code>fields_flagged.geojson</code> (open in QGIS or Google Earth).</p>
<h2>How to read this</h2><p class="note">{CAVEAT}</p>"""
    (out_dir / "index.html").write_text(_page(cfg["name"], body))
    log(f"Summary: {out_dir / 'index.html'}")
