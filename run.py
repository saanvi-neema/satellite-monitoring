#!/usr/bin/env python3
"""Run the water & fertiliser check for one area/season.

    .venv/bin/python run.py configs/baramati_demo.toml
"""
import argparse
import re
import tomllib
from pathlib import Path

from agrisat import analysis, fetch, indices, report

ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--fields", type=Path, help="GeoJSON of field boundaries (overrides config)")
    ap.add_argument("--out", type=Path, help="output folder (default reports/<config name>)")
    args = ap.parse_args()

    cfg = tomllib.loads(args.config.read_text())
    cfg.setdefault("max_cloud", 30)
    cfg.setdefault("plot_size_m", 100)
    cfg.setdefault("max_reports", 25)
    fields = args.fields or (Path(cfg["fields"]) if cfg.get("fields") else None)
    if fields and not fields.is_absolute():
        fields = (args.config.parent / fields) if not fields.exists() else fields
    slug = re.sub(r"[^a-z0-9]+", "_", cfg["name"].lower()).strip("_")
    out = args.out or ROOT / "reports" / slug

    stack = fetch.load_stack(cfg, ROOT / "cache")
    idx = indices.compute(stack.bands)
    A = analysis.analyse(stack, idx, fields, cfg["plot_size_m"])
    report.write_reports(A, stack, cfg, out, cfg["max_reports"], user_fields=fields is not None)


if __name__ == "__main__":
    main()
