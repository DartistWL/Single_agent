"""
Merge manually downloaded EirGrid ROI CO2 JSON files (e.g. 12 monthly files)
into one CSV usable by evaluation/strategy_carbon_comparison.py.

- Parses EffectiveTime as Europe/Dublin (EirGrid local), converts to UTC
- Builds a strict 30-minute UTC grid (aligned with HEMS steps_per_day=48):
  each row is the mean of the two 15-minute samples in that half-hour
- Missing 15-minute values: linear time interpolation on the 15-minute series, then resample
- Any remaining missing 30-minute slots: interpolate in time, then (prev+next)/2, then ffill/bfill

Output columns: timestamp_utc, carbon_intensity_gco2_per_kwh, region

Usage:
  python data/merge_eirgrid_roi_co2_json.py --dir data/carbon_2020
"""
from __future__ import annotations

import argparse
import json
import os
from glob import glob
from typing import Any, Dict, List

import pandas as pd


def _rows_from_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("Status") != "Success":
        raise SystemExit(f"{path}: Status is not Success: {data!r}")
    rows = data.get("Rows") or []
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "effective_time": r.get("EffectiveTime"),
                "carbon_intensity_gco2_per_kwh": r.get("Value"),
                "field_name": r.get("FieldName"),
                "region": r.get("Region"),
            }
        )
    return out


def _localize_effective_to_utc(series: pd.Series) -> pd.DatetimeIndex:
    dt = pd.to_datetime(series, format="%d-%b-%Y %H:%M:%S")
    try:
        loc = dt.dt.tz_localize(
            "Europe/Dublin", ambiguous="infer", nonexistent="shift_forward"
        )
    except Exception:
        loc = dt.dt.tz_localize(
            "Europe/Dublin", ambiguous=False, nonexistent="shift_forward"
        )
    return loc.dt.tz_convert("UTC")


def _fill_neighbor_mean(s: pd.Series) -> pd.Series:
    """Fill NaN with average of previous and next valid values (iterates until stable)."""
    out = s.copy()
    guard = 0
    while out.isna().any() and guard < len(out) + 5:
        prev = out.shift(1)
        nxt = out.shift(-1)
        both = prev.notna() & nxt.notna()
        fill = (prev + nxt) / 2.0
        out = out.where(out.notna(), fill.where(both))
        guard += 1
    return out


def _build_dataframe_30min(all_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=["effective_time"])
    df["carbon_intensity_gco2_per_kwh"] = pd.to_numeric(df["carbon_intensity_gco2_per_kwh"], errors="coerce")
    df["timestamp_utc"] = _localize_effective_to_utc(df["effective_time"])
    df = df.sort_values("timestamp_utc")
    # Duplicate UTC stamps -> mean
    s15 = df.groupby("timestamp_utc", sort=True)["carbon_intensity_gco2_per_kwh"].mean()
    # Full 15-minute grid for interpolation across gaps
    if len(s15) == 0:
        raise SystemExit("No rows after parsing")
    t_start = s15.index.min()
    t_end = s15.index.max()
    full_15 = pd.date_range(t_start, t_end, freq="15min", tz="UTC")
    s15 = s15.reindex(full_15)
    s15 = s15.interpolate(method="time", limit_direction="both")
    s15 = s15.bfill().ffill()
    # Strict 30-min: mean of [t, t+15m) inside each [k, k+30m) — use resample on series indexed at 15m
    s30 = s15.resample("30min", label="left", closed="left").mean()
    # Complete 30-min index from first to last half-hour in range
    full_30 = pd.date_range(s30.index.min(), s30.index.max(), freq="30min", tz="UTC")
    s30 = s30.reindex(full_30)
    s30 = s30.interpolate(method="time", limit_direction="both")
    s30 = _fill_neighbor_mean(s30)
    s30 = s30.bfill().ffill()
    s30.name = "carbon_intensity_gco2_per_kwh"
    out = s30.reset_index()
    out.columns = ["timestamp_utc", "carbon_intensity_gco2_per_kwh"]
    out["region"] = "ROI"
    return out[["timestamp_utc", "carbon_intensity_gco2_per_kwh", "region"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge EirGrid CO2 JSON files to CSV (UTC timestamps).")
    ap.add_argument(
        "inputs",
        nargs="*",
        help="JSON file paths, or leave empty and use --dir",
    )
    ap.add_argument(
        "--dir",
        default="",
        help="Directory containing *.json (used if no positional args)",
    )
    ap.add_argument(
        "--out",
        default="data/carbon_2020/eirgrid_roi_co2_2020_30min.csv",
        help="Output CSV path (30-minute grid; relative to project root if not absolute)",
    )
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths: List[str] = []
    if args.inputs:
        paths = list(args.inputs)
    elif args.dir:
        d = args.dir if os.path.isabs(args.dir) else os.path.join(root, args.dir)
        paths = sorted(glob(os.path.join(d, "*.json")))
    else:
        raise SystemExit("Provide JSON files as arguments or use --dir folder")

    if not paths:
        raise SystemExit("No JSON files found")

    all_rows: List[Dict[str, Any]] = []
    for p in paths:
        all_rows.extend(_rows_from_file(p))
        print(f"  + {os.path.basename(p)}: ok")

    df = _build_dataframe_30min(all_rows)
    out_path = args.out if os.path.isabs(args.out) else os.path.normpath(os.path.join(root, args.out))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows (30-min UTC) to {out_path}")
    print(f"  UTC range: {df['timestamp_utc'].min()} .. {df['timestamp_utc'].max()}")


if __name__ == "__main__":
    main()
