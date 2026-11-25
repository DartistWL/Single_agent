import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


HALF_HOUR_COLUMNS = [f"{h:.1f}" for h in np.arange(0.0, 24.0, 0.5)]


@dataclass(frozen=True)
class StoreNetColumns:
    # StoreNet Ireland raw files commonly have:
    #   date, Discharge(W), Charge(W), Production(W), Consumption(W), State of Charge(%), Hx_W
    date: str = "date"
    production_w: str = "Production(W)"
    consumption_w: str = "Consumption(W)"


def _find_header_row(csv_path: Path, needle: str = "date", max_lines: int = 50) -> int:
    """
    StoreNet raw CSVs may contain a few info rows before the header.
    We scan the first `max_lines` lines and return the row index containing the header.
    """
    needle_norm = needle.strip().lower()
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        for i in range(max_lines):
            line = f.readline()
            if not line:
                break
            parts = [p.strip().lower() for p in line.split(",")]
            if any(p == needle_norm for p in parts):
                return i
    raise ValueError(f"Cannot find header row containing '{needle}' in {csv_path}")


def _load_storenet_power_csv(csv_path: Path, cols: StoreNetColumns) -> pd.DataFrame:
    header_row = _find_header_row(csv_path, needle=cols.date)
    df = pd.read_csv(csv_path, skiprows=header_row)

    # Some raw CSVs contain leading spaces in header names, e.g. " Production(W)".
    # Normalize to make downstream column matching robust.
    df.columns = [c.strip() for c in df.columns]

    if cols.date not in df.columns:
        raise ValueError(f"Missing '{cols.date}' column in {csv_path}. Columns={list(df.columns)}")

    # Parse timestamp
    df[cols.date] = pd.to_datetime(df[cols.date], errors="coerce")
    df = df.dropna(subset=[cols.date]).sort_values(cols.date)
    df = df.set_index(cols.date)

    # Keep only what we need; coerce to numeric
    keep = [c for c in [cols.production_w, cols.consumption_w] if c in df.columns]
    if len(keep) < 2:
        raise ValueError(
            f"Power file must contain '{cols.production_w}' and '{cols.consumption_w}'. "
            f"Got columns={list(df.columns)} in {csv_path}"
        )

    df = df[keep].apply(pd.to_numeric, errors="coerce")
    return df


def _resample_to_half_hour_power_kw(df_w: pd.DataFrame) -> pd.DataFrame:
    """
    Input: 1-min power in W, indexed by datetime.
    Output: 30-min power in kW (mean over 30 minutes), with missing values interpolated in time.
    """
    df_w = df_w.sort_index()

    # Build a complete 1-minute index aligned to day boundaries so that
    # we can safely generate 48 half-hour slots per day (0.0, 0.5, ..., 23.5).
    start = df_w.index.min().floor("D")
    end = df_w.index.max().floor("D") + pd.Timedelta(hours=23, minutes=59)
    full_index = pd.date_range(start=start, end=end, freq="1min")

    df_1min = df_w.reindex(full_index)
    df_1min = df_1min.interpolate(method="time", limit_direction="both")

    # 30-min mean power (W) -> kW
    df_30min_w = df_1min.resample("30min").mean()
    df_30min_kw = df_30min_w / 1000.0
    return df_30min_kw


def _pivot_daily_48(df_30min_kw: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Convert a 30-min series into a daily pivot table (1 row per date, 48 half-hour columns).
    Drops incomplete days.
    """
    df = df_30min_kw[[value_col]].copy()
    df["date"] = df.index.date.astype(str)
    # Robust half-hour "time of day" bin -> steps 0..47 -> value = step/2
    tod_steps = df.index.hour * 2 + (df.index.minute // 30)
    df["tod"] = (tod_steps / 2.0).astype(float)

    pivot = df.pivot_table(index="date", columns="tod", values=value_col, aggfunc="mean")

    # Reindex to fixed 48 columns and drop incomplete days
    pivot.columns = [f"{c:.1f}" for c in pivot.columns.astype(float)]
    pivot = pivot.reindex(columns=HALF_HOUR_COLUMNS)
    pivot = pivot.dropna(axis=0, how="any")

    pivot = pivot.reset_index().rename(columns={"date": "date"})
    return pivot


def convert_house(
    raw_power_csv: Path,
    out_dir: Path,
    house_id: str,
    cols: StoreNetColumns,
):
    df_w = _load_storenet_power_csv(raw_power_csv, cols=cols)
    df_30min_kw = _resample_to_half_hour_power_kw(df_w)

    cons = _pivot_daily_48(df_30min_kw, cols.consumption_w)
    prod = _pivot_daily_48(df_30min_kw, cols.production_w)

    out_dir.mkdir(parents=True, exist_ok=True)
    cons_path = out_dir / f"daily_pivot_cons_2020_{house_id}.csv"
    prod_path = out_dir / f"daily_pivot_prod_2020_{house_id}.csv"

    cons.to_csv(cons_path, index=False)
    prod.to_csv(prod_path, index=False)
    return cons_path, prod_path, (cons.shape[0], prod.shape[0])


def _guess_power_files(raw_dir: Path):
    """
    Try to locate 'Power' raw CSVs for H1..H20.
    The Figshare repo contains an 'original data' folder; naming commonly includes 'H1_W'.
    We search for files matching '*H*_W*.csv' and then pick the latest per house.
    """
    # Important: there are both:
    #   - H1_W.csv   (power, with Production(W)/Consumption(W))
    #   - H1_Wh.csv  (energy, with Production(Wh)/Consumption(Wh))
    # We must ONLY pick the power files (exact stem match: "{hid}_W").
    candidates = list(raw_dir.rglob("*_W*.csv"))
    by_house = {}
    for p in candidates:
        name = p.stem.upper()
        # Extract house id like H1..H20
        for hid in [f"H{i}" for i in range(1, 21)]:
            if name == f"{hid}_W":
                by_house.setdefault(hid, []).append(p)
                break
    picked = {}
    for hid, files in by_house.items():
        files_sorted = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)
        picked[hid] = files_sorted[0]
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="解压后的 StoreNet 原始数据目录（包含 H1_W...H20_W 等CSV，可能在 original data/Power/ 下）",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/storenet_ireland_2020",
        help="输出目录（生成 daily_pivot_cons_2020_Hx.csv / daily_pivot_prod_2020_Hx.csv）",
    )
    parser.add_argument(
        "--houses",
        type=str,
        default="H1-H20",
        help="要转换的住户范围，例如 'H1-H20' 或 'H1,H2,H10'",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not raw_dir.exists():
        raise SystemExit(f"--raw_dir not found: {raw_dir}")

    cols = StoreNetColumns()

    # Determine target houses
    houses = []
    spec = args.houses.strip().upper()
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        if not (a.startswith("H") and b.startswith("H")):
            raise SystemExit(f"Invalid --houses range: {args.houses}")
        start = int(a[1:])
        end = int(b[1:])
        houses = [f"H{i}" for i in range(start, end + 1)]
    else:
        houses = [s.strip() for s in spec.split(",") if s.strip()]

    power_files = _guess_power_files(raw_dir)
    missing = [h for h in houses if h not in power_files]
    if missing:
        raise SystemExit(
            "未在 --raw_dir 下找到这些住户的功率文件（*_W*.csv）："
            + ", ".join(missing)
            + "\n你可以检查是否解压到了更深层目录，或把 --raw_dir 指到包含 original data 的那一层。"
        )

    summary = []
    for hid in houses:
        src = power_files[hid]
        cons_path, prod_path, (n_cons, n_prod) = convert_house(src, out_dir, hid, cols=cols)
        summary.append((hid, str(src), str(cons_path), str(prod_path), n_cons, n_prod))

    # Write a small manifest for quick inspection
    manifest = pd.DataFrame(
        summary,
        columns=["house_id", "source_power_csv", "cons_pivot_csv", "prod_pivot_csv", "cons_days", "prod_days"],
    )
    manifest_path = out_dir / "MANIFEST.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Done. Manifest written to {manifest_path}")


if __name__ == "__main__":
    # Make pandas print less noisy if user runs interactively
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    main()

