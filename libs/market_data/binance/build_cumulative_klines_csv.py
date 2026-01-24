from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd


def build_cumulative_klines_csv(
    *,
    dumped_root_dir: str | Path,
    ticker: str,
    interval: str,
    date_start: dt.date,
    date_end: dt.date,
    out_csv: str | Path,
) -> Path:
    root = Path(dumped_root_dir)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start_ms = int(dt.datetime(date_start.year, date_start.month, date_start.day, tzinfo=dt.timezone.utc).timestamp() * 1000)
    end_exclusive_date = date_end + dt.timedelta(days=1)
    end_exclusive_ms = int(
        dt.datetime(end_exclusive_date.year, end_exclusive_date.month, end_exclusive_date.day, tzinfo=dt.timezone.utc).timestamp()
        * 1000
    )

    csv_files: list[Path] = []
    for p in root.rglob("*.csv"):
        ps = p.as_posix()
        if "CHECKSUM" in p.name:
            continue
        if ticker not in ps:
            continue
        if f"/{interval}/" not in ps and f"_{interval}_" not in p.name and interval not in ps:
            continue
        csv_files.append(p)

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {root} for {ticker} {interval}")

    frames: list[pd.DataFrame] = []
    for p in sorted(csv_files):
        df = pd.read_csv(p, header=None)
        if df.shape[1] < 6:
            continue
        df = df.iloc[:, :6]
        df.columns = ["ts", "open", "high", "low", "close", "volume"]

        df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
        df = df.dropna(subset=["ts"])
        df["ts"] = df["ts"].astype("int64")

        for c in ("open", "high", "low", "close", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["open", "high", "low", "close", "volume"])
        df = df[(df["ts"] >= start_ms) & (df["ts"] < end_exclusive_ms)]
        frames.append(df)

    if not frames:
        raise ValueError("No rows loaded after parsing/filtering")

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["ts"], keep="last")
    out = out.sort_values("ts").reset_index(drop=True)

    out.to_csv(out_path, index=False)
    return out_path
