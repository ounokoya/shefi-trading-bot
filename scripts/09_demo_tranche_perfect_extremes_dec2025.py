from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.blocks.add_macd_tv_columns_df import add_macd_tv_columns_df
from libs.blocks.extract_tranche_perfect_series_extremes_df import extract_tranche_perfect_series_extremes_df
from libs.blocks.segment_macd_hist_tranches_df import segment_macd_hist_tranches_df
from libs.indicators.momentum.cci_tv import cci_tv
from libs.indicators.moving_averages.vwma_tv import vwma_tv
from libs.market_data.binance.build_cumulative_klines_csv import build_cumulative_klines_csv
from libs.market_data.binance.dump_um_klines import dump_um_klines


def _parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _ms_to_iso_utc(ms: object) -> str:
    if ms is None:
        return ""
    try:
        msi = int(ms)
    except Exception:
        return ""
    return pd.to_datetime(msi, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="LINKUSDT")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--start", default="2025-12-01", type=_parse_date)
    ap.add_argument("--end", default="2025-12-31", type=_parse_date, help="Inclusive end date")
    ap.add_argument("--root-dir", default="data/raw/binance_data_vision")
    ap.add_argument("--update-existing", action="store_true")
    ap.add_argument("--macd-fast", type=int, default=12)
    ap.add_argument("--macd-slow", type=int, default=26)
    ap.add_argument("--macd-signal", type=int, default=9)
    ap.add_argument("--series-col", default="close")
    ap.add_argument(
        "--out-extremes-csv",
        default=None,
    )
    args = ap.parse_args()

    dumped_root = dump_um_klines(
        root_dir=args.root_dir,
        ticker=str(args.ticker),
        interval=str(args.interval),
        date_start=args.start,
        date_end=args.end,
        update_existing=bool(args.update_existing),
    )

    cumulative_csv = Path("data/processed/klines") / f"{str(args.ticker)}_{str(args.interval)}_{args.start}_{args.end}.csv"
    cumulative_csv.parent.mkdir(parents=True, exist_ok=True)

    build_cumulative_klines_csv(
        dumped_root_dir=dumped_root,
        ticker=str(args.ticker),
        interval=str(args.interval),
        date_start=args.start,
        date_end=args.end,
        out_csv=cumulative_csv,
    )

    df = pd.read_csv(cumulative_csv)
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)

    if "macd_hist" not in df.columns or "macd_line" not in df.columns or "macd_signal" not in df.columns:
        df = add_macd_tv_columns_df(
            df,
            close_col="close",
            fast_period=int(args.macd_fast),
            slow_period=int(args.macd_slow),
            signal_period=int(args.macd_signal),
        )

    high = pd.to_numeric(df["high"], errors="coerce").astype(float).tolist()
    low = pd.to_numeric(df["low"], errors="coerce").astype(float).tolist()
    close = pd.to_numeric(df["close"], errors="coerce").astype(float).tolist()
    volume = pd.to_numeric(df["volume"], errors="coerce").astype(float).tolist()

    if "cci_30" not in df.columns:
        df["cci_30"] = cci_tv(high, low, close, 30)
    if "cci_120" not in df.columns:
        df["cci_120"] = cci_tv(high, low, close, 120)
    if "cci_300" not in df.columns:
        df["cci_300"] = cci_tv(high, low, close, 300)
    if "vwma_4" not in df.columns:
        df["vwma_4"] = vwma_tv(close, volume, 4)

    series_col = str(args.series_col)
    if series_col not in df.columns:
        raise ValueError(f"--series-col {series_col} not found in dataframe")

    df = segment_macd_hist_tranches_df(
        df,
        ts_col="ts",
        high_col="high",
        low_col="low",
        close_col="close",
        hist_col="macd_hist",
        extremes_on="close",
    )

    extremes = extract_tranche_perfect_series_extremes_df(
        df,
        series_col=series_col,
        ts_col="ts",
        tranche_id_col="tranche_id",
        tranche_sign_col="tranche_sign",
        tranche_start_ts_col="tranche_start_ts",
        tranche_end_ts_col="tranche_end_ts",
        tranche_len_col="tranche_len",
    )

    out_path = Path(
        str(args.out_extremes_csv)
        if args.out_extremes_csv is not None
        else f"data/processed/klines/tranche_perfect_extremes_{str(args.ticker)}_{str(args.interval)}_{args.start}_{args.end}_{series_col}.csv"
    )

    if extremes.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        extremes.to_csv(out_path, index=False)
        print(f"Wrote 0 tranche extremes to: {out_path}")
        return 0

    extremes["tranche_start_dt"] = extremes["tranche_start_ts"].map(_ms_to_iso_utc)
    extremes["tranche_end_dt"] = extremes["tranche_end_ts"].map(_ms_to_iso_utc)
    extremes["extreme_dt"] = extremes["extreme_ts"].map(_ms_to_iso_utc)

    # Join feature values at the extreme candle
    feat_cols = [
        "ts",
        "close",
        "macd_hist",
        "macd_line",
        "cci_30",
        "cci_120",
        "cci_300",
        "vwma_4",
        series_col,
    ]
    df_feats = df[feat_cols].copy().reset_index(drop=True)
    df_feats["extreme_row_index"] = df_feats.index.astype(int)
    extremes = extremes.merge(df_feats, on="extreme_row_index", how="left", suffixes=("", "_at_extreme"))

    for _, r in extremes.iterrows():
        print(
            f"tranche extreme | series={r['series_col']} tid={int(r['tranche_id'])} sign={r['tranche_sign']} side={r['open_side']} "
            f"start={r['tranche_start_dt']} end={r['tranche_end_dt']} len={int(r['tranche_len'])} "
            f"extreme={r['extreme_dt']} value={float(r['extreme_value'])}"
        )

    cols = [
        "tranche_id",
        "tranche_sign",
        "open_side",
        "tranche_start_ts",
        "tranche_start_dt",
        "tranche_end_ts",
        "tranche_end_dt",
        "tranche_len",
        "series_col",
        "extreme_kind",
        "extreme_ts",
        "extreme_dt",
        "extreme_value",
        "extreme_row_index",
        "close",
        "macd_hist",
        "macd_line",
        "cci_30",
        "cci_120",
        "cci_300",
        "vwma_4",
    ]
    extremes = extremes[cols]
    extremes.to_csv(out_path, index=False)
    print(f"\nWrote {len(extremes)} tranche extremes to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
