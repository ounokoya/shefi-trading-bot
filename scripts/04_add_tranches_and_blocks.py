from __future__ import annotations

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VENV_ROOT = PROJECT_ROOT / "venv_optuna"
VENV_PY = VENV_ROOT / "bin" / "python"
if VENV_PY.exists() and Path(sys.prefix).resolve() != VENV_ROOT.resolve():
    os.execv(str(VENV_PY), [str(VENV_PY), *sys.argv])

import argparse

import pandas as pd

from libs.blocks.add_blocks_multislot_df import add_blocks_multislot_df
from libs.blocks.add_macd_tv_columns_df import add_macd_tv_columns_df
from libs.blocks.add_tranche_hist_extreme_candidates_df import add_tranche_hist_extreme_candidates_df
from libs.blocks.segment_macd_hist_tranches_df import segment_macd_hist_tranches_df
from libs.indicators.moving_averages.vwma_tv import vwma_tv
from libs.indicators.volatility.atr_tv import atr_tv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-csv",
        default="data/processed/klines/LINKUSDT_5m_2020-01-01_2024-12-31_with_macd_12_26_9.csv",
    )
    ap.add_argument("--out-csv", default="")
    ap.add_argument("--close-col", default="close")
    ap.add_argument("--extremes-on", default="high_low", choices=["high_low", "close"])
    ap.add_argument("--block-extremes", default="perfect", choices=["perfect", "first_candidate"])
    ap.add_argument("--candidate-scope", default="tfav_only", choices=["tfav_only", "all"])
    ap.add_argument(
        "--candidate-filter",
        default="none",
        choices=["none", "vwma4_align", "vwma4_12_align", "vwma4_12_macd_align"],
    )
    ap.add_argument("--candidate-stop-mode", default="none", choices=["none", "pct", "atr"])
    ap.add_argument("--candidate-stop-pct", type=float, default=0.01)
    ap.add_argument("--candidate-stop-atr-period", type=int, default=14)
    ap.add_argument("--candidate-stop-atr-mult", type=float, default=2.0)
    ap.add_argument("--fast", type=int, default=12)
    ap.add_argument("--slow", type=int, default=26)
    ap.add_argument("--signal", type=int, default=9)
    args = ap.parse_args()

    in_csv = Path(str(args.in_csv))
    out_csv_str = str(args.out_csv).strip()
    if out_csv_str:
        out_csv = Path(out_csv_str)
    else:
        suffix = "_with_tranches_and_blocks"
        if str(args.block_extremes) == "first_candidate":
            suffix += "_first_candidate"
            if str(args.candidate_filter) != "none":
                suffix += f"_{str(args.candidate_filter)}"
            if str(args.candidate_stop_mode) != "none":
                if str(args.candidate_stop_mode) == "pct":
                    pct_tag = int(round(float(args.candidate_stop_pct) * 100))
                    suffix += f"_stop{pct_tag}pct"
                else:
                    suffix += f"_stop{float(args.candidate_stop_atr_mult)}atr{int(args.candidate_stop_atr_period)}"
        out_csv = Path(str(in_csv).replace(".csv", f"{suffix}.csv"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    if "ts" in df.columns:
        df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("Int64")
        df = df.sort_values("ts").reset_index(drop=True)

    if "macd_hist" not in df.columns or "macd_line" not in df.columns or "macd_signal" not in df.columns:
        df = add_macd_tv_columns_df(
            df,
            close_col=str(args.close_col),
            fast_period=int(args.fast),
            slow_period=int(args.slow),
            signal_period=int(args.signal),
        )

    df = segment_macd_hist_tranches_df(
        df,
        close_col=str(args.close_col),
        extremes_on=str(args.extremes_on),
    )

    if str(args.block_extremes) == "first_candidate":
        if str(args.candidate_stop_mode) == "atr":
            if "high" not in df.columns or "low" not in df.columns or str(args.close_col) not in df.columns:
                raise ValueError("Missing required columns for ATR: high/low/close")
            atr_col = f"atr_{int(args.candidate_stop_atr_period)}"
            if atr_col not in df.columns:
                high = pd.to_numeric(df["high"], errors="coerce").astype(float).tolist()
                low = pd.to_numeric(df["low"], errors="coerce").astype(float).tolist()
                close_for_atr = pd.to_numeric(df[str(args.close_col)], errors="coerce").astype(float).tolist()
                df[atr_col] = atr_tv(high, low, close_for_atr, int(args.candidate_stop_atr_period))

        if str(args.candidate_filter) in ("vwma4_align", "vwma4_12_align", "vwma4_12_macd_align"):
            if "volume" not in df.columns:
                raise ValueError("Missing required column: volume")
            close = pd.to_numeric(df[str(args.close_col)], errors="coerce").astype(float).tolist()
            volume = pd.to_numeric(df["volume"], errors="coerce").astype(float).tolist()
            if "vwma_4" not in df.columns:
                df["vwma_4"] = vwma_tv(close, volume, 4)
            if str(args.candidate_filter) in ("vwma4_12_align", "vwma4_12_macd_align") and "vwma_12" not in df.columns:
                df["vwma_12"] = vwma_tv(close, volume, 12)

        df = add_tranche_hist_extreme_candidates_df(
            df,
            price_col=str(args.close_col),
            candidate_filter=str(args.candidate_filter),
            stop_mode=str(args.candidate_stop_mode),
            stop_pct=float(args.candidate_stop_pct),
            atr_col=f"atr_{int(args.candidate_stop_atr_period)}",
            atr_mult=float(args.candidate_stop_atr_mult),
        )
        df = add_blocks_multislot_df(
            df,
            use_tranche_candidate_extremes=True,
            candidate_extremes_scope=str(args.candidate_scope),
        )
    else:
        df = add_blocks_multislot_df(df)

    df.to_csv(out_csv, index=False)
    print(str(out_csv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
