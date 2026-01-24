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

from libs.validation.validate_ohlc_consistency_df import validate_ohlc_consistency_df
from libs.validation.validate_ohlcv_numeric_df import validate_ohlcv_numeric_df
from libs.validation.validate_required_columns_df import validate_required_columns_df
from libs.validation.validate_ts_sorted_unique_df import validate_ts_sorted_unique_df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-csv",
        default="data/processed/blocks/LINKUSDT_4h_2020-01-01_2025-12-31_with_tranches_and_blocks.csv",
    )
    ap.add_argument("--out-csv", default="")
    ap.add_argument("--out-parquet", default="")
    args = ap.parse_args()

    df = pd.read_csv(str(args.in_csv))

    validate_required_columns_df(df, required=["ts", "open", "high", "low", "close", "volume"])
    validate_ts_sorted_unique_df(df, ts_col="ts")
    validate_ohlcv_numeric_df(df, cols=["open", "high", "low", "close", "volume"])
    validate_ohlc_consistency_df(df)

    if args.out_csv:
        out_csv = Path(str(args.out_csv))
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    if args.out_parquet:
        out_parquet = Path(str(args.out_parquet))
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_parquet, index=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
