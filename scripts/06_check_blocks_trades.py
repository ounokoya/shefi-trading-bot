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

from libs.blocks.extract_block_trades_df import extract_block_trades_df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-csv",
        default="data/processed/blocks/LINKUSDT_4h_2020-01-01_2025-12-31_with_tranches_and_blocks.csv",
    )
    ap.add_argument(
        "--out-trades-csv",
        default="data/processed/blocks/LINKUSDT_4h_2020-01-01_2025-12-31_trades.csv",
    )
    ap.add_argument(
        "--out-issues-csv",
        default="data/processed/blocks/LINKUSDT_4h_2020-01-01_2025-12-31_trade_issues.csv",
    )
    args = ap.parse_args()

    df = pd.read_csv(str(args.in_csv))
    trades_df, issues_df = extract_block_trades_df(df)

    out_trades = Path(str(args.out_trades_csv))
    out_trades.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(out_trades, index=False)

    out_issues = Path(str(args.out_issues_csv))
    out_issues.parent.mkdir(parents=True, exist_ok=True)
    issues_df.to_csv(out_issues, index=False)

    print(f"trades={len(trades_df)} issues={len(issues_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
