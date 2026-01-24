from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.backtest_triple_vwma_trend.config import load_config_yaml
from libs.backtest_triple_vwma_trend.engine import run_backtest_from_config


def _add_dt_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df2 = df.copy()
    for c in cols:
        if c not in df2.columns:
            continue
        s = pd.to_numeric(df2[c], errors="coerce")
        df2[f"{c}_dt"] = pd.to_datetime(s, unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return df2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--trace", action="store_true", help="Print progress logs (useful when no_lookahead=true)")
    ap.add_argument("--trace-every", type=int, default=500, help="Print progress every N bars")
    args = ap.parse_args()

    cfg = load_config_yaml(str(args.config))

    out_dir = Path(str(args.out_dir) if args.out_dir is not None else cfg.output.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = run_backtest_from_config(cfg=cfg, trace=bool(args.trace), trace_every=int(args.trace_every))
    trades = res["trades"]
    equity = res["equity"]
    summary = res["summary"]

    if len(trades):
        trades = _add_dt_cols(trades, ["entry_ts", "exit_ts"])
    if len(equity):
        equity = _add_dt_cols(equity, ["ts"])

    trades_csv = out_dir / "trades.csv"
    equity_csv = out_dir / "equity.csv"
    trades.to_csv(trades_csv, index=False, float_format="%.6f")
    equity.to_csv(equity_csv, index=False, float_format="%.6f")

    print("Backtest summary:")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"- {k}: {v:.6f}")
        else:
            print(f"- {k}: {v}")

    print(f"Wrote: {trades_csv}")
    print(f"Wrote: {equity_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
