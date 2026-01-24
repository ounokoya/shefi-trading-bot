from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.backtest_price_action_tranche.config import load_config_yaml
from libs.backtest_price_action_tranche.engine import run_backtest_from_config


def _ms_to_iso_utc(ms: object) -> str:
    try:
        msi = int(ms)
    except Exception:
        return ""
    return pd.to_datetime(msi, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")


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
    args = ap.parse_args()

    cfg = load_config_yaml(str(args.config))

    out_dir = Path(str(args.out_dir) if args.out_dir is not None else cfg.output.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = run_backtest_from_config(cfg=cfg)
    trades = res["trades"]
    equity = res["equity"]
    pivot_map = res.get("pivot_map")
    summary = res["summary"]

    trades = _add_dt_cols(
        trades,
        [
            "entry_ts",
            "exit_ts",
            "entry_signal_ts",
            "entry_signal_cand_ts",
            "exit_signal_ts",
        ],
    )
    equity = _add_dt_cols(equity, ["ts"])

    trades_csv = out_dir / "trades.csv"
    equity_csv = out_dir / "equity.csv"
    pivot_map_csv = out_dir / "pivot_map.csv"
    trades.to_csv(trades_csv, index=False, float_format="%.6f")
    equity.to_csv(equity_csv, index=False, float_format="%.6f")
    if pivot_map is not None and len(pivot_map):
        pivot_map = _add_dt_cols(pivot_map, ["entry_ts", "exit_ts"])
        pivot_map.to_csv(pivot_map_csv, index=False, float_format="%.6f")

    print("Backtest summary:")
    print(f"- n_trades: {summary['n_trades']}")
    if "n_wins" in summary:
        print(f"- n_wins: {summary['n_wins']}")
    if "n_losses" in summary:
        print(f"- n_losses: {summary['n_losses']}")
    if "winrate" in summary:
        print(f"- winrate: {summary['winrate']:.4f}")
    if "avg_win" in summary:
        print(f"- avg_win: {summary['avg_win']:.6f}")
    if "avg_loss" in summary:
        print(f"- avg_loss: {summary['avg_loss']:.6f}")
    print(f"- equity_end: {summary['equity_end']:.6f}")
    print(f"- max_dd: {summary['max_dd']:.6f}")
    print(f"- ratio: {summary['ratio']}")
    if len(equity):
        print(f"- first_ts: {_ms_to_iso_utc(equity['ts'].iloc[0])}")
        print(f"- last_ts:  {_ms_to_iso_utc(equity['ts'].iloc[-1])}")

    try:
        import matplotlib.pyplot as plt

        if bool(cfg.output.png):
            dt_index = pd.to_datetime(pd.to_numeric(equity["ts"], errors="coerce"), unit="ms", utc=True)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            ax1.plot(dt_index, equity["equity"], label="equity")
            ax1.set_title("Equity (additive)")
            ax1.grid(True, alpha=0.3)
            ax2.fill_between(dt_index, equity["dd"], 0.0, alpha=0.25, label="drawdown")
            ax2.set_title("Drawdown")
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            png_path = out_dir / "equity.png"
            fig.savefig(png_path)
            print(f"Wrote: {png_path}")
    except Exception as e:
        print(f"PNG skipped: {e}")

    print(f"Wrote: {trades_csv}")
    print(f"Wrote: {equity_csv}")
    if pivot_map is not None and len(pivot_map):
        print(f"Wrote: {pivot_map_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
