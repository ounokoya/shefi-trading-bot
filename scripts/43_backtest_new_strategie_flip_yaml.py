from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.data_loader import fetch_bybit_klines_range
from libs.new_strategie.backtest_config import load_config_yaml
from libs.new_strategie.backtest_flip import run_backtest_flip
from libs.new_strategie.config import NewStrategieConfig
from libs.new_strategie.indicators import ensure_indicators_df
from libs.new_strategie.pivots import build_top_pivots, pivots_to_chrono_rows
from libs.new_strategie.signals import find_signals


def _tf_minutes(tf: str) -> int:
    t = str(tf or "").strip().lower()
    if t.endswith("m") and t[:-1].isdigit():
        return int(t[:-1])
    if t.endswith("min") and t[:-3].isdigit():
        return int(t[:-3])
    if t.endswith("h") and t[:-1].isdigit():
        return int(t[:-1]) * 60
    raise ValueError(f"Unsupported tf: {tf!r}")


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

    start_dt = pd.Timestamp(str(cfg.bybit.start), tz="UTC")
    end_dt = pd.Timestamp(str(cfg.bybit.end), tz="UTC")

    if start_dt > end_dt:
        raise SystemExit(
            f"Invalid bybit date range: start={cfg.bybit.start!r} > end={cfg.bybit.end!r}. "
            "Fix the YAML so start <= end."
        )

    df = fetch_bybit_klines_range(
        symbol=str(cfg.bybit.symbol),
        interval=str(cfg.bybit.interval),
        start_ms=int(start_dt.timestamp() * 1000),
        end_ms=int(end_dt.timestamp() * 1000),
        base_url=str(cfg.bybit.base_url),
        category=str(cfg.bybit.category),
    )

    if df.empty:
        raise SystemExit("No data fetched")

    df = df.sort_values("ts").reset_index(drop=True)
    if int(len(df)) > int(cfg.bybit.limit):
        df = df.iloc[-int(cfg.bybit.limit) :].reset_index(drop=True)

    ns_cfg = NewStrategieConfig(
        macd_fast=int(cfg.indicators.macd_fast),
        macd_slow=int(cfg.indicators.macd_slow),
        macd_signal=int(cfg.indicators.macd_signal),
        dmi_period=int(cfg.indicators.dmi_period),
        dmi_adx_smoothing=int(cfg.indicators.dmi_adx_smoothing),
        stoch_k_period=int(cfg.indicators.stoch_k),
        stoch_k_smooth_period=int(cfg.indicators.stoch_k_smooth),
        stoch_d_period=int(cfg.indicators.stoch_d),
        cci_period=int(cfg.indicators.cci_period),
        pivot_zone_pct=float(cfg.pivots.zone_pct),
        pivot_merge_pct=float(cfg.pivots.merge_pct),
        max_pivots=int(cfg.pivots.max_pivots),
        signal_condition_window_bars=int(cfg.signals.condition_window_bars),
    )

    df_ind = ensure_indicators_df(df, cfg=ns_cfg, force=False)

    tf_min = _tf_minutes(str(cfg.bybit.interval))
    window_days = int(cfg.window.window_days)
    window_bars = int(round(float(window_days) * 24.0 * 60.0 / float(tf_min))) if tf_min > 0 else int(len(df_ind))
    if window_bars < 1:
        window_bars = 1
    if window_bars > int(len(df_ind)):
        window_bars = int(len(df_ind))

    df_win = df_ind.iloc[-int(window_bars) :].reset_index(drop=True)

    pivots = build_top_pivots(df_win, cfg=ns_cfg)
    pivot_rows = pivots_to_chrono_rows(pivots)
    pivots_df = pd.DataFrame(pivot_rows)

    signals = find_signals(df_win, pivots=pivots, cfg=ns_cfg, max_signals=5000)

    if not bool(cfg.signals.enable_premature):
        signals = [s for s in signals if str(s.kind) != "premature"]
    signals_df = pd.DataFrame(
        [
            {
                "kind": s.kind,
                "pos": int(s.pos),
                "ts": int(s.ts),
                "dt": str(s.dt),
                "side": str(s.side),
                **{f"meta_{k}": v for k, v in dict(s.meta).items()},
            }
            for s in signals
        ]
    )

    bt = run_backtest_flip(df_win, signals=signals, cfg=ns_cfg, sl_pct=float(cfg.backtest.sl_pct))
    trades = bt["trades"]
    equity = bt["equity"]
    summary = bt["summary"]

    # Exports
    if bool(cfg.output.save_csv):
        df_csv = out_dir / "ohlcv_indicators.csv"
        trades_csv = out_dir / "trades.csv"
        equity_csv = out_dir / "equity.csv"
        pivots_csv = out_dir / "pivots.csv"
        signals_csv = out_dir / "signals.csv"

        df_out = _add_dt_cols(df_win, ["ts"])
        df_out.to_csv(df_csv, index=False, float_format="%.6f")

        if isinstance(trades, pd.DataFrame) and (not trades.empty):
            trades.to_csv(trades_csv, index=False, float_format="%.6f")
        else:
            pd.DataFrame([]).to_csv(trades_csv, index=False)

        if isinstance(equity, pd.DataFrame) and (not equity.empty):
            equity2 = _add_dt_cols(equity, ["ts"])
            equity2.to_csv(equity_csv, index=False, float_format="%.6f")
        else:
            pd.DataFrame([]).to_csv(equity_csv, index=False)

        if not pivots_df.empty:
            pivots_df.to_csv(pivots_csv, index=False, float_format="%.6f")
        else:
            pd.DataFrame([]).to_csv(pivots_csv, index=False)

        if not signals_df.empty:
            signals_df.to_csv(signals_csv, index=False, float_format="%.6f")
        else:
            pd.DataFrame([]).to_csv(signals_csv, index=False)

    print("Backtest summary:")
    print(f"- symbol: {cfg.bybit.symbol}")
    print(f"- tf: {cfg.bybit.interval}")
    print(f"- bars_fetch_kept: {len(df_ind)}")
    print(f"- bars_window: {len(df_win)} (window_days={window_days})")
    print(f"- n_signals: {len(signals)}")
    print(f"- n_pivots: {len(pivots)}")
    print(f"- n_trades: {summary.get('n_trades')}")
    print(f"- equity_end: {summary.get('equity_end')}")
    print(f"- max_dd: {summary.get('max_dd')}")
    print(f"- winrate: {summary.get('winrate')}")
    print(f"- sl_pct: {summary.get('sl_pct')}")

    # PNG (equity + dd)
    try:
        import matplotlib.pyplot as plt

        if bool(cfg.output.png) and isinstance(equity, pd.DataFrame) and len(equity):
            dt_index = pd.to_datetime(pd.to_numeric(equity["ts"], errors="coerce"), unit="ms", utc=True)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            ax1.plot(dt_index, equity["equity"], label="equity")
            ax1.set_title("Equity (gross, additive)")
            ax1.grid(True, alpha=0.3)

            if "dd" in equity.columns:
                ax2.fill_between(dt_index, equity["dd"], 0.0, alpha=0.25, label="drawdown")
            ax2.set_title("Drawdown")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            png_path = out_dir / "equity.png"
            fig.savefig(png_path)
            print(f"Wrote: {png_path}")
    except Exception as e:
        print(f"PNG skipped: {e}")

    print(f"Out dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
