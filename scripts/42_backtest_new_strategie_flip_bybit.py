from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from libs.data_loader import fetch_bybit_klines_range
from libs.new_strategie.backtest_flip import run_backtest_flip
from libs.new_strategie.config import NewStrategieConfig
from libs.new_strategie.indicators import ensure_indicators_df
from libs.new_strategie.pivots import build_top_pivots
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


def _bybit_supported_minutes() -> set[int]:
    return {1, 3, 5, 15, 30, 60, 120, 240, 360, 720}


def _fetch_tf_klines(*, symbol: str, tf: str, limit: int, days: int) -> pd.DataFrame:
    target_min = _tf_minutes(str(tf))
    end_dt = pd.Timestamp.now(tz="UTC")
    start_dt = end_dt - pd.Timedelta(days=int(days))

    supported = _bybit_supported_minutes()
    if int(target_min) not in supported:
        raise ValueError(f"TF not supported by Bybit for direct fetch: {tf} ({target_min}m)")

    df = fetch_bybit_klines_range(
        symbol=str(symbol),
        interval=str(tf),
        start_ms=int(start_dt.timestamp() * 1000),
        end_ms=int(end_dt.timestamp() * 1000),
    )

    if df.empty:
        return df

    df = df.sort_values("ts").reset_index(drop=True)
    if int(len(df)) > int(limit):
        df = df.iloc[-int(limit) :].reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--tf", default="6h")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--window-days", type=int, default=90)

    ap.add_argument("--macd", default="12,26,9")
    ap.add_argument("--dmi", default="14,6")
    ap.add_argument("--stoch", default="12,2,3")
    ap.add_argument("--cci", type=int, default=20)

    ap.add_argument("--pivot-zone", type=float, default=0.01)
    ap.add_argument("--pivot-merge", type=float, default=0.01)
    ap.add_argument("--max-pivots", type=int, default=10)
    ap.add_argument("--signal-win", type=int, default=30)
    ap.add_argument("--enable-premature", type=int, default=1)

    ap.add_argument("--sl-pct", type=float, default=0.05)
    ap.add_argument("--max-trades-print", type=int, default=50)

    args = ap.parse_args()

    macd_fast, macd_slow, macd_sig = [int(x.strip()) for x in str(args.macd).split(",")]
    dmi_p, dmi_smooth = [int(x.strip()) for x in str(args.dmi).split(",")]

    stoch_parts = [int(x.strip()) for x in str(args.stoch).split(",") if str(x).strip() != ""]
    if len(stoch_parts) == 2:
        stoch_k, stoch_k_smooth, stoch_d = int(stoch_parts[0]), 1, int(stoch_parts[1])
    elif len(stoch_parts) == 3:
        stoch_k, stoch_k_smooth, stoch_d = int(stoch_parts[0]), int(stoch_parts[1]), int(stoch_parts[2])
    else:
        raise ValueError(f"Invalid --stoch={args.stoch!r} (expected 'k,k_smooth,d' or 'k,d')")

    cfg = NewStrategieConfig(
        macd_fast=int(macd_fast),
        macd_slow=int(macd_slow),
        macd_signal=int(macd_sig),
        dmi_period=int(dmi_p),
        dmi_adx_smoothing=int(dmi_smooth),
        stoch_k_period=int(stoch_k),
        stoch_k_smooth_period=int(stoch_k_smooth),
        stoch_d_period=int(stoch_d),
        cci_period=int(args.cci),
        pivot_zone_pct=float(args.pivot_zone),
        pivot_merge_pct=float(args.pivot_merge),
        max_pivots=int(args.max_pivots),
        signal_condition_window_bars=int(args.signal_win),
    )

    df = _fetch_tf_klines(symbol=str(args.symbol), tf=str(args.tf), limit=int(args.limit), days=int(args.days))
    if df.empty:
        raise SystemExit("No data fetched")

    df_ind = ensure_indicators_df(df, cfg=cfg, force=False)

    tf_min = _tf_minutes(str(args.tf))
    window_days = int(args.window_days)
    window_bars = int(round(float(window_days) * 24.0 * 60.0 / float(tf_min))) if tf_min > 0 else int(len(df_ind))
    if window_bars < 1:
        window_bars = 1
    if window_bars > int(len(df_ind)):
        window_bars = int(len(df_ind))

    df_win = df_ind.iloc[-int(window_bars) :].reset_index(drop=True)

    pivots = build_top_pivots(df_win, cfg=cfg)
    signals = find_signals(df_win, pivots=pivots, cfg=cfg, max_signals=5000)

    if int(args.enable_premature) == 0:
        signals = [s for s in signals if str(s.kind) != "premature"]

    bt = run_backtest_flip(df_win, signals=signals, cfg=cfg, sl_pct=float(args.sl_pct))
    summary = dict(bt.get("summary") or {})
    trades_df = bt.get("trades")

    print("\n==============================")
    print(" NEW_STRATEGIE FLIP BACKTEST")
    print("==============================")
    print(f"Symbol: {args.symbol} TF: {args.tf} Bars(window): {len(df_win)} Signals: {len(signals)}")
    print(f"MACD: {macd_fast}/{macd_slow}/{macd_sig} DMI: {dmi_p}/{dmi_smooth} STOCH: {stoch_k},{stoch_k_smooth},{stoch_d} CCI: {args.cci}")
    print(f"SL pct: {float(args.sl_pct):.4f}")
    print("\n--- SUMMARY ---")
    for k in ["n_trades", "equity_end", "max_dd", "winrate", "avg_ret", "sl_pct"]:
        if k in summary:
            print(f"{k}: {summary.get(k)}")

    if isinstance(trades_df, pd.DataFrame) and (not trades_df.empty):
        print("\n--- LAST TRADES ---")
        n = int(args.max_trades_print)
        t = trades_df.tail(n)
        for _, r in t.iterrows():
            print(
                f"{r.get('entry_dt')} -> {r.get('exit_dt')} | side={r.get('side')} entry={r.get('entry_price')} exit={r.get('exit_price')} "
                f"ret={r.get('gross_ret')} reason={r.get('exit_reason')}"
            )


if __name__ == "__main__":
    main()
