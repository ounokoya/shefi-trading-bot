from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from libs.data_loader import fetch_bybit_klines_range
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


def _bybit_supported_minutes() -> set[int]:
    return {1, 3, 5, 15, 30, 60, 120, 240, 360, 720}


def _aggregate_ohlcv(df: pd.DataFrame, *, base_tf_min: int, target_tf_min: int) -> pd.DataFrame:
    if df.empty:
        return df
    if int(target_tf_min) % int(base_tf_min) != 0:
        raise ValueError(f"Cannot aggregate {base_tf_min}m -> {target_tf_min}m")

    n = int(target_tf_min) // int(base_tf_min)
    if n <= 1:
        return df

    work = df.sort_values("ts").reset_index(drop=True).copy()
    work["_grp"] = (work.index // int(n)).astype(int)

    agg = (
        work.groupby("_grp", as_index=False)
        .agg(
            {
                "ts": "last",
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .drop(columns=["_grp"], errors="ignore")
    )
    return agg


def _fetch_tf_klines(*, symbol: str, tf: str, limit: int, days: int) -> pd.DataFrame:
    target_min = _tf_minutes(str(tf))
    end_dt = pd.Timestamp.now(tz="UTC")
    start_dt = end_dt - pd.Timedelta(days=int(days))

    supported = _bybit_supported_minutes()
    if int(target_min) in supported:
        return fetch_bybit_klines_range(
            symbol=str(symbol),
            interval=str(tf),
            start_ms=int(start_dt.timestamp() * 1000),
            end_ms=int(end_dt.timestamp() * 1000),
        )

    if int(target_min) % 240 == 0:
        base_tf = "4h"
        base_min = 240
    else:
        base_tf = "1h"
        base_min = 60

    df_base = fetch_bybit_klines_range(
        symbol=str(symbol),
        interval=str(base_tf),
        start_ms=int(start_dt.timestamp() * 1000),
        end_ms=int(end_dt.timestamp() * 1000),
    )
    if df_base.empty:
        return df_base

    df_agg = _aggregate_ohlcv(df_base, base_tf_min=int(base_min), target_tf_min=int(target_min))
    if int(len(df_agg)) > int(limit):
        df_agg = df_agg.iloc[-int(limit) :].reset_index(drop=True)
    return df_agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--tf", default="6h")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--days", type=int, default=400, help="history window to fetch (should exceed limit*tf)")
    ap.add_argument("--window-days", type=int, default=90, help="analysis window (approx). pivots/signals computed on last window-days")
    ap.add_argument("--macd", default="12,26,9")
    ap.add_argument("--dmi", default="14,14")
    ap.add_argument("--stoch", default="14,3,3")
    ap.add_argument("--cci", type=int, default=20)
    ap.add_argument("--pivot-zone", type=float, default=0.01)
    ap.add_argument("--pivot-merge", type=float, default=0.01)
    ap.add_argument("--max-pivots", type=int, default=10)
    ap.add_argument("--signal-win", type=int, default=30)
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

    df = df.sort_values("ts").reset_index(drop=True)
    if int(len(df)) > int(args.limit):
        df = df.iloc[-int(args.limit) :].reset_index(drop=True)

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
    signals = find_signals(df_win, pivots=pivots, cfg=cfg, max_signals=1000)

    print("\n==============================")
    print(" NEW_STRATEGIE DEMO (last bars)")
    print("==============================")
    print(f"Symbol: {args.symbol} TF: {args.tf} Bars(fetch kept): {len(df_ind)} Bars(window): {len(df_win)}")
    print(f"MACD: {macd_fast}/{macd_slow}/{macd_sig} DMI: {dmi_p}/{dmi_smooth} STOCH: {stoch_k}/{stoch_d} CCI: {args.cci}")
    print(f"Pivot zone: {args.pivot_zone:.4f} merge: {args.pivot_merge:.4f} max_pivots: {args.max_pivots}")
    print(f"Signal condition window bars: {args.signal_win}")
    print(f"Analysis window_days: {window_days} => window_bars: {window_bars}")

    print("\n--- PIVOTS (chronological) ---")
    prow = pivots_to_chrono_rows(pivots)
    if not prow:
        print("(none)")
    else:
        for r in prow:
            print(
                f"{r['first_dt']} | id={r['pivot_id']} side={r['side']} kind={r['kind']} level={r['level']:.6g} "
                f"zone=[{r['zone_low']:.6g},{r['zone_high']:.6g}] dx={r['dx_max_last']:.4f} tests={r['n_tests']}"
            )

    print("\n--- SIGNALS (chronological) ---")
    if not signals:
        print("(none)")
    else:
        for s in signals:
            m = dict(s.meta)
            print(
                f"{s.dt} | kind={s.kind} side={s.side} close={m.get('close'):.6g} "
                f"k={m.get('k')} d={m.get('d')} cci={m.get('cci')} adx={m.get('adx')} dx={m.get('dx')}"
            )


if __name__ == "__main__":
    main()
