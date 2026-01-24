import sys
from pathlib import Path
import argparse
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.data_loader import get_crypto_data
from libs.strategies.perp_hedge.models import AccountState
from libs.strategies.perp_hedge.brains.vwma_dx_mfi_cci_mtf import VWMADxMfiCciMtfBrain


def _intentions_to_str(ints):
    if not ints:
        return ""
    return "|".join([i.name for i in ints])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="LINKUSDT")
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--higher_tf", type=str, default="1h")
    parser.add_argument("--htf_close_opposite", action="store_true")
    parser.add_argument("--start", type=str, default="2026-01-01")
    parser.add_argument("--end", type=str, default="2026-01-18")

    parser.add_argument("--vwma_fast_ltf", type=int, default=12)
    parser.add_argument("--vwma_slow_ltf", type=int, default=72)
    parser.add_argument("--vwma_fast_htf", type=int, default=12)
    parser.add_argument("--vwma_slow_htf", type=int, default=72)

    parser.add_argument("--dmi_period", type=int, default=14)
    parser.add_argument("--adx_smoothing", type=int, default=6)
    parser.add_argument("--mfi_period", type=int, default=14)
    parser.add_argument("--cci_period", type=int, default=20)

    parser.add_argument("--mfi_high", type=float, default=60.0)
    parser.add_argument("--mfi_low", type=float, default=40.0)
    parser.add_argument("--cci_high", type=float, default=80.0)
    parser.add_argument("--cci_low", type=float, default=-80.0)

    parser.add_argument("--ltf_gate_mode", type=str, default="increase_only", choices=["both", "increase_only", "none"])

    args = parser.parse_args()

    df = get_crypto_data(args.symbol, args.start, args.end, args.timeframe, PROJECT_ROOT)
    if df.empty:
        print("No data")
        return

    if "open_time" not in df.columns and "ts" in df.columns:
        df["open_time"] = df["ts"]
    if "ts" in df.columns:
        df["ts"] = df["ts"].astype(int)
    if "open_time" in df.columns:
        df["open_time"] = df["open_time"].astype(int)

    brain = VWMADxMfiCciMtfBrain(
        source_df=df,
        source_tf=args.timeframe,
        higher_tf=args.higher_tf,
        htf_close_opposite=bool(args.htf_close_opposite),
        vwma_fast_ltf=args.vwma_fast_ltf,
        vwma_slow_ltf=args.vwma_slow_ltf,
        vwma_fast_htf=args.vwma_fast_htf,
        vwma_slow_htf=args.vwma_slow_htf,
        dmi_period=args.dmi_period,
        adx_smoothing=args.adx_smoothing,
        mfi_period=args.mfi_period,
        cci_period=args.cci_period,
        mfi_high=args.mfi_high,
        mfi_low=args.mfi_low,
        cci_high=args.cci_high,
        cci_low=args.cci_low,
        ltf_gate_mode=args.ltf_gate_mode,
    )

    dummy_state = AccountState(
        wallet_balance=0.0,
        margin_invested=0.0,
        long_size=0.0,
        long_entry_price=0.0,
        long_invested=0.0,
        short_size=0.0,
        short_entry_price=0.0,
        short_invested=0.0,
        current_price=0.0,
        pnl_unrealized_long=0.0,
        pnl_unrealized_short=0.0,
    )

    rows = []
    for _, r in df.iterrows():
        ts = int(r.get("open_time", r.get("ts")))
        dt_utc = pd.to_datetime(ts, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S")

        intentions = brain.get_intentions(ts, dummy_state)

        lock_dir = int(getattr(brain, "_lock_dir_map", {}).get(ts, 0) or 0)
        lock_side = "LONG" if lock_dir > 0 else ("SHORT" if lock_dir < 0 else "")

        ltf_cross_dir = int(getattr(brain, "_ltf_cross_dir_map", {}).get(ts, 0) or 0)
        htf_cross_dir = int(getattr(brain, "_htf_cross_dir_map", {}).get(ts, 0) or 0)

        dip_done = bool(getattr(brain, "_dip_done_map", {}).get(ts, False))
        long_extreme_entry = bool(getattr(brain, "_long_extreme_entry_map", {}).get(ts, False))
        short_extreme_entry = bool(getattr(brain, "_short_extreme_entry_map", {}).get(ts, False))
        extreme_entry_for_lock = bool(long_extreme_entry) if lock_dir > 0 else (bool(short_extreme_entry) if lock_dir < 0 else False)

        dx = getattr(brain, "_dx_map", {}).get(ts, None)
        plus_di = getattr(brain, "_plus_di_map", {}).get(ts, None)
        minus_di = getattr(brain, "_minus_di_map", {}).get(ts, None)
        mfi = getattr(brain, "_mfi_map", {}).get(ts, None)
        cci = getattr(brain, "_cci_map", {}).get(ts, None)

        di_sup = None
        dx_gt_di_sup = False
        if dx is not None and plus_di is not None and minus_di is not None and not pd.isna(dx) and not pd.isna(plus_di) and not pd.isna(minus_di):
            di_sup = max(float(plus_di), float(minus_di))
            dx_gt_di_sup = float(dx) > di_sup

        ltf_cross_ok = (ltf_cross_dir == lock_dir) and (lock_dir != 0)
        htf_cross_ok = (htf_cross_dir == lock_dir) and (lock_dir != 0)

        mfi_extreme_for_lock = False
        if mfi is not None and not pd.isna(mfi) and lock_dir != 0:
            if lock_dir > 0:
                mfi_extreme_for_lock = float(mfi) >= float(getattr(brain, "mfi_high", 60.0))
            else:
                mfi_extreme_for_lock = float(mfi) <= float(getattr(brain, "mfi_low", 40.0))

        cci_extreme_for_lock = False
        if cci is not None and not pd.isna(cci) and lock_dir != 0:
            if lock_dir > 0:
                cci_extreme_for_lock = float(cci) >= float(getattr(brain, "cci_high", 80.0))
            else:
                cci_extreme_for_lock = float(cci) <= float(getattr(brain, "cci_low", -80.0))

        decrease_conditions = bool(dx_gt_di_sup and mfi_extreme_for_lock and cci_extreme_for_lock)

        rows.append(
            {
                "timestamp": ts,
                "dt_utc": dt_utc,
                "open": float(r.get("open", 0.0)),
                "high": float(r.get("high", 0.0)),
                "low": float(r.get("low", 0.0)),
                "close": float(r.get("close", 0.0)),
                "volume": float(r.get("volume", 0.0)),
                "intentions": _intentions_to_str(intentions),
                "lock_dir": lock_dir,
                "lock_side": lock_side,
                "htf_cross_dir": htf_cross_dir,
                "ltf_cross_dir": ltf_cross_dir,
                "htf_cross_ok": bool(htf_cross_ok),
                "ltf_cross_ok": bool(ltf_cross_ok),
                "dip_done": bool(dip_done),
                "long_extreme_entry": bool(long_extreme_entry),
                "short_extreme_entry": bool(short_extreme_entry),
                "extreme_entry_for_lock": bool(extreme_entry_for_lock),
                "dx": dx,
                "plus_di": plus_di,
                "minus_di": minus_di,
                "di_sup": di_sup,
                "dx_gt_di_sup": bool(dx_gt_di_sup),
                "mfi": mfi,
                "cci": cci,
                "mfi_extreme_for_lock": bool(mfi_extreme_for_lock),
                "cci_extreme_for_lock": bool(cci_extreme_for_lock),
                "decrease_conditions": bool(decrease_conditions),
                "higher_tf": str(getattr(brain, "higher_tf", "")),
                "htf_close_opposite": bool(getattr(brain, "htf_close_opposite", False)),
                "ltf_gate_mode": str(getattr(brain, "ltf_gate_mode", "")),
                "vwma_fast_ltf": int(getattr(brain, "vwma_fast_ltf", 0)),
                "vwma_slow_ltf": int(getattr(brain, "vwma_slow_ltf", 0)),
                "vwma_fast_htf": int(getattr(brain, "vwma_fast_htf", 0)),
                "vwma_slow_htf": int(getattr(brain, "vwma_slow_htf", 0)),
                "dmi_period": int(getattr(brain, "dmi_period", 0)),
                "adx_smoothing": int(getattr(brain, "adx_smoothing", 0)),
                "mfi_period": int(getattr(brain, "mfi_period", 0)),
                "cci_period": int(getattr(brain, "cci_period", 0)),
                "mfi_high": float(getattr(brain, "mfi_high", 0.0)),
                "mfi_low": float(getattr(brain, "mfi_low", 0.0)),
                "cci_high": float(getattr(brain, "cci_high", 0.0)),
                "cci_low": float(getattr(brain, "cci_low", 0.0)),
            }
        )

    out_df = pd.DataFrame(rows)

    out_dir = PROJECT_ROOT / "data" / "processed" / "backtests" / "perp_hedge"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = f"vwma_dx_mfi_cci_mtf_signals_{args.symbol}_{args.timeframe}_{args.higher_tf}_{args.start}_{args.end}"

    out_all_path = out_dir / f"{base}_all_bars.csv"
    out_df.to_csv(out_all_path, index=False)

    out_signals = out_df[out_df["intentions"].fillna("") != ""].copy()
    out_signals_path = out_dir / f"{base}_signals.csv"
    out_signals.to_csv(out_signals_path, index=False)

    out_events = out_df[
        (out_df["intentions"].fillna("") != "")
        | (out_df["htf_cross_dir"].fillna(0).astype(int) != 0)
        | (out_df["ltf_cross_dir"].fillna(0).astype(int) != 0)
        | (out_df["decrease_conditions"].fillna(False).astype(bool))
    ].copy()
    out_events_path = out_dir / f"{base}_events.csv"
    out_events.to_csv(out_events_path, index=False)

    print(f"Saved: {out_all_path}")
    print(f"Saved: {out_signals_path}")
    print(f"Saved: {out_events_path}")
    print(f"Bars: {len(out_df)}")
    print(f"Signals: {len(out_signals)}")
    print(f"Events: {len(out_events)}")


if __name__ == "__main__":
    main()
