import sys
from pathlib import Path
import argparse
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.data_loader import get_crypto_data
from libs.strategies.perp_hedge.models import AccountState
from libs.strategies.perp_hedge.models import SignalIntent
from libs.strategies.perp_hedge.brains.cci_1h_tf_5m import CCI1hTf5mBrain


def _intentions_to_str(ints):
    if not ints:
        return ""
    return "|".join([i.name for i in ints])


def _get_indicator_maps(brain):
    if not hasattr(brain, "_cci_map"):
        brain._cci_map = dict(zip(brain.source_df["ts"].astype(int), brain.cci_series))

    if brain.filter_mode in ["kvo", "signal"] and not hasattr(brain, "_kvo_map"):
        brain._kvo_map = dict(zip(brain.source_df["ts"].astype(int), brain.kvo_series))
        brain._kvo_signal_map = dict(zip(brain.source_df["ts"].astype(int), brain.kvo_signal_series))


def _expected_intentions_for_bar(brain, ts: int):
    cci_val = brain._cci_map.get(ts, None)
    if cci_val is None or pd.isna(cci_val):
        return None, None, None, None, None, None, []

    allow_long = True
    allow_short = True
    kvo = None
    kvo_signal = None
    k_val = None

    if brain.filter_mode in ["kvo", "signal"]:
        kvo = brain._kvo_map.get(ts, 0.0)
        kvo_signal = brain._kvo_signal_map.get(ts, 0.0)
        k_val = kvo if brain.filter_mode == "kvo" else kvo_signal
        if k_val >= 0:
            allow_long = False
        if k_val <= 0:
            allow_short = False

    expected = []
    if cci_val < -100:
        if allow_long:
            expected.append(SignalIntent.LONG_CAN_INCREASE)
        expected.append(SignalIntent.SHORT_CAN_DECREASE)
    elif cci_val > 100:
        if allow_short:
            expected.append(SignalIntent.SHORT_CAN_INCREASE)
        expected.append(SignalIntent.LONG_CAN_DECREASE)

    return float(cci_val), kvo, kvo_signal, k_val, allow_long, allow_short, expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="LINKUSDT")
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--filter_mode", type=str, default="none", choices=["none", "kvo", "signal"])
    parser.add_argument("--start", type=str, default="2025-12-01")
    parser.add_argument("--end", type=str, default="2026-01-01")
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

    brain = CCI1hTf5mBrain(source_df=df, source_tf=args.timeframe, filter_mode=args.filter_mode)
    _get_indicator_maps(brain)

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
    mismatch_count = 0
    extreme_count = 0
    allowed_count = 0
    ok_count = 0

    for i, r in df.iterrows():
        ts_open_time = int(r["open_time"]) if "open_time" in r else None
        ts_ts = int(r["ts"]) if "ts" in r else None
        ts = ts_open_time if ts_open_time is not None else ts_ts
        if ts is None:
            continue
        dt = pd.to_datetime(ts, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S")

        cci_val, kvo, kvo_signal, k_val, allow_long, allow_short, expected = _expected_intentions_for_bar(brain, ts)
        actual = brain.get_intentions(ts, dummy_state)

        cci_row = None
        try:
            v = brain.cci_series.iloc[int(i)]
            if v is not None and not pd.isna(v):
                cci_row = float(v)
        except Exception:
            pass

        is_extreme = False
        if cci_val is not None:
            is_extreme = (cci_val < -100) or (cci_val > 100)
            if is_extreme:
                extreme_count += 1

        if cci_val is None:
            status = "MISSING_CCI"
            mismatch_count += 1
        elif is_extreme and expected and not actual:
            status = "EXTREME_BUT_NO_INTENTION"
            mismatch_count += 1
        elif is_extreme and _intentions_to_str(expected) != _intentions_to_str(actual):
            status = "MISMATCH"
            mismatch_count += 1
        elif actual:
            status = "OK"
            ok_count += 1
        else:
            status = ""

        if actual:
            allowed_count += 1

        rows.append(
            {
                "timestamp": ts,
                "dt_utc": dt,
                "ts_open_time": ts_open_time,
                "ts_ts": ts_ts,
                "open": float(r.get("open", 0.0)),
                "high": float(r.get("high", 0.0)),
                "low": float(r.get("low", 0.0)),
                "close": float(r.get("close", 0.0)),
                "volume": float(r.get("volume", 0.0)),
                "cci": cci_val,
                "cci_row": cci_row,
                "is_extreme": bool(is_extreme),
                "extreme_side": "neg" if (cci_val is not None and cci_val < -100) else ("pos" if (cci_val is not None and cci_val > 100) else ""),
                "ts_in_cci_map": bool(ts in brain._cci_map),
                "kvo": None if kvo is None else float(kvo),
                "kvo_signal": None if kvo_signal is None else float(kvo_signal),
                "k_val": None if k_val is None else float(k_val),
                "allow_long": None if allow_long is None else bool(allow_long),
                "allow_short": None if allow_short is None else bool(allow_short),
                "filter_mode": args.filter_mode,
                "expected_intentions": _intentions_to_str(expected),
                "actual_intentions": _intentions_to_str(actual),
                "status": status,
            }
        )

    out_df = pd.DataFrame(rows)

    out_dir = PROJECT_ROOT / "data" / "processed" / "backtests" / "perp_hedge"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"brain_signals_{args.symbol}_{args.timeframe}_{args.start}_{args.end}_filter-{args.filter_mode}.csv"
    out_df.to_csv(out_path, index=False)

    events_mask = (
        (out_df["is_extreme"] == True)
        | (out_df["actual_intentions"].fillna("") != "")
        | (out_df["status"].fillna("") != "")
    )
    out_events = out_df[events_mask].copy()
    out_events_path = out_dir / f"brain_signals_events_{args.symbol}_{args.timeframe}_{args.start}_{args.end}_filter-{args.filter_mode}.csv"
    out_events.to_csv(out_events_path, index=False)

    out_df["date_utc"] = out_df["dt_utc"].str.slice(0, 10)
    daily = out_df.groupby("date_utc").agg(
        bars=("timestamp", "count"),
        extreme_bars=("is_extreme", "sum"),
        ok=("status", lambda s: int((s == "OK").sum())),
        missing_cci=("status", lambda s: int((s == "MISSING_CCI").sum())),
        mismatches=("status", lambda s: int((s == "MISMATCH").sum() + (s == "EXTREME_BUT_NO_INTENTION").sum())),
    ).reset_index()

    out_daily_path = out_dir / f"brain_signals_daily_{args.symbol}_{args.timeframe}_{args.start}_{args.end}_filter-{args.filter_mode}.csv"
    daily.to_csv(out_daily_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Saved: {out_events_path}")
    print(f"Saved: {out_daily_path}")
    print(f"Bars: {len(df)}")
    print(f"CCI extreme bars: {extreme_count}")
    print(f"Bars with intentions: {allowed_count}")
    print(f"OK: {ok_count}")
    print(f"Mismatches: {mismatch_count}")


if __name__ == "__main__":
    main()
