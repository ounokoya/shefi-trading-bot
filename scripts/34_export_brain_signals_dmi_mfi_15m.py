import sys
from pathlib import Path
import argparse
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.data_loader import get_crypto_data
from libs.strategies.perp_hedge.models import AccountState, SignalIntent
from libs.strategies.perp_hedge.brains.dmi_mfi_tf_15m import DMIMFITf15mBrain


def _intentions_to_str(ints):
    if not ints:
        return ""
    return "|".join([i.name for i in ints])


def _expected_intentions_for_ts(brain: DMIMFITf15mBrain, ts_ms: int):
    dt = pd.to_datetime(int(ts_ms), unit="ms")

    if dt.minute % 15 != 0 or dt.second != 0:
        return None

    bar15 = dt.floor("15min") - pd.Timedelta(minutes=15)
    prev15 = bar15 - pd.Timedelta(minutes=15)

    df_ind = getattr(brain, "df_15m_ind", None)
    if df_ind is None:
        return None

    if bar15 not in df_ind.index or prev15 not in df_ind.index:
        return None

    row = df_ind.loc[bar15]
    prev = df_ind.loc[prev15]

    adx_now = row.get("adx")
    plus_di_now = row.get("plus_di")
    minus_di_now = row.get("minus_di")
    dx_now = row.get("dx")
    mfi_now = row.get("mfi")

    adx_prev = prev.get("adx")
    dx_prev = prev.get("dx")

    vals = [adx_now, plus_di_now, minus_di_now, dx_now, mfi_now, adx_prev, dx_prev]
    if any(pd.isna(v) for v in vals):
        return None

    adx_now = float(adx_now)
    plus_di_now = float(plus_di_now)
    minus_di_now = float(minus_di_now)
    dx_now = float(dx_now)
    mfi_now = float(mfi_now)

    adx_prev = float(adx_prev)
    dx_prev = float(dx_prev)

    maturity_mode = str(getattr(brain, "maturity_mode", "di_max") or "di_max").strip().lower()
    adx_min_threshold = float(getattr(brain, "adx_min_threshold", 20.0))

    if maturity_mode == "di_max":
        is_mature = adx_now > max(plus_di_now, minus_di_now)
    else:
        is_mature = adx_now >= adx_min_threshold
    dx_cross_under = (dx_prev > adx_prev) and (dx_now <= adx_now)

    direction = ""
    mfi_ok = False
    if plus_di_now > minus_di_now:
        direction = "bull"
        mfi_ok = mfi_now > 50.0
    elif minus_di_now > plus_di_now:
        direction = "bear"
        mfi_ok = mfi_now < 50.0

    expected = []
    if is_mature and dx_cross_under:
        if direction == "bull" and mfi_ok:
            expected = [SignalIntent.LONG_CAN_DECREASE, SignalIntent.SHORT_CAN_INCREASE]
        elif direction == "bear" and mfi_ok:
            expected = [SignalIntent.SHORT_CAN_DECREASE, SignalIntent.LONG_CAN_INCREASE]

    return {
        "bar15_dt": bar15,
        "prev15_dt": prev15,
        "adx": adx_now,
        "dx": dx_now,
        "plus_di": plus_di_now,
        "minus_di": minus_di_now,
        "mfi": mfi_now,
        "adx_prev": adx_prev,
        "dx_prev": dx_prev,
        "maturity_mode": maturity_mode,
        "adx_min_threshold": adx_min_threshold,
        "is_mature": bool(is_mature),
        "dx_cross_under": bool(dx_cross_under),
        "direction": direction,
        "mfi_ok": bool(mfi_ok),
        "expected": expected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="LINKUSDT")
    parser.add_argument("--timeframe", type=str, default="15m")
    parser.add_argument("--start", type=str, default="2026-01-01")
    parser.add_argument("--end", type=str, default="2026-01-18")
    parser.add_argument("--maturity_mode", type=str, default="di_max", choices=["di_max", "adx_threshold"])
    parser.add_argument("--adx_min_threshold", type=float, default=20.0)
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

    brain = DMIMFITf15mBrain(
        source_df=df,
        source_tf=args.timeframe,
        maturity_mode=args.maturity_mode,
        adx_min_threshold=args.adx_min_threshold,
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
    mismatch = 0

    for _, r in df.iterrows():
        ts = int(r.get("open_time", r.get("ts")))
        dt_utc = pd.to_datetime(ts, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S")

        expected_pack = _expected_intentions_for_ts(brain, ts)
        expected_ints = [] if not expected_pack else expected_pack["expected"]
        actual_ints = brain.get_intentions(ts, dummy_state)

        status = ""
        if expected_pack is None and actual_ints:
            status = "UNEXPECTED_INTENTION"
            mismatch += 1
        elif expected_pack is not None and _intentions_to_str(expected_ints) != _intentions_to_str(actual_ints):
            status = "MISMATCH"
            mismatch += 1

        row_out = {
            "timestamp": ts,
            "dt_utc": dt_utc,
            "open": float(r.get("open", 0.0)),
            "high": float(r.get("high", 0.0)),
            "low": float(r.get("low", 0.0)),
            "close": float(r.get("close", 0.0)),
            "volume": float(r.get("volume", 0.0)),
            "expected_intentions": _intentions_to_str(expected_ints),
            "actual_intentions": _intentions_to_str(actual_ints),
            "status": status,
        }

        if expected_pack is not None:
            row_out.update(
                {
                    "bar15_dt": str(expected_pack["bar15_dt"]),
                    "prev15_dt": str(expected_pack["prev15_dt"]),
                    "adx": expected_pack["adx"],
                    "dx": expected_pack["dx"],
                    "plus_di": expected_pack["plus_di"],
                    "minus_di": expected_pack["minus_di"],
                    "mfi": expected_pack["mfi"],
                    "adx_prev": expected_pack["adx_prev"],
                    "dx_prev": expected_pack["dx_prev"],
                    "maturity_mode": expected_pack["maturity_mode"],
                    "adx_min_threshold": expected_pack["adx_min_threshold"],
                    "is_mature": expected_pack["is_mature"],
                    "dx_cross_under": expected_pack["dx_cross_under"],
                    "direction": expected_pack["direction"],
                    "mfi_ok": expected_pack["mfi_ok"],
                }
            )

        rows.append(row_out)

    out_df = pd.DataFrame(rows)

    out_dir = PROJECT_ROOT / "data" / "processed" / "backtests" / "perp_hedge"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = f"dmi_mfi_signals_{args.symbol}_{args.timeframe}_{args.start}_{args.end}"

    out_all_path = out_dir / f"{base}_all_bars.csv"
    out_df.to_csv(out_all_path, index=False)

    out_signals = out_df[out_df["actual_intentions"].fillna("") != ""].copy()
    out_signals_path = out_dir / f"{base}_signals.csv"
    out_signals.to_csv(out_signals_path, index=False)

    out_events = out_df[(out_df["expected_intentions"].fillna("") != "") | (out_df["actual_intentions"].fillna("") != "") | (out_df["status"].fillna("") != "")].copy()
    out_events_path = out_dir / f"{base}_events.csv"
    out_events.to_csv(out_events_path, index=False)

    print(f"Saved: {out_all_path}")
    print(f"Saved: {out_signals_path}")
    print(f"Saved: {out_events_path}")
    print(f"Bars: {len(out_df)}")
    print(f"Signals: {len(out_signals)}")
    print(f"Mismatches: {mismatch}")


if __name__ == "__main__":
    main()
