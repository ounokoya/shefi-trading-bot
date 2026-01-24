from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

import numpy as np
import pandas as pd

from libs.indicators.momentum.cci_tv import cci_tv
from libs.indicators.momentum.dmi_tv import dmi_tv
from libs.indicators.momentum.vortex_tv import vortex_tv
from libs.indicators.volatility.atr_tv import atr_tv


def _leg_side(block_side: str, leg: str) -> str:
    if leg not in ("A", "B"):
        raise ValueError("leg must be A or B")
    if block_side == "LONG":
        return "LONG" if leg == "A" else "SHORT"
    if block_side == "SHORT":
        return "SHORT" if leg == "A" else "LONG"
    raise ValueError(f"Unexpected block_side: {block_side}")


def _ret_pct(entry: float, exit_: float, side: str) -> float:
    if entry == 0.0 or np.isnan(entry) or np.isnan(exit_):
        return float("nan")
    if side == "LONG":
        return (exit_ - entry) / entry
    if side == "SHORT":
        return (entry - exit_) / entry
    raise ValueError(f"Unexpected side: {side}")


def _floating_dd_series(entry: float, highs: np.ndarray, lows: np.ndarray, side: str) -> np.ndarray:
    if entry == 0.0 or np.isnan(entry):
        return np.array([], dtype=float)
    if side == "LONG":
        dd = (entry - lows) / entry
    elif side == "SHORT":
        dd = (highs - entry) / entry
    else:
        raise ValueError(f"Unexpected side: {side}")

    dd = np.where(np.isnan(dd), np.nan, np.maximum(dd, 0.0))
    return dd


def _bucket_metrics(r: pd.Series, target: float) -> dict[str, object]:
    r2 = pd.to_numeric(r, errors="coerce")
    r2 = r2.dropna()
    losers = r2[r2 < 0.0]
    small = r2[(r2 >= 0.0) & (r2 < float(target))]
    winners = r2[r2 >= float(target)]
    return {
        "n_trades": int(r2.shape[0]),
        "n_losers": int(losers.shape[0]),
        "n_small_winners": int(small.shape[0]),
        "n_target_winners": int(winners.shape[0]),
        "sum_losers": float(losers.sum()),
        "sum_small_winners": float(small.sum()),
        "sum_target_winners": float(winners.sum()),
    }


def _stop_threshold_pct(
    *,
    stop_mode: str,
    stop_pct: float,
    atr_val: float | None,
    atr_mult: float,
    entry: float,
) -> float | None:
    if stop_mode == "pct":
        return float(stop_pct)
    if stop_mode == "atr":
        if atr_val is None or (not np.isfinite(float(atr_val))) or entry == 0.0 or (not np.isfinite(float(entry))):
            return None
        return float(atr_mult) * float(atr_val) / float(entry)
    return None


def _stop_exit_price(*, side: str, entry: float, thr_pct: float) -> float:
    if side == "LONG":
        return float(entry) * (1.0 - float(thr_pct))
    if side == "SHORT":
        return float(entry) * (1.0 + float(thr_pct))
    raise ValueError(f"Unexpected side: {side}")


def _hx_trend_side(
    *,
    i: int,
    vi_plus: np.ndarray,
    vi_minus: np.ndarray,
    di_plus: np.ndarray,
    di_minus: np.ndarray,
) -> str:
    vp = float(vi_plus[i])
    vm = float(vi_minus[i])
    dp = float(di_plus[i])
    dm = float(di_minus[i])
    if (not np.isfinite(vp)) or (not np.isfinite(vm)) or (not np.isfinite(dp)) or (not np.isfinite(dm)):
        return "NEUTRE"
    if vp > vm and dp > dm:
        return "LONG"
    if vp < vm and dp < dm:
        return "SHORT"
    return "NEUTRE"


def _hx_cci_entry_ok(*, side: str, cci_fast: float, cci_mid: float) -> bool:
    if (not np.isfinite(float(cci_fast))) or (not np.isfinite(float(cci_mid))):
        return False
    if side == "LONG":
        return float(cci_mid) < 0.0 and float(cci_fast) < -100.0
    if side == "SHORT":
        return float(cci_mid) > 0.0 and float(cci_fast) > 100.0
    raise ValueError(f"Unexpected side: {side}")


def _tcci_exit_extreme_ok(*, side: str, cci_fast: float, cci_mid: float) -> bool:
    if (not np.isfinite(float(cci_fast))) or (not np.isfinite(float(cci_mid))):
        return False
    if side == "LONG":
        return float(cci_mid) > 0.0 and float(cci_fast) > 100.0
    if side == "SHORT":
        return float(cci_mid) < 0.0 and float(cci_fast) < -100.0
    raise ValueError(f"Unexpected side: {side}")


def _tcci_cci300_flat_ok(
    *,
    side: str,
    cci_slow: np.ndarray,
    i: int,
    k: int,
    eps: float,
    mode: str,
) -> bool:
    if int(k) <= 0:
        raise ValueError("tcci_flat_k must be >= 1")
    j = int(i) + int(k)
    if not (0 <= int(i) < len(cci_slow) and 0 <= int(j) < len(cci_slow)):
        return False
    v0 = float(cci_slow[int(i)])
    v1 = float(cci_slow[int(j)])
    if (not np.isfinite(v0)) or (not np.isfinite(v1)):
        return False
    if str(mode) == "abs_delta":
        return abs(v1 - v0) <= float(eps)
    if str(mode) == "non_worsening":
        if side == "LONG":
            return v1 >= (v0 - float(eps))
        if side == "SHORT":
            return v1 <= (v0 + float(eps))
        raise ValueError(f"Unexpected side: {side}")
    raise ValueError(f"Unexpected tcci_flat_mode: {mode}")


def _hx_cci_entry_ok_variant(
    *,
    variant: str,
    side: str,
    cci_fast: float,
    cci_mid: float,
    cci_slow: float,
) -> bool:
    if (
        (not np.isfinite(float(cci_fast)))
        or (not np.isfinite(float(cci_mid)))
        or (not np.isfinite(float(cci_slow)))
    ):
        return False
    v = str(variant)
    if v == "hx":
        return _hx_cci_entry_ok(side=side, cci_fast=cci_fast, cci_mid=cci_mid)
    if v == "hxz":
        if side == "LONG":
            return float(cci_fast) < 0.0 and float(cci_mid) < 0.0 and float(cci_slow) < 0.0
        if side == "SHORT":
            return float(cci_fast) > 0.0 and float(cci_mid) > 0.0 and float(cci_slow) > 0.0
        raise ValueError(f"Unexpected side: {side}")
    if v == "hxzt":
        if side == "LONG":
            return float(cci_fast) < 0.0 and float(cci_mid) < 0.0
        if side == "SHORT":
            return float(cci_fast) > 0.0 and float(cci_mid) > 0.0
        raise ValueError(f"Unexpected side: {side}")
    if v == "hxd":
        if side == "LONG":
            return float(cci_fast) < -100.0 and float(cci_mid) < 0.0 and float(cci_slow) < 0.0
        if side == "SHORT":
            return float(cci_fast) > 100.0 and float(cci_mid) > 0.0 and float(cci_slow) > 0.0
        raise ValueError(f"Unexpected side: {side}")
    if v == "hxdt":
        if side == "LONG":
            return float(cci_fast) < -100.0 and float(cci_mid) < 0.0
        if side == "SHORT":
            return float(cci_fast) > 100.0 and float(cci_mid) > 0.0
        raise ValueError(f"Unexpected side: {side}")
    raise ValueError(f"Unexpected hx variant: {variant}")


def _hx_cci_exit_ok_variant(
    *,
    variant: str,
    side: str,
    cci_fast: float,
    cci_mid: float,
    cci_slow: float,
) -> bool:
    if (
        (not np.isfinite(float(cci_fast)))
        or (not np.isfinite(float(cci_mid)))
        or (not np.isfinite(float(cci_slow)))
    ):
        return False
    v = str(variant)
    if v == "hx":
        return _hx_cci_exit_ok(side=side, cci_fast=cci_fast, cci_mid=cci_mid, cci_slow=cci_slow)
    if v == "hxz":
        if side == "LONG":
            return float(cci_fast) > 0.0 and float(cci_mid) > 0.0 and float(cci_slow) > 0.0
        if side == "SHORT":
            return float(cci_fast) < 0.0 and float(cci_mid) < 0.0 and float(cci_slow) < 0.0
        raise ValueError(f"Unexpected side: {side}")
    if v == "hxzt":
        if side == "LONG":
            return float(cci_fast) > 0.0 and float(cci_mid) > 0.0
        if side == "SHORT":
            return float(cci_fast) < 0.0 and float(cci_mid) < 0.0
        raise ValueError(f"Unexpected side: {side}")
    if v == "hxd":
        if side == "LONG":
            return float(cci_fast) > 100.0 and float(cci_mid) > 0.0 and float(cci_slow) > 0.0
        if side == "SHORT":
            return float(cci_fast) < -100.0 and float(cci_mid) < 0.0 and float(cci_slow) < 0.0
        raise ValueError(f"Unexpected side: {side}")
    if v == "hxdt":
        if side == "LONG":
            return float(cci_fast) > 100.0 and float(cci_mid) > 0.0
        if side == "SHORT":
            return float(cci_fast) < -100.0 and float(cci_mid) < 0.0
        raise ValueError(f"Unexpected side: {side}")
    raise ValueError(f"Unexpected hx variant: {variant}")


def _hx_cci_exit_ok(*, side: str, cci_fast: float, cci_mid: float, cci_slow: float) -> bool:
    if (
        (not np.isfinite(float(cci_fast)))
        or (not np.isfinite(float(cci_mid)))
        or (not np.isfinite(float(cci_slow)))
    ):
        return False
    if side == "LONG":
        return float(cci_mid) > 100.0 and float(cci_fast) > 100.0 and float(cci_slow) > 100.0
    if side == "SHORT":
        return float(cci_mid) < -100.0 and float(cci_fast) < -100.0 and float(cci_slow) < -100.0
    raise ValueError(f"Unexpected side: {side}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-csv",
        default="data/processed/klines/LINKUSDT_5m_2020-01-01_2024-12-31_with_macd_12_26_9_with_tranches_and_blocks.csv",
    )
    ap.add_argument("--out-legs-csv", default="")
    ap.add_argument("--out-trades-csv", default="")
    ap.add_argument("--mode", default="legs", choices=["legs", "trades"])
    ap.add_argument("--price-col", default="close", choices=["close", "open"])
    ap.add_argument("--target", type=float, default=0.007)
    ap.add_argument("--stop-mode", default="none", choices=["none", "pct", "atr"])
    ap.add_argument("--stop-pct", type=float, default=0.01)
    ap.add_argument("--stop-atr-period", type=int, default=14)
    ap.add_argument("--stop-atr-mult", type=float, default=2.0)
    ap.add_argument("--stop-fill", default="stop_level", choices=["stop_level", "close"])
    ap.add_argument("--hx-enable", action="store_true")
    ap.add_argument("--hx-variant", default="hx", choices=["hx", "hxz", "hxzt", "hxd", "hxdt"])
    ap.add_argument("--hx-vortex-period", type=int, default=300)
    ap.add_argument("--hx-dmi-period", type=int, default=300)
    ap.add_argument("--hx-cci-fast", type=int, default=30)
    ap.add_argument("--hx-cci-mid", type=int, default=100)
    ap.add_argument("--hx-cci-slow", type=int, default=300)
    ap.add_argument("--tcci-enable", action="store_true")
    ap.add_argument("--tcci-cci-fast", type=int, default=30)
    ap.add_argument("--tcci-cci-mid", type=int, default=100)
    ap.add_argument("--tcci-cci-slow", type=int, default=300)
    ap.add_argument("--tcci-flat-k", type=int, default=6)
    ap.add_argument("--tcci-flat-eps", type=float, default=20.0)
    ap.add_argument("--tcci-flat-mode", default="non_worsening", choices=["non_worsening", "abs_delta"])
    args = ap.parse_args()

    in_csv = Path(str(args.in_csv))
    if not in_csv.exists():
        raise FileNotFoundError(str(in_csv))

    out_legs = None
    out_trades = None
    if str(args.mode) == "legs":
        out_legs_str = str(args.out_legs_csv).strip()
        if out_legs_str:
            out_legs = Path(out_legs_str)
        else:
            out_legs = Path(str(in_csv).replace(".csv", "_legs.csv"))
        out_legs.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_trades_str = str(args.out_trades_csv).strip()
        if out_trades_str:
            out_trades = Path(out_trades_str)
        else:
            stop_mode = str(args.stop_mode)
            stop_fill = str(args.stop_fill)
            stop_suffix = ""
            if stop_mode == "pct":
                stop_suffix = f"_stop{(float(args.stop_pct) * 100.0):g}pct_{stop_fill}"
            elif stop_mode == "atr":
                stop_suffix = f"_stop{float(args.stop_atr_mult):g}atr{int(args.stop_atr_period)}_{stop_fill}"

            hx_suffix = ""
            if bool(args.hx_enable):
                hx_tag = str(args.hx_variant)
                hx_prefix = "_hx" if hx_tag == "hx" else f"_{hx_tag}"
                hx_fast = int(args.hx_cci_fast)
                hx_mid = int(args.hx_cci_mid)
                hx_slow = int(args.hx_cci_slow)
                if hx_tag in ("hxz", "hxzt", "hxd", "hxdt"):
                    hx_fast, hx_mid, hx_slow = 30, 120, 300
                hx_suffix = (
                    f"{hx_prefix}v{int(args.hx_vortex_period)}d{int(args.hx_dmi_period)}"
                    f"c{hx_fast}_{hx_mid}_{hx_slow}"
                )

            tcci_suffix = ""
            if bool(args.tcci_enable):
                tcci_suffix = (
                    f"_tcci{int(args.tcci_cci_fast)}_{int(args.tcci_cci_mid)}_{int(args.tcci_cci_slow)}"
                    f"k{int(args.tcci_flat_k)}e{float(args.tcci_flat_eps):g}{str(args.tcci_flat_mode)[:2]}"
                )

            out_trades = Path(str(in_csv).replace(".csv", f"_trades{stop_suffix}{hx_suffix}{tcci_suffix}.csv"))
        out_trades.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)

    required = ["ts", "open", "high", "low", "close"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    has_event_cols = (
        "event_t0_block_id" in df.columns
        and "event_tfav_block_id" in df.columns
        and "event_t1_block_id" in df.columns
        and "event_t0_side" in df.columns
        and "event_tfav_side" in df.columns
        and "event_t1_side" in df.columns
    )

    if not has_event_cols:
        raise ValueError(
            "Missing event columns. Expected event_t0_block_id/event_tfav_block_id/event_t1_block_id and side columns. "
            "(Make sure you generated the dataset via scripts/04_add_tranches_and_blocks.py)"
        )

    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    if df["ts"].isna().any():
        raise ValueError("ts contains NaN after numeric coercion")
    df = df.sort_values("ts").reset_index(drop=True)

    ts = df["ts"].astype("int64").to_numpy()
    open_ = pd.to_numeric(df["open"], errors="coerce").astype(float).to_numpy()
    high = pd.to_numeric(df["high"], errors="coerce").astype(float).to_numpy()
    low = pd.to_numeric(df["low"], errors="coerce").astype(float).to_numpy()
    close = pd.to_numeric(df["close"], errors="coerce").astype(float).to_numpy()

    price = close if str(args.price_col) == "close" else open_

    def _idx_for_ts(t: int) -> int | None:
        i = int(np.searchsorted(ts, int(t), side="left"))
        if 0 <= i < len(ts) and int(ts[i]) == int(t):
            return i
        return None

    block_ids = (
        pd.to_numeric(df["event_tfav_block_id"], errors="coerce")
        .dropna()
        .astype("int64")
        .unique()
        .tolist()
    )
    block_ids = sorted(set(int(b) for b in block_ids))

    stop_mode = str(args.stop_mode)
    if stop_mode not in ("none", "pct", "atr"):
        raise ValueError("stop_mode must be one of: none, pct, atr")

    atr_col = f"atr_{int(args.stop_atr_period)}"
    atr = None
    if str(args.mode) == "trades" and stop_mode == "atr":
        if atr_col in df.columns:
            atr = pd.to_numeric(df[atr_col], errors="coerce").astype(float).to_numpy()
        else:
            atr = np.array(
                atr_tv(high.tolist(), low.tolist(), close.tolist(), int(args.stop_atr_period)),
                dtype=float,
            )

    hx_enable = bool(args.hx_enable) and str(args.mode) == "trades"
    hx_variant = str(args.hx_variant)
    hx_vi_plus = None
    hx_vi_minus = None
    hx_di_plus = None
    hx_di_minus = None
    hx_cci_fast = None
    hx_cci_mid = None
    hx_cci_slow = None

    hx_cci_fast_p = int(args.hx_cci_fast)
    hx_cci_mid_p = int(args.hx_cci_mid)
    hx_cci_slow_p = int(args.hx_cci_slow)

    hx_skipped_total = 0
    hx_skipped_trend_neutral = 0
    hx_skipped_trend_mismatch = 0
    hx_skipped_cci_entry = 0
    hx_skipped_cci_exit = 0

    if hx_enable:
        vx_p = int(args.hx_vortex_period)
        dmi_p = int(args.hx_dmi_period)
        if hx_variant in ("hxz", "hxzt", "hxd", "hxdt"):
            hx_cci_fast_p = 30
            hx_cci_mid_p = 120
            hx_cci_slow_p = 300

        vx_plus_col = f"vortex_plus_{vx_p}"
        vx_minus_col = f"vortex_minus_{vx_p}"
        di_plus_col = f"plus_di_{dmi_p}"
        di_minus_col = f"minus_di_{dmi_p}"
        cci_fast_col = f"cci_{hx_cci_fast_p}"
        cci_mid_col = f"cci_{hx_cci_mid_p}"
        cci_slow_col = f"cci_{hx_cci_slow_p}"

        if vx_plus_col in df.columns and vx_minus_col in df.columns:
            hx_vi_plus = pd.to_numeric(df[vx_plus_col], errors="coerce").astype(float).to_numpy()
            hx_vi_minus = pd.to_numeric(df[vx_minus_col], errors="coerce").astype(float).to_numpy()
        else:
            vp, vm = vortex_tv(high.tolist(), low.tolist(), close.tolist(), vx_p)
            hx_vi_plus = np.asarray(vp, dtype=float)
            hx_vi_minus = np.asarray(vm, dtype=float)

        if di_plus_col in df.columns and di_minus_col in df.columns:
            hx_di_plus = pd.to_numeric(df[di_plus_col], errors="coerce").astype(float).to_numpy()
            hx_di_minus = pd.to_numeric(df[di_minus_col], errors="coerce").astype(float).to_numpy()
        else:
            _, dp, dm = dmi_tv(high.tolist(), low.tolist(), close.tolist(), dmi_p)
            hx_di_plus = np.asarray(dp, dtype=float)
            hx_di_minus = np.asarray(dm, dtype=float)

        if cci_fast_col in df.columns:
            hx_cci_fast = pd.to_numeric(df[cci_fast_col], errors="coerce").astype(float).to_numpy()
        else:
            hx_cci_fast = np.asarray(cci_tv(high.tolist(), low.tolist(), close.tolist(), hx_cci_fast_p), dtype=float)

        if cci_mid_col in df.columns:
            hx_cci_mid = pd.to_numeric(df[cci_mid_col], errors="coerce").astype(float).to_numpy()
        else:
            hx_cci_mid = np.asarray(cci_tv(high.tolist(), low.tolist(), close.tolist(), hx_cci_mid_p), dtype=float)

        if cci_slow_col in df.columns:
            hx_cci_slow = pd.to_numeric(df[cci_slow_col], errors="coerce").astype(float).to_numpy()
        else:
            hx_cci_slow = np.asarray(cci_tv(high.tolist(), low.tolist(), close.tolist(), hx_cci_slow_p), dtype=float)

    tcci_enable = bool(args.tcci_enable) and str(args.mode) == "trades"
    tcci_cci_fast = None
    tcci_cci_mid = None
    tcci_cci_slow = None
    tcci_skipped_total = 0
    tcci_skipped_entry_extreme = 0
    tcci_skipped_entry_flat = 0
    tcci_skipped_exit_extreme = 0
    tcci_skipped_exit_flat = 0

    if tcci_enable:
        tcci_fast_p = int(args.tcci_cci_fast)
        tcci_mid_p = int(args.tcci_cci_mid)
        tcci_slow_p = int(args.tcci_cci_slow)
        tcci_fast_col = f"cci_{tcci_fast_p}"
        tcci_mid_col = f"cci_{tcci_mid_p}"
        tcci_slow_col = f"cci_{tcci_slow_p}"

        if tcci_fast_col in df.columns:
            tcci_cci_fast = pd.to_numeric(df[tcci_fast_col], errors="coerce").astype(float).to_numpy()
        else:
            tcci_cci_fast = np.asarray(cci_tv(high.tolist(), low.tolist(), close.tolist(), tcci_fast_p), dtype=float)

        if tcci_mid_col in df.columns:
            tcci_cci_mid = pd.to_numeric(df[tcci_mid_col], errors="coerce").astype(float).to_numpy()
        else:
            tcci_cci_mid = np.asarray(cci_tv(high.tolist(), low.tolist(), close.tolist(), tcci_mid_p), dtype=float)

        if tcci_slow_col in df.columns:
            tcci_cci_slow = pd.to_numeric(df[tcci_slow_col], errors="coerce").astype(float).to_numpy()
        else:
            tcci_cci_slow = np.asarray(cci_tv(high.tolist(), low.tolist(), close.tolist(), tcci_slow_p), dtype=float)

    legs_rows: list[dict[str, object]] = []
    trades_rows: list[dict[str, object]] = []
    skipped = 0

    for b in block_ids:
        m0 = df["event_t0_block_id"] == b
        mf = df["event_tfav_block_id"] == b
        m1 = df["event_t1_block_id"] == b

        if int(m0.sum()) != 1 or int(mf.sum()) != 1 or int(m1.sum()) != 1:
            skipped += 1
            continue

        t0_ts = int(df.loc[m0, "ts"].iloc[0])
        tfav_ts = int(df.loc[mf, "ts"].iloc[0])
        t1_ts = int(df.loc[m1, "ts"].iloc[0])

        side_values = (
            df.loc[m0, "event_t0_side"].dropna().tolist()
            + df.loc[mf, "event_tfav_side"].dropna().tolist()
            + df.loc[m1, "event_t1_side"].dropna().tolist()
        )
        block_side = str(side_values[0]) if side_values else None
        if block_side not in ("LONG", "SHORT"):
            skipped += 1
            continue

        i0 = _idx_for_ts(t0_ts)
        ifv = _idx_for_ts(tfav_ts)
        i1 = _idx_for_ts(t1_ts)
        if i0 is None or ifv is None or i1 is None:
            skipped += 1
            continue

        if str(args.mode) == "legs":
            def _emit_leg(leg: str, start_i: int, end_i: int) -> None:
                if start_i > end_i:
                    return

                leg_side = _leg_side(block_side, leg)

                entry = float(price[start_i])
                exit_ = float(price[end_i])
                r = _ret_pct(entry, exit_, leg_side)

                sl = slice(start_i, end_i + 1)
                dd_series = _floating_dd_series(entry, high[sl], low[sl], leg_side)

                dd_min = float(np.nanmin(dd_series)) if dd_series.size else float("nan")
                dd_max = float(np.nanmax(dd_series)) if dd_series.size else float("nan")
                dd_mean = float(np.nanmean(dd_series)) if dd_series.size else float("nan")

                legs_rows.append(
                    {
                        "block_id": int(b),
                        "block_side": block_side,
                        "leg": leg,
                        "leg_side": leg_side,
                        "start_ts": int(ts[start_i]),
                        "end_ts": int(ts[end_i]),
                        "n_bars": int(end_i - start_i + 1),
                        "entry_price": entry,
                        "exit_price": exit_,
                        "ret_pct": r,
                        "dd_float_min": dd_min,
                        "dd_float_max": dd_max,
                        "dd_float_mean": dd_mean,
                        "meets_target": int((not np.isnan(r)) and (float(r) >= float(args.target))),
                    }
                )

            _emit_leg("A", i0, ifv)
            _emit_leg("B", ifv, i1)
        else:
            tfav_tranche_id = None
            if "event_tfav_tranche_id" in df.columns:
                tv = pd.to_numeric(df.loc[mf, "event_tfav_tranche_id"], errors="coerce")
                if len(tv.dropna()) == 1:
                    tfav_tranche_id = int(tv.dropna().iloc[0])

            tranche_b_mask = None
            if tfav_tranche_id is not None and "tranche_id" in df.columns:
                tranche_b_mask = (pd.to_numeric(df["tranche_id"], errors="coerce") == float(tfav_tranche_id)).fillna(False)

            cand_idx = []
            if tranche_b_mask is not None:
                if "tranche_cand_is_event" not in df.columns:
                    raise ValueError("Missing tranche_cand_is_event column required for trades mode")
                cm = tranche_b_mask & (pd.to_numeric(df["tranche_cand_is_event"], errors="coerce").fillna(0).astype(int) == 1)
                cand_idx = df.index[cm].to_list()
                cand_idx = [int(i) for i in cand_idx if int(i) >= int(ifv)]
                cand_idx.sort()

            def _hx_entry_gate(entry_i: int, leg_side: str) -> bool:
                nonlocal hx_skipped_total
                nonlocal hx_skipped_trend_neutral
                nonlocal hx_skipped_trend_mismatch
                nonlocal hx_skipped_cci_entry

                if not hx_enable:
                    return True
                if hx_vi_plus is None or hx_vi_minus is None or hx_di_plus is None or hx_di_minus is None:
                    raise ValueError("Hx enabled but missing trend indicators")
                if hx_cci_fast is None or hx_cci_mid is None or hx_cci_slow is None:
                    raise ValueError("Hx enabled but missing CCI indicators")

                trend_side = _hx_trend_side(
                    i=int(entry_i),
                    vi_plus=hx_vi_plus,
                    vi_minus=hx_vi_minus,
                    di_plus=hx_di_plus,
                    di_minus=hx_di_minus,
                )
                if trend_side == "NEUTRE":
                    hx_skipped_total += 1
                    hx_skipped_trend_neutral += 1
                    return False
                if str(leg_side) != str(trend_side):
                    hx_skipped_total += 1
                    hx_skipped_trend_mismatch += 1
                    return False

                if not _hx_cci_entry_ok_variant(
                    variant=str(hx_variant),
                    side=str(leg_side),
                    cci_fast=float(hx_cci_fast[int(entry_i)]),
                    cci_mid=float(hx_cci_mid[int(entry_i)]),
                    cci_slow=float(hx_cci_slow[int(entry_i)]),
                ):
                    hx_skipped_total += 1
                    hx_skipped_cci_entry += 1
                    return False

                return True

            def _tcci_entry_gate(entry_i: int, leg_side: str) -> bool:
                nonlocal tcci_skipped_total
                nonlocal tcci_skipped_entry_extreme
                nonlocal tcci_skipped_entry_flat

                if not tcci_enable:
                    return True
                if tcci_cci_fast is None or tcci_cci_mid is None or tcci_cci_slow is None:
                    raise ValueError("tcci enabled but missing CCI arrays")

                if not _hx_cci_entry_ok(
                    side=str(leg_side),
                    cci_fast=float(tcci_cci_fast[int(entry_i)]),
                    cci_mid=float(tcci_cci_mid[int(entry_i)]),
                ):
                    tcci_skipped_total += 1
                    tcci_skipped_entry_extreme += 1
                    return False

                if not _tcci_cci300_flat_ok(
                    side=str(leg_side),
                    cci_slow=tcci_cci_slow,
                    i=int(entry_i),
                    k=int(args.tcci_flat_k),
                    eps=float(args.tcci_flat_eps),
                    mode=str(args.tcci_flat_mode),
                ):
                    tcci_skipped_total += 1
                    tcci_skipped_entry_flat += 1
                    return False

                return True

            def _tcci_exit_gate(exit_i: int, leg_side: str) -> bool:
                nonlocal tcci_skipped_total
                nonlocal tcci_skipped_exit_extreme
                nonlocal tcci_skipped_exit_flat

                if not tcci_enable:
                    return True
                if tcci_cci_fast is None or tcci_cci_mid is None or tcci_cci_slow is None:
                    raise ValueError("tcci enabled but missing CCI arrays")

                if not _tcci_exit_extreme_ok(
                    side=str(leg_side),
                    cci_fast=float(tcci_cci_fast[int(exit_i)]),
                    cci_mid=float(tcci_cci_mid[int(exit_i)]),
                ):
                    tcci_skipped_total += 1
                    tcci_skipped_exit_extreme += 1
                    return False

                if not _tcci_cci300_flat_ok(
                    side=str(leg_side),
                    cci_slow=tcci_cci_slow,
                    i=int(exit_i),
                    k=int(args.tcci_flat_k),
                    eps=float(args.tcci_flat_eps),
                    mode=str(args.tcci_flat_mode),
                ):
                    tcci_skipped_total += 1
                    tcci_skipped_exit_flat += 1
                    return False

                return True

            def _hx_exit_gate(exit_i: int, leg_side: str) -> bool:
                nonlocal hx_skipped_total
                nonlocal hx_skipped_cci_exit

                if not hx_enable:
                    return True
                if hx_cci_fast is None or hx_cci_mid is None or hx_cci_slow is None:
                    raise ValueError("Hx enabled but missing CCI indicators")

                if not _hx_cci_exit_ok_variant(
                    variant=str(hx_variant),
                    side=str(leg_side),
                    cci_fast=float(hx_cci_fast[int(exit_i)]),
                    cci_mid=float(hx_cci_mid[int(exit_i)]),
                    cci_slow=float(hx_cci_slow[int(exit_i)]),
                ):
                    hx_skipped_total += 1
                    hx_skipped_cci_exit += 1
                    return False

                return True

            def _emit_trade(
                *,
                leg: str,
                entry_i: int,
                exit_i: int,
                leg_side: str,
                exit_price_override: float | None,
                stopped: int,
                attempt: int,
            ) -> None:
                entry = float(price[entry_i])
                exit_ = float(price[exit_i]) if exit_price_override is None else float(exit_price_override)
                r = _ret_pct(entry, exit_, leg_side)

                sl = slice(int(entry_i), int(exit_i) + 1)
                dd_series = _floating_dd_series(entry, high[sl], low[sl], leg_side)

                dd_min = float(np.nanmin(dd_series)) if dd_series.size else float("nan")
                dd_max = float(np.nanmax(dd_series)) if dd_series.size else float("nan")
                dd_mean = float(np.nanmean(dd_series)) if dd_series.size else float("nan")

                trades_rows.append(
                    {
                        "block_id": int(b),
                        "block_side": block_side,
                        "leg": leg,
                        "leg_side": leg_side,
                        "attempt": int(attempt),
                        "stopped": int(stopped),
                        "entry_ts": int(ts[int(entry_i)]),
                        "exit_ts": int(ts[int(exit_i)]),
                        "n_bars": int(int(exit_i) - int(entry_i) + 1),
                        "entry_price": entry,
                        "exit_price": exit_,
                        "ret_pct": r,
                        "dd_float_min": dd_min,
                        "dd_float_max": dd_max,
                        "dd_float_mean": dd_mean,
                        "meets_target": int((not np.isnan(r)) and (float(r) >= float(args.target))),
                    }
                )

            leg_a_side = _leg_side(block_side, "A")
            if (
                _hx_entry_gate(int(i0), leg_a_side)
                and _tcci_entry_gate(int(i0), leg_a_side)
                and _hx_exit_gate(int(ifv), leg_a_side)
                and _tcci_exit_gate(int(ifv), leg_a_side)
            ):
                _emit_trade(
                    leg="A",
                    entry_i=int(i0),
                    exit_i=int(ifv),
                    leg_side=leg_a_side,
                    exit_price_override=None,
                    stopped=0,
                    attempt=1,
                )

            leg_b_side = _leg_side(block_side, "B")
            if not cand_idx:
                if (
                    _hx_entry_gate(int(ifv), leg_b_side)
                    and _tcci_entry_gate(int(ifv), leg_b_side)
                    and _hx_exit_gate(int(i1), leg_b_side)
                    and _tcci_exit_gate(int(i1), leg_b_side)
                ):
                    _emit_trade(
                        leg="B",
                        entry_i=int(ifv),
                        exit_i=int(i1),
                        leg_side=leg_b_side,
                        exit_price_override=None,
                        stopped=0,
                        attempt=1,
                    )
            else:
                attempt = 1
                executed_final = False

                end_open_i = None
                if tranche_b_mask is not None:
                    idxs = df.index[tranche_b_mask].to_list()
                    if idxs:
                        end_open_i = int(max(int(x) for x in idxs))
                if end_open_i is None:
                    end_open_i = int(ifv)

                hist_series = pd.to_numeric(df["macd_hist"], errors="coerce").astype(float).to_numpy()

                ptr = 0
                while ptr < len(cand_idx):
                    entry_i = int(cand_idx[ptr])
                    entry = float(price[entry_i])
                    if not np.isfinite(entry) or entry == 0.0:
                        ptr += 1
                        attempt += 1
                        continue

                    if not _hx_entry_gate(int(entry_i), leg_b_side):
                        ptr += 1
                        attempt += 1
                        continue

                    if not _tcci_entry_gate(int(entry_i), leg_b_side):
                        ptr += 1
                        attempt += 1
                        continue

                    atr_val = float(atr[entry_i]) if (atr is not None and 0 <= entry_i < len(atr)) else None
                    thr_pct = _stop_threshold_pct(
                        stop_mode=stop_mode,
                        stop_pct=float(args.stop_pct),
                        atr_val=atr_val,
                        atr_mult=float(args.stop_atr_mult),
                        entry=entry,
                    )

                    stop_i = None
                    if stop_mode != "none" and thr_pct is not None and np.isfinite(float(thr_pct)) and float(thr_pct) > 0:
                        opposite_started = False
                        for i in range(entry_i + 1, int(end_open_i) + 1):
                            if i <= 0 or i >= len(hist_series):
                                continue
                            dh = float(hist_series[i] - hist_series[i - 1])
                            slope = 0.0
                            if np.isfinite(dh):
                                slope = float(np.sign(dh))

                            if not opposite_started:
                                if leg_b_side == "LONG" and slope < 0:
                                    opposite_started = True
                                elif leg_b_side == "SHORT" and slope > 0:
                                    opposite_started = True

                            if opposite_started:
                                dd = None
                                if leg_b_side == "LONG":
                                    dd = (entry - float(low[i])) / entry
                                else:
                                    dd = (float(high[i]) - entry) / entry
                                if np.isfinite(dd) and float(dd) >= float(thr_pct):
                                    stop_i = int(i)
                                    break

                    if stop_i is not None:
                        exit_price_override = None
                        if str(args.stop_fill) == "stop_level" and thr_pct is not None and np.isfinite(float(thr_pct)):
                            exit_price_override = _stop_exit_price(side=leg_b_side, entry=entry, thr_pct=float(thr_pct))
                        _emit_trade(
                            leg="B",
                            entry_i=int(entry_i),
                            exit_i=int(stop_i),
                            leg_side=leg_b_side,
                            exit_price_override=exit_price_override,
                            stopped=1,
                            attempt=int(attempt),
                        )

                        nxt = None
                        for j in range(ptr + 1, len(cand_idx)):
                            if int(cand_idx[j]) > int(stop_i):
                                nxt = j
                                break
                        if nxt is None:
                            break
                        ptr = int(nxt)
                        attempt += 1
                        continue

                    if not _hx_exit_gate(int(i1), leg_b_side):
                        ptr += 1
                        attempt += 1
                        continue

                    if not _tcci_exit_gate(int(i1), leg_b_side):
                        ptr += 1
                        attempt += 1
                        continue

                    _emit_trade(
                        leg="B",
                        entry_i=int(entry_i),
                        exit_i=int(i1),
                        leg_side=leg_b_side,
                        exit_price_override=None,
                        stopped=0,
                        attempt=int(attempt),
                    )
                    executed_final = True
                    break

                if not executed_final and stop_mode == "none":
                    if (
                        _hx_entry_gate(int(ifv), leg_b_side)
                        and _tcci_entry_gate(int(ifv), leg_b_side)
                        and _hx_exit_gate(int(i1), leg_b_side)
                        and _tcci_exit_gate(int(i1), leg_b_side)
                    ):
                        _emit_trade(
                            leg="B",
                            entry_i=int(ifv),
                            exit_i=int(i1),
                            leg_side=leg_b_side,
                            exit_price_override=None,
                            stopped=0,
                            attempt=1,
                        )

    summary: dict[str, object] = {
        "in_csv": str(in_csv),
        "mode": str(args.mode),
        "n_blocks": int(len(block_ids)),
        "n_blocks_skipped": int(skipped),
    }

    if str(args.mode) == "legs":
        legs_df = pd.DataFrame(legs_rows)
        if out_legs is None:
            raise ValueError("out_legs must be set in legs mode")
        legs_df.to_csv(out_legs, index=False)
        summary.update({"out_legs_csv": str(out_legs), "n_legs": int(len(legs_df))})

        if len(legs_df) > 0:
            r = pd.to_numeric(legs_df["ret_pct"], errors="coerce")
            ddmax = pd.to_numeric(legs_df["dd_float_max"], errors="coerce")
            ddmin = pd.to_numeric(legs_df["dd_float_min"], errors="coerce")
            ddmean = pd.to_numeric(legs_df["dd_float_mean"], errors="coerce")
            meets = pd.to_numeric(legs_df["meets_target"], errors="coerce").fillna(0).astype(int)

            summary.update(
                {
                    "target": float(args.target),
                    "capture_pct_sum": float(r.dropna().sum()),
                    "capture_pct_sum_meeting_target": float(r[meets == 1].dropna().sum()),
                    "capture_pct_mean": float(r.dropna().mean()) if int(r.dropna().shape[0]) > 0 else None,
                    "capture_pct_mean_meeting_target": float(r[meets == 1].dropna().mean())
                    if int(r[meets == 1].dropna().shape[0]) > 0
                    else None,
                    "winrate_meeting_target": float(meets.mean()) if len(meets) else None,
                    **_bucket_metrics(r, float(args.target)),
                    "dd_max": float(ddmax.dropna().max()) if int(ddmax.dropna().shape[0]) > 0 else None,
                    "dd_float_min": float(ddmin.dropna().min()) if int(ddmin.dropna().shape[0]) > 0 else None,
                    "dd_float_max": float(ddmax.dropna().max()) if int(ddmax.dropna().shape[0]) > 0 else None,
                    "dd_float_mean": float(ddmean.dropna().mean()) if int(ddmean.dropna().shape[0]) > 0 else None,
                }
            )

            by_leg = (
                legs_df.assign(ret_pct=r, dd_float_max=ddmax, dd_float_mean=ddmean, meets_target=meets)
                .groupby(["leg", "leg_side"], dropna=False)
                .agg(
                    n_legs=("ret_pct", "size"),
                    capture_sum=("ret_pct", "sum"),
                    capture_mean=("ret_pct", "mean"),
                    capture_sum_meeting_target=("ret_pct", lambda s: float(s[meets.loc[s.index] == 1].sum())),
                    winrate=("meets_target", "mean"),
                    dd_max=("dd_float_max", "max"),
                    dd_mean=("dd_float_mean", "mean"),
                    n_losers=("ret_pct", lambda s: int((pd.to_numeric(s, errors="coerce") < 0.0).fillna(False).sum())),
                    n_small_winners=(
                        "ret_pct",
                        lambda s: int(
                            (
                                (pd.to_numeric(s, errors="coerce") >= 0.0)
                                & (pd.to_numeric(s, errors="coerce") < float(args.target))
                            )
                            .fillna(False)
                            .sum()
                        ),
                    ),
                    n_target_winners=(
                        "ret_pct",
                        lambda s: int((pd.to_numeric(s, errors="coerce") >= float(args.target)).fillna(False).sum()),
                    ),
                    sum_losers=(
                        "ret_pct",
                        lambda s: float(
                            pd.to_numeric(s, errors="coerce")[(pd.to_numeric(s, errors="coerce") < 0.0)].dropna().sum()
                        ),
                    ),
                    sum_small_winners=(
                        "ret_pct",
                        lambda s: float(
                            pd.to_numeric(s, errors="coerce")[
                                (pd.to_numeric(s, errors="coerce") >= 0.0)
                                & (pd.to_numeric(s, errors="coerce") < float(args.target))
                            ]
                            .dropna()
                            .sum()
                        ),
                    ),
                    sum_target_winners=(
                        "ret_pct",
                        lambda s: float(
                            pd.to_numeric(s, errors="coerce")[(pd.to_numeric(s, errors="coerce") >= float(args.target))]
                            .dropna()
                            .sum()
                        ),
                    ),
                )
                .reset_index()
            )
            summary["by_leg"] = by_leg.to_dict(orient="records")
    else:
        trades_df = pd.DataFrame(trades_rows)
        if out_trades is None:
            raise ValueError("out_trades must be set in trades mode")
        trades_df.to_csv(out_trades, index=False)
        summary.update(
            {
                "out_trades_csv": str(out_trades),
                "target": float(args.target),
                "stop_mode": str(args.stop_mode),
                "stop_pct": float(args.stop_pct),
                "stop_atr_period": int(args.stop_atr_period),
                "stop_atr_mult": float(args.stop_atr_mult),
                "stop_fill": str(args.stop_fill),
                "hx_enable": bool(hx_enable),
                "hx_variant": str(hx_variant),
                "hx_vortex_period": int(args.hx_vortex_period),
                "hx_dmi_period": int(args.hx_dmi_period),
                "hx_cci_fast": int(hx_cci_fast_p),
                "hx_cci_mid": int(hx_cci_mid_p),
                "hx_cci_slow": int(hx_cci_slow_p),
                "hx_skipped_total": int(hx_skipped_total),
                "hx_skipped_trend_neutral": int(hx_skipped_trend_neutral),
                "hx_skipped_trend_mismatch": int(hx_skipped_trend_mismatch),
                "hx_skipped_cci_entry": int(hx_skipped_cci_entry),
                "hx_skipped_cci_exit": int(hx_skipped_cci_exit),
                "tcci_enable": bool(tcci_enable),
                "tcci_cci_fast": int(args.tcci_cci_fast),
                "tcci_cci_mid": int(args.tcci_cci_mid),
                "tcci_cci_slow": int(args.tcci_cci_slow),
                "tcci_flat_k": int(args.tcci_flat_k),
                "tcci_flat_eps": float(args.tcci_flat_eps),
                "tcci_flat_mode": str(args.tcci_flat_mode),
                "tcci_skipped_total": int(tcci_skipped_total),
                "tcci_skipped_entry_extreme": int(tcci_skipped_entry_extreme),
                "tcci_skipped_entry_flat": int(tcci_skipped_entry_flat),
                "tcci_skipped_exit_extreme": int(tcci_skipped_exit_extreme),
                "tcci_skipped_exit_flat": int(tcci_skipped_exit_flat),
                "n_trades": int(len(trades_df)),
            }
        )

        if len(trades_df) > 0:
            r = pd.to_numeric(trades_df["ret_pct"], errors="coerce")
            ddmax = pd.to_numeric(trades_df["dd_float_max"], errors="coerce")
            ddmin = pd.to_numeric(trades_df["dd_float_min"], errors="coerce")
            ddmean = pd.to_numeric(trades_df["dd_float_mean"], errors="coerce")
            meets = pd.to_numeric(trades_df["meets_target"], errors="coerce").fillna(0).astype(int)
            stopped = pd.to_numeric(trades_df["stopped"], errors="coerce").fillna(0).astype(int)

            summary.update(
                {
                    "capture_pct_sum": float(r.dropna().sum()),
                    "capture_pct_sum_meeting_target": float(r[meets == 1].dropna().sum()),
                    "capture_pct_mean": float(r.dropna().mean()) if int(r.dropna().shape[0]) > 0 else None,
                    "capture_pct_mean_meeting_target": float(r[meets == 1].dropna().mean())
                    if int(r[meets == 1].dropna().shape[0]) > 0
                    else None,
                    "winrate_meeting_target": float(meets.mean()) if len(meets) else None,
                    **_bucket_metrics(r, float(args.target)),
                    "n_stop_exits": int(stopped.sum()),
                    "sum_stop_exits": float(r[stopped == 1].dropna().sum()),
                    "dd_max": float(ddmax.dropna().max()) if int(ddmax.dropna().shape[0]) > 0 else None,
                    "dd_float_min": float(ddmin.dropna().min()) if int(ddmin.dropna().shape[0]) > 0 else None,
                    "dd_float_max": float(ddmax.dropna().max()) if int(ddmax.dropna().shape[0]) > 0 else None,
                    "dd_float_mean": float(ddmean.dropna().mean()) if int(ddmean.dropna().shape[0]) > 0 else None,
                }
            )

            by_leg = (
                trades_df.assign(ret_pct=r, dd_float_max=ddmax, dd_float_mean=ddmean, meets_target=meets)
                .groupby(["leg", "leg_side"], dropna=False)
                .agg(
                    n_trades=("ret_pct", "size"),
                    capture_sum=("ret_pct", "sum"),
                    capture_mean=("ret_pct", "mean"),
                    capture_sum_meeting_target=("ret_pct", lambda s: float(s[meets.loc[s.index] == 1].sum())),
                    winrate=("meets_target", "mean"),
                    dd_max=("dd_float_max", "max"),
                    dd_mean=("dd_float_mean", "mean"),
                    n_losers=("ret_pct", lambda s: int((pd.to_numeric(s, errors="coerce") < 0.0).fillna(False).sum())),
                    n_small_winners=(
                        "ret_pct",
                        lambda s: int(
                            (
                                (pd.to_numeric(s, errors="coerce") >= 0.0)
                                & (pd.to_numeric(s, errors="coerce") < float(args.target))
                            )
                            .fillna(False)
                            .sum()
                        ),
                    ),
                    n_target_winners=(
                        "ret_pct",
                        lambda s: int((pd.to_numeric(s, errors="coerce") >= float(args.target)).fillna(False).sum()),
                    ),
                    sum_losers=(
                        "ret_pct",
                        lambda s: float(
                            pd.to_numeric(s, errors="coerce")[(pd.to_numeric(s, errors="coerce") < 0.0)].dropna().sum()
                        ),
                    ),
                    sum_small_winners=(
                        "ret_pct",
                        lambda s: float(
                            pd.to_numeric(s, errors="coerce")[
                                (pd.to_numeric(s, errors="coerce") >= 0.0)
                                & (pd.to_numeric(s, errors="coerce") < float(args.target))
                            ]
                            .dropna()
                            .sum()
                        ),
                    ),
                    sum_target_winners=(
                        "ret_pct",
                        lambda s: float(
                            pd.to_numeric(s, errors="coerce")[(pd.to_numeric(s, errors="coerce") >= float(args.target))]
                            .dropna()
                            .sum()
                        ),
                    ),
                )
                .reset_index()
            )
            summary["by_leg"] = by_leg.to_dict(orient="records")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
