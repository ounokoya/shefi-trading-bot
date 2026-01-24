from __future__ import annotations

import numpy as np
import pandas as pd


def get_current_tranche_series_extreme_signal(
    df: pd.DataFrame,
    *,
    series_col: str,
    ts_col: str = "ts",
    hist_col: str = "macd_hist",
) -> dict[str, object]:
    for c in (ts_col, hist_col, series_col):
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if len(df) == 0:
        return {
            "has_tranche": False,
            "tranche_sign": None,
            "tranche_start_ts": None,
            "tranche_len": 0,
            "is_extreme_confirmed_now": False,
            "extreme_kind": None,
            "extreme_index": None,
            "extreme_ts": None,
            "series_col": str(series_col),
            "extreme_value": None,
            "open_side": None,
        }

    ts = pd.to_numeric(df[ts_col], errors="coerce").astype("Int64").to_numpy()
    hist = pd.to_numeric(df[hist_col], errors="coerce").astype(float).to_numpy()
    series = pd.to_numeric(df[series_col], errors="coerce").astype(float).to_numpy()

    n = int(len(df))

    sign = np.zeros(n, dtype=int)
    prev = 0
    for i in range(n):
        h = float(hist[i])
        if (not np.isfinite(h)) or pd.isna(ts[i]):
            sign[i] = 0
            continue
        if h > 0:
            prev = 1
        elif h < 0:
            prev = -1
        else:
            prev = prev
        sign[i] = prev

    current_sign = int(sign[-1])
    if current_sign == 0:
        return {
            "has_tranche": False,
            "tranche_sign": None,
            "tranche_start_ts": None,
            "tranche_len": 0,
            "is_extreme_confirmed_now": False,
            "extreme_kind": None,
            "extreme_index": None,
            "extreme_ts": None,
            "series_col": str(series_col),
            "extreme_value": None,
            "open_side": None,
        }

    start_i = 0
    for i in range(n - 2, -1, -1):
        si = int(sign[i])
        if si == 0:
            continue
        if si != current_sign:
            start_i = i + 1
            break

    tranche_sign = "+" if current_sign > 0 else "-"
    tranche_start_ts = int(ts[start_i]) if not pd.isna(ts[start_i]) else None
    tranche_len = int(n - start_i)

    if int(n - start_i) < 2:
        return {
            "has_tranche": True,
            "tranche_sign": tranche_sign,
            "tranche_start_ts": tranche_start_ts,
            "tranche_len": tranche_len,
            "is_extreme_confirmed_now": False,
            "extreme_kind": "LOW" if current_sign < 0 else "HIGH",
            "extreme_index": None,
            "extreme_ts": None,
            "series_col": str(series_col),
            "extreme_value": None,
            "open_side": None,
        }

    cand_i = int(n - 2)
    now_i = int(n - 1)

    if int(sign[cand_i]) != int(current_sign):
        return {
            "has_tranche": True,
            "tranche_sign": tranche_sign,
            "tranche_start_ts": tranche_start_ts,
            "tranche_len": tranche_len,
            "is_extreme_confirmed_now": False,
            "extreme_kind": "LOW" if current_sign < 0 else "HIGH",
            "extreme_index": None,
            "extreme_ts": None,
            "series_col": str(series_col),
            "extreme_value": None,
            "open_side": None,
        }

    cand_v = float(series[cand_i])
    now_v = float(series[now_i])
    if (not np.isfinite(cand_v)) or (not np.isfinite(now_v)):
        return {
            "has_tranche": True,
            "tranche_sign": tranche_sign,
            "tranche_start_ts": tranche_start_ts,
            "tranche_len": tranche_len,
            "is_extreme_confirmed_now": False,
            "extreme_kind": "LOW" if current_sign < 0 else "HIGH",
            "extreme_index": None,
            "extreme_ts": None,
            "series_col": str(series_col),
            "extreme_value": None,
            "open_side": None,
        }

    is_confirmed = False
    extreme_kind = "LOW" if current_sign < 0 else "HIGH"
    open_side = None

    if current_sign < 0:
        prev_min = float("inf")
        if cand_i > start_i:
            s = series[start_i:cand_i]
            if bool(np.isfinite(s).any()):
                prev_min = float(np.nanmin(s))
        is_new_record = (cand_i == start_i) or (np.isfinite(prev_min) and cand_v < prev_min)
        is_confirmed = bool(is_new_record and (now_v >= cand_v))
        if is_confirmed:
            open_side = "LONG"
    else:
        prev_max = float("-inf")
        if cand_i > start_i:
            s = series[start_i:cand_i]
            if bool(np.isfinite(s).any()):
                prev_max = float(np.nanmax(s))
        is_new_record = (cand_i == start_i) or (np.isfinite(prev_max) and cand_v > prev_max)
        is_confirmed = bool(is_new_record and (now_v <= cand_v))
        if is_confirmed:
            open_side = "SHORT"

    extreme_ts = int(ts[cand_i]) if (is_confirmed and (not pd.isna(ts[cand_i]))) else None

    return {
        "has_tranche": True,
        "tranche_sign": tranche_sign,
        "tranche_start_ts": tranche_start_ts,
        "tranche_len": tranche_len,
        "is_extreme_confirmed_now": bool(is_confirmed),
        "extreme_kind": extreme_kind,
        "extreme_index": int(cand_i) if is_confirmed else None,
        "extreme_ts": extreme_ts,
        "series_col": str(series_col),
        "extreme_value": float(cand_v) if is_confirmed else None,
        "open_side": open_side,
    }
