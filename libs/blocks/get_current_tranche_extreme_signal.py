from __future__ import annotations

import numpy as np
import pandas as pd


def get_current_tranche_extreme_signal(
    df: pd.DataFrame,
    *,
    ts_col: str = "ts",
    close_col: str = "close",
    hist_col: str = "macd_hist",
) -> dict[str, object]:
    for c in (ts_col, close_col, hist_col):
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
            "extreme_close": None,
            "open_side": None,
        }

    ts = pd.to_numeric(df[ts_col], errors="coerce").astype("Int64").to_numpy()
    close = pd.to_numeric(df[close_col], errors="coerce").astype(float).to_numpy()
    hist = pd.to_numeric(df[hist_col], errors="coerce").astype(float).to_numpy()

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
            "extreme_close": None,
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

    w = close[start_i:]
    if w.size == 0 or (not np.isfinite(w).any()):
        return {
            "has_tranche": True,
            "tranche_sign": tranche_sign,
            "tranche_start_ts": tranche_start_ts,
            "tranche_len": tranche_len,
            "is_extreme_confirmed_now": False,
            "extreme_kind": None,
            "extreme_index": None,
            "extreme_ts": None,
            "extreme_close": None,
            "open_side": None,
        }

    # Runtime rule (no lookahead): an extreme at index i is only validated at i+1.
    # Therefore, we can only confirm an extreme when we have at least 2 bars in the current tranche.
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
            "extreme_close": None,
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
            "extreme_close": None,
            "open_side": None,
        }

    cand_close = float(close[cand_i])
    now_close = float(close[now_i])
    if (not np.isfinite(cand_close)) or (not np.isfinite(now_close)):
        return {
            "has_tranche": True,
            "tranche_sign": tranche_sign,
            "tranche_start_ts": tranche_start_ts,
            "tranche_len": tranche_len,
            "is_extreme_confirmed_now": False,
            "extreme_kind": "LOW" if current_sign < 0 else "HIGH",
            "extreme_index": None,
            "extreme_ts": None,
            "extreme_close": None,
            "open_side": None,
        }

    is_confirmed = False
    extreme_kind = "LOW" if current_sign < 0 else "HIGH"
    open_side = None

    if current_sign < 0:
        # Candidate must be a new record low (strictly) at cand_i.
        prev_min = float("inf")
        if cand_i > start_i:
            prev_min = float(np.nanmin(close[start_i:cand_i]))
        is_new_record = (cand_i == start_i) or (np.isfinite(prev_min) and cand_close < prev_min)
        # Confirmation must happen at now_i: now_close is NOT more extreme than candidate.
        is_confirmed = bool(is_new_record and (now_close >= cand_close))
        if is_confirmed:
            open_side = "LONG"
    else:
        # Candidate must be a new record high (strictly) at cand_i.
        prev_max = float("-inf")
        if cand_i > start_i:
            prev_max = float(np.nanmax(close[start_i:cand_i]))
        is_new_record = (cand_i == start_i) or (np.isfinite(prev_max) and cand_close > prev_max)
        # Confirmation must happen at now_i: now_close is NOT more extreme than candidate.
        is_confirmed = bool(is_new_record and (now_close <= cand_close))
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
        "extreme_close": float(cand_close) if is_confirmed else None,
        "open_side": open_side,
    }
