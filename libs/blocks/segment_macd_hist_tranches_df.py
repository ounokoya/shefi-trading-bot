from __future__ import annotations

import math

import numpy as np
import pandas as pd


def segment_macd_hist_tranches_df(
    df: pd.DataFrame,
    *,
    ts_col: str = "ts",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    hist_col: str = "macd_hist",
    extremes_on: str = "high_low",
) -> pd.DataFrame:
    out = df.copy()

    if extremes_on not in ("high_low", "close"):
        raise ValueError("extremes_on must be one of: high_low, close")

    ts = pd.to_numeric(out[ts_col], errors="coerce").astype("Int64").to_numpy()
    hist = pd.to_numeric(out[hist_col], errors="coerce").astype(float).to_numpy()

    high = None
    low = None
    close = None
    if extremes_on == "close":
        close = pd.to_numeric(out[close_col], errors="coerce").astype(float).to_numpy()
    else:
        high = pd.to_numeric(out[high_col], errors="coerce").astype(float).to_numpy()
        low = pd.to_numeric(out[low_col], errors="coerce").astype(float).to_numpy()

    n = int(len(out))

    tranche_id = np.full(n, np.nan, dtype=float)
    tranche_sign = np.full(n, None, dtype=object)
    tranche_pos = np.full(n, np.nan, dtype=float)

    tranche_start_ts = np.full(n, np.nan, dtype=float)
    tranche_end_ts = np.full(n, np.nan, dtype=float)
    tranche_len = np.full(n, np.nan, dtype=float)
    tranche_high = np.full(n, np.nan, dtype=float)
    tranche_low = np.full(n, np.nan, dtype=float)
    tranche_high_ts = np.full(n, np.nan, dtype=float)
    tranche_low_ts = np.full(n, np.nan, dtype=float)

    current_id = -1
    current_sign = 0
    prev_effective_sign = 0

    tranche_starts: list[int] = []
    tranche_ends: list[int] = []
    tranche_signs: list[int] = []

    for i in range(n):
        h = hist[i]
        if math.isnan(h) or pd.isna(ts[i]):
            current_sign = 0
            continue

        if h > 0:
            current_sign = 1
        elif h < 0:
            current_sign = -1
        else:
            current_sign = prev_effective_sign

        if current_sign == 0:
            continue

        if prev_effective_sign == 0 or current_id < 0:
            current_id += 1
            tranche_starts.append(i)
            tranche_signs.append(current_sign)
        elif current_sign != prev_effective_sign:
            tranche_ends.append(i - 1)
            current_id += 1
            tranche_starts.append(i)
            tranche_signs.append(current_sign)

        prev_effective_sign = current_sign
        tranche_id[i] = float(current_id)
        tranche_sign[i] = "+" if current_sign > 0 else "-"

    if tranche_starts:
        if len(tranche_ends) < len(tranche_starts):
            tranche_ends.append(n - 1)

        for tidx, start_i in enumerate(tranche_starts):
            end_i = tranche_ends[tidx]
            sign_i = tranche_signs[tidx]
            length_i = end_i - start_i + 1

            if extremes_on == "close" and close is not None:
                w = close[start_i : end_i + 1]
                max_high = float(np.nanmax(w))
                min_low = float(np.nanmin(w))
                idx_high_local = int(np.nanargmax(w))
                idx_low_local = int(np.nanargmin(w))
            else:
                if high is None or low is None:
                    raise ValueError("extremes_on=high_low requires high_col and low_col")
                w_high = high[start_i : end_i + 1]
                w_low = low[start_i : end_i + 1]
                max_high = float(np.nanmax(w_high))
                min_low = float(np.nanmin(w_low))
                idx_high_local = int(np.nanargmax(w_high))
                idx_low_local = int(np.nanargmin(w_low))

            idx_high = start_i + idx_high_local
            idx_low = start_i + idx_low_local

            start_ts = float(ts[start_i])
            end_ts = float(ts[end_i])

            for j in range(start_i, end_i + 1):
                tranche_id[j] = float(tidx)
                tranche_sign[j] = "+" if sign_i > 0 else "-"
                tranche_pos[j] = float(j - start_i)

                tranche_start_ts[j] = start_ts
                tranche_end_ts[j] = end_ts
                tranche_len[j] = float(length_i)
                tranche_high[j] = max_high
                tranche_low[j] = min_low
                tranche_high_ts[j] = float(ts[idx_high])
                tranche_low_ts[j] = float(ts[idx_low])

    out["tranche_id"] = pd.Series(tranche_id).astype("Int64")
    out["tranche_sign"] = tranche_sign
    out["tranche_pos"] = pd.Series(tranche_pos).astype("Int64")

    out["tranche_start_ts"] = pd.Series(tranche_start_ts).astype("Int64")
    out["tranche_end_ts"] = pd.Series(tranche_end_ts).astype("Int64")
    out["tranche_len"] = pd.Series(tranche_len).astype("Int64")

    out["tranche_high"] = tranche_high
    out["tranche_low"] = tranche_low
    out["tranche_high_ts"] = pd.Series(tranche_high_ts).astype("Int64")
    out["tranche_low_ts"] = pd.Series(tranche_low_ts).astype("Int64")

    return out
