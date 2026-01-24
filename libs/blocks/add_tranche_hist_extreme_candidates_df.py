from __future__ import annotations

import numpy as np
import pandas as pd


def add_tranche_hist_extreme_candidates_df(
    df: pd.DataFrame,
    *,
    ts_col: str = "ts",
    hist_col: str = "macd_hist",
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    vwma_col: str = "vwma_4",
    vwma2_col: str = "vwma_12",
    macd_line_col: str = "macd_line",
    candidate_filter: str = "none",
    stop_mode: str = "none",
    stop_pct: float = 0.01,
    atr_col: str = "atr_14",
    atr_mult: float = 2.0,
    tranche_id_col: str = "tranche_id",
    tranche_pos_col: str = "tranche_pos",
    tranche_len_col: str = "tranche_len",
    tranche_sign_col: str = "tranche_sign",
) -> pd.DataFrame:
    out = df.copy()

    required = [
        ts_col,
        hist_col,
        price_col,
        tranche_id_col,
        tranche_pos_col,
        tranche_len_col,
        tranche_sign_col,
    ]
    candidate_filter_str = str(candidate_filter)
    if candidate_filter_str not in ("none", "vwma4_align", "vwma4_12_align", "vwma4_12_macd_align"):
        raise ValueError(
            "candidate_filter must be one of: none, vwma4_align, vwma4_12_align, vwma4_12_macd_align"
        )
    if candidate_filter_str == "vwma4_align":
        required.append(vwma_col)
    elif candidate_filter_str == "vwma4_12_align":
        required.extend([vwma_col, vwma2_col])
    elif candidate_filter_str == "vwma4_12_macd_align":
        required.extend([vwma_col, vwma2_col, macd_line_col])
    for c in required:
        if c not in out.columns:
            raise ValueError(f"Missing required column: {c}")

    stop_mode_str = str(stop_mode)
    if stop_mode_str not in ("none", "pct", "atr"):
        raise ValueError("stop_mode must be one of: none, pct, atr")
    if stop_mode_str != "none":
        for c in (high_col, low_col):
            if c not in out.columns:
                raise ValueError(f"Missing required column: {c}")
        if stop_mode_str == "atr" and atr_col not in out.columns:
            raise ValueError(f"Missing required column: {atr_col}")

    ts = pd.to_numeric(out[ts_col], errors="coerce").astype("Int64").to_numpy()
    if pd.isna(ts).any():
        raise ValueError(f"{ts_col} contains NaN after numeric coercion")

    hist = pd.to_numeric(out[hist_col], errors="coerce").astype(float).to_numpy()
    price = pd.to_numeric(out[price_col], errors="coerce").astype(float).to_numpy()

    high = None
    low = None
    atr = None
    if stop_mode_str != "none":
        high = pd.to_numeric(out[high_col], errors="coerce").astype(float).to_numpy()
        low = pd.to_numeric(out[low_col], errors="coerce").astype(float).to_numpy()
        if stop_mode_str == "atr":
            atr = pd.to_numeric(out[atr_col], errors="coerce").astype(float).to_numpy()

    vwma = None
    vwma2 = None
    macd_line = None
    if candidate_filter_str == "vwma4_align":
        vwma = pd.to_numeric(out[vwma_col], errors="coerce").astype(float).to_numpy()
    elif candidate_filter_str == "vwma4_12_align":
        vwma = pd.to_numeric(out[vwma_col], errors="coerce").astype(float).to_numpy()
        vwma2 = pd.to_numeric(out[vwma2_col], errors="coerce").astype(float).to_numpy()
    elif candidate_filter_str == "vwma4_12_macd_align":
        vwma = pd.to_numeric(out[vwma_col], errors="coerce").astype(float).to_numpy()
        vwma2 = pd.to_numeric(out[vwma2_col], errors="coerce").astype(float).to_numpy()
        macd_line = pd.to_numeric(out[macd_line_col], errors="coerce").astype(float).to_numpy()

    tranche_id = pd.to_numeric(out[tranche_id_col], errors="coerce").astype("Int64").to_numpy()
    tranche_pos = pd.to_numeric(out[tranche_pos_col], errors="coerce").astype("Int64").to_numpy()
    tranche_len = pd.to_numeric(out[tranche_len_col], errors="coerce").astype("Int64").to_numpy()
    tranche_sign = out[tranche_sign_col].to_numpy(dtype=object)

    n = int(len(out))

    cand_min_ts = np.full(n, np.nan, dtype=float)
    cand_min_price = np.full(n, np.nan, dtype=float)
    cand_max_ts = np.full(n, np.nan, dtype=float)
    cand_max_price = np.full(n, np.nan, dtype=float)

    cand_extreme_ts = np.full(n, np.nan, dtype=float)
    cand_extreme_price = np.full(n, np.nan, dtype=float)

    cand_first_ts = np.full(n, np.nan, dtype=float)
    cand_first_price = np.full(n, np.nan, dtype=float)
    cand_last_ts = np.full(n, np.nan, dtype=float)
    cand_last_price = np.full(n, np.nan, dtype=float)

    cand_rank = np.full(n, np.nan, dtype=float)
    cand_is_event = np.zeros(n, dtype=int)

    cand_running_ts = np.full(n, np.nan, dtype=float)
    cand_running_price = np.full(n, np.nan, dtype=float)

    first_rows = np.where(pd.Series(tranche_pos).fillna(-1).to_numpy() == 0)[0]

    for start_i in first_rows:
        if pd.isna(tranche_id[start_i]) or pd.isna(tranche_len[start_i]):
            continue

        length = int(tranche_len[start_i])
        if length <= 0:
            continue

        end_i = start_i + length - 1
        if end_i >= n:
            end_i = n - 1

        w_hist = hist[start_i : end_i + 1]
        w_ts = ts[start_i : end_i + 1]
        w_price = price[start_i : end_i + 1]
        w_high = None
        w_low = None
        w_atr = None
        if stop_mode_str != "none" and high is not None and low is not None:
            w_high = high[start_i : end_i + 1]
            w_low = low[start_i : end_i + 1]
            if stop_mode_str == "atr" and atr is not None:
                w_atr = atr[start_i : end_i + 1]
        w_vwma = None
        w_vwma2 = None
        w_macd = None
        if vwma is not None:
            w_vwma = vwma[start_i : end_i + 1]
        if vwma2 is not None:
            w_vwma2 = vwma2[start_i : end_i + 1]
        if macd_line is not None:
            w_macd = macd_line[start_i : end_i + 1]

        if len(w_hist) < 3:
            continue

        d = np.diff(w_hist)
        s = np.sign(d)

        for k in range(1, len(s)):
            if s[k] == 0:
                s[k] = s[k - 1]

        if len(s) > 0 and s[0] == 0:
            nz = np.where(s != 0)[0]
            if len(nz) > 0:
                s[: int(nz[0]) + 1] = s[int(nz[0])]

        sign = tranche_sign[start_i]

        if sign not in ("-", "+"):
            continue

        min_idx_all: list[int] = []
        max_idx_all: list[int] = []
        for k in range(1, len(s)):
            if (s[k - 1] < 0) and (s[k] > 0):
                min_idx_all.append(int(k))
            elif (s[k - 1] > 0) and (s[k] < 0):
                max_idx_all.append(int(k))

        min_idx_local = min_idx_all[0] if min_idx_all else None
        max_idx_local = max_idx_all[0] if max_idx_all else None

        if min_idx_local is not None:
            ts_i = float(w_ts[int(min_idx_local)])
            p_i = float(w_price[int(min_idx_local)])
            cand_min_ts[start_i : end_i + 1] = ts_i
            cand_min_price[start_i : end_i + 1] = p_i

        if max_idx_local is not None:
            ts_i = float(w_ts[int(max_idx_local)])
            p_i = float(w_price[int(max_idx_local)])
            cand_max_ts[start_i : end_i + 1] = ts_i
            cand_max_price[start_i : end_i + 1] = p_i

        cand_idx_all = min_idx_all if sign == "-" else max_idx_all

        if candidate_filter_str in ("vwma4_align", "vwma4_12_align", "vwma4_12_macd_align"):
            needs_vwma2 = candidate_filter_str in ("vwma4_12_align", "vwma4_12_macd_align")
            needs_macd = candidate_filter_str == "vwma4_12_macd_align"
            if w_vwma is None or (needs_vwma2 and w_vwma2 is None) or (needs_macd and w_macd is None):
                cand_idx_all = []
            else:
                d_v = np.diff(w_vwma)
                s_v = np.sign(d_v)

                d_v2 = None
                s_v2 = None
                if needs_vwma2 and w_vwma2 is not None:
                    d_v2 = np.diff(w_vwma2)
                    s_v2 = np.sign(d_v2)

                d_m = None
                s_m = None
                if needs_macd and w_macd is not None:
                    d_m = np.diff(w_macd)
                    s_m = np.sign(d_m)
                    s_m = np.where(np.isfinite(s_m), s_m, 0.0)

                for k in range(1, len(s_v)):
                    if s_v[k] == 0:
                        s_v[k] = s_v[k - 1]

                if s_v2 is not None:
                    for k in range(1, len(s_v2)):
                        if s_v2[k] == 0:
                            s_v2[k] = s_v2[k - 1]

                if s_m is not None:
                    for k in range(1, len(s_m)):
                        if s_m[k] == 0:
                            s_m[k] = s_m[k - 1]

                if len(s_v) > 0 and s_v[0] == 0:
                    nzv = np.where(s_v != 0)[0]
                    if len(nzv) > 0:
                        s_v[: int(nzv[0]) + 1] = s_v[int(nzv[0])]

                if s_v2 is not None and len(s_v2) > 0 and s_v2[0] == 0:
                    nzv2 = np.where(s_v2 != 0)[0]
                    if len(nzv2) > 0:
                        s_v2[: int(nzv2[0]) + 1] = s_v2[int(nzv2[0])]

                if s_m is not None and len(s_m) > 0 and s_m[0] == 0:
                    nzm = np.where(s_m != 0)[0]
                    if len(nzm) > 0:
                        s_m[: int(nzm[0]) + 1] = s_m[int(nzm[0])]

                filtered: list[int] = []
                active = False

                for k in range(1, len(s)):
                    if sign == "-":
                        if not active and (s[k - 1] < 0) and (s[k] > 0):
                            active = True

                        if not active:
                            continue

                        if not (s[k] > 0):
                            active = False
                            continue

                        ok_v = k < len(s_v) and (s_v[k] > 0)
                        ok_v2 = True
                        if s_v2 is not None:
                            ok_v2 = k < len(s_v2) and (s_v2[k] > 0)
                        ok_m = True
                        if s_m is not None:
                            ok_m = k < len(s_m) and (s_m[k] > 0)
                        if ok_v and ok_v2 and ok_m:
                            filtered.append(int(k))
                            active = False
                    else:
                        if not active and (s[k - 1] > 0) and (s[k] < 0):
                            active = True

                        if not active:
                            continue

                        if not (s[k] < 0):
                            active = False
                            continue

                        ok_v = k < len(s_v) and (s_v[k] < 0)
                        ok_v2 = True
                        if s_v2 is not None:
                            ok_v2 = k < len(s_v2) and (s_v2[k] < 0)
                        ok_m = True
                        if s_m is not None:
                            ok_m = k < len(s_m) and (s_m[k] < 0)
                        if ok_v and ok_v2 and ok_m:
                            filtered.append(int(k))
                            active = False

                cand_idx_all = filtered

        selected_idx = None
        if cand_idx_all:
            if stop_mode_str == "none":
                selected_idx = int(cand_idx_all[0])
            else:
                if w_high is None or w_low is None:
                    selected_idx = int(cand_idx_all[0])
                else:
                    ptr = 0
                    while ptr < len(cand_idx_all):
                        entry_idx = int(cand_idx_all[ptr])
                        entry = float(w_price[entry_idx])
                        if not np.isfinite(entry) or entry == 0.0:
                            stop_i = entry_idx
                        else:
                            thr = None
                            if stop_mode_str == "pct":
                                thr = float(stop_pct)
                            else:
                                if w_atr is None:
                                    thr = None
                                else:
                                    a = float(w_atr[entry_idx])
                                    thr = float(atr_mult) * a / entry if np.isfinite(a) else None

                            if thr is None or (not np.isfinite(thr)):
                                stop_i = None
                            else:
                                dd = None
                                if sign == "-":
                                    dd = (entry - w_low[entry_idx:]) / entry
                                else:
                                    dd = (w_high[entry_idx:] - entry) / entry
                                dd = np.where(np.isnan(dd), np.nan, np.maximum(dd, 0.0))

                                opposite_started = False
                                stop_i = None
                                for i in range(entry_idx + 1, len(w_hist)):
                                    slope_in = float(s[i - 1]) if 0 <= i - 1 < len(s) else 0.0
                                    if not opposite_started:
                                        if sign == "-" and slope_in < 0:
                                            opposite_started = True
                                        elif sign == "+" and slope_in > 0:
                                            opposite_started = True

                                    if opposite_started:
                                        ddi = float(dd[i - entry_idx]) if 0 <= i - entry_idx < len(dd) else float("nan")
                                        if np.isfinite(ddi) and ddi >= float(thr):
                                            stop_i = int(i)
                                            break

                        if stop_i is None:
                            selected_idx = entry_idx
                            break

                        nxt = None
                        for j in range(ptr + 1, len(cand_idx_all)):
                            if int(cand_idx_all[j]) > int(stop_i):
                                nxt = j
                                break
                        if nxt is None:
                            selected_idx = None
                            break
                        ptr = int(nxt)

        if selected_idx is not None:
            first_idx = int(cand_idx_all[0])
            last_idx = int(cand_idx_all[-1])
            first_ts = float(w_ts[first_idx])
            first_p = float(w_price[first_idx])
            last_ts = float(w_ts[last_idx])
            last_p = float(w_price[last_idx])

            sel_ts = float(w_ts[int(selected_idx)])
            sel_p = float(w_price[int(selected_idx)])

            if sign == "-":
                cand_min_ts[start_i : end_i + 1] = first_ts
                cand_min_price[start_i : end_i + 1] = first_p
            else:
                cand_max_ts[start_i : end_i + 1] = first_ts
                cand_max_price[start_i : end_i + 1] = first_p

            cand_first_ts[start_i : end_i + 1] = first_ts
            cand_first_price[start_i : end_i + 1] = first_p
            cand_last_ts[start_i : end_i + 1] = last_ts
            cand_last_price[start_i : end_i + 1] = last_p

            cand_extreme_ts[start_i : end_i + 1] = sel_ts
            cand_extreme_price[start_i : end_i + 1] = sel_p

            current_ts = float("nan")
            current_p = float("nan")
            ptr = 0
            for j in range(0, end_i - start_i + 1):
                if ptr < len(cand_idx_all) and j == int(cand_idx_all[ptr]):
                    current_ts = float(w_ts[int(cand_idx_all[ptr])])
                    current_p = float(w_price[int(cand_idx_all[ptr])])
                    cand_is_event[start_i + j] = 1
                    cand_rank[start_i + j] = float(ptr + 1)
                    ptr += 1
                cand_running_ts[start_i + j] = current_ts
                cand_running_price[start_i + j] = current_p

    out["tranche_cand_min_ts"] = pd.Series(cand_min_ts).astype("Int64")
    out["tranche_cand_min_price"] = cand_min_price
    out["tranche_cand_max_ts"] = pd.Series(cand_max_ts).astype("Int64")
    out["tranche_cand_max_price"] = cand_max_price

    out["tranche_cand_is_event"] = cand_is_event
    out["tranche_cand_rank"] = pd.Series(cand_rank).astype("Int64")
    out["tranche_cand_running_ts"] = pd.Series(cand_running_ts).astype("Int64")
    out["tranche_cand_running_price"] = cand_running_price

    out["tranche_cand_first_ts"] = pd.Series(cand_first_ts).astype("Int64")
    out["tranche_cand_first_price"] = cand_first_price
    out["tranche_cand_last_ts"] = pd.Series(cand_last_ts).astype("Int64")
    out["tranche_cand_last_price"] = cand_last_price
    out["tranche_cand_extreme_ts"] = pd.Series(cand_extreme_ts).astype("Int64")
    out["tranche_cand_extreme_price"] = cand_extreme_price

    return out
