from __future__ import annotations

import pandas as pd

from libs.blocks.get_current_tranche_series_extreme_signal import get_current_tranche_series_extreme_signal


def _get_current_tranche_start_index_and_sign(
    df: pd.DataFrame,
    *,
    ts_col: str,
    hist_col: str,
) -> tuple[int | None, int]:
    if len(df) == 0:
        return None, 0

    ts = pd.to_numeric(df[ts_col], errors="coerce").astype("Int64").to_numpy()
    hist = pd.to_numeric(df[hist_col], errors="coerce").astype(float).to_numpy()

    n = int(len(df))
    sign = [0] * n
    prev = 0
    for i in range(n):
        h = float(hist[i])
        if pd.isna(ts[i]) or pd.isna(h):
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
        return None, 0

    start_i = 0
    for i in range(n - 2, -1, -1):
        si = int(sign[i])
        if si == 0:
            continue
        if si != current_sign:
            start_i = i + 1
            break

    return int(start_i), int(current_sign)


def _first_confirm_pos_in_tranche(
    tranche_series: pd.Series,
    *,
    direction: str,
) -> int | None:
    v = pd.to_numeric(tranche_series, errors="coerce").astype(float).to_numpy()
    m = int(len(v))
    if m < 2:
        return None

    if str(direction).upper() == "MIN":
        record = float("inf")
        for j in range(0, m - 1):
            a = float(v[j])
            b = float(v[j + 1])
            if (not pd.notna(a)) or (not pd.notna(b)):
                continue
            is_new_record = a < record
            if is_new_record:
                record = a
            if is_new_record and b >= a:
                return int(j + 1)
        return None

    if str(direction).upper() == "MAX":
        record = float("-inf")
        for j in range(0, m - 1):
            a = float(v[j])
            b = float(v[j + 1])
            if (not pd.notna(a)) or (not pd.notna(b)):
                continue
            is_new_record = a > record
            if is_new_record:
                record = a
            if is_new_record and b <= a:
                return int(j + 1)
        return None

    raise ValueError(f"Unexpected direction: {direction}")


def _first_threshold_hit_pos_in_tranche(
    tranche_series: pd.Series,
    *,
    side: str,
    threshold: float,
) -> int | None:
    thr = float(threshold)
    if thr <= 0:
        return None
    s = str(side).upper()
    v = pd.to_numeric(tranche_series, errors="coerce").astype(float).to_numpy()
    for j in range(int(len(v))):
        x = v[j]
        if not pd.notna(x):
            continue
        if s == "LONG" and float(x) <= (-thr):
            return int(j)
        if s == "SHORT" and float(x) >= (+thr):
            return int(j)
    return None


def _cci_filter_ok(*, side: str, value: object, threshold: float | None) -> bool:
    if threshold is None:
        return True

    if float(threshold) <= 0:
        return True
    v = pd.to_numeric(value, errors="coerce")
    if pd.isna(v):
        return False
    thr = float(threshold)
    if side == "LONG":
        return float(v) <= (-thr)
    if side == "SHORT":
        return float(v) >= (+thr)
    raise ValueError(f"Unexpected side: {side}")


def _cci_filter_ok_tranche(*, side: str, tranche_series: pd.Series, threshold: float | None) -> bool:
    if threshold is None:
        return True
    thr = float(threshold)
    if thr <= 0:
        return True

    s = str(side).upper()
    v = pd.to_numeric(tranche_series, errors="coerce").astype(float)
    if len(v) == 0:
        return False

    if s == "LONG":
        return bool((v <= (-thr)).any())
    if s == "SHORT":
        return bool((v >= (+thr)).any())
    raise ValueError(f"Unexpected side: {side}")


def _cci_threshold_for_col(
    col: str,
    *,
    cci_fast_col: str = "cci_30",
    cci_medium_col: str = "cci_120",
    cci_slow_col: str = "cci_300",
    cci_fast_threshold: float | None,
    cci_medium_threshold: float | None,
    cci_slow_threshold: float | None,
) -> float | None:
    if col == str(cci_fast_col):
        return cci_fast_threshold
    if col == str(cci_medium_col):
        return cci_medium_threshold
    if col == str(cci_slow_col):
        return cci_slow_threshold
    return None


def _trend_side_from_pair(*, plus: object, minus: object) -> str | None:
    p = pd.to_numeric(plus, errors="coerce")
    m = pd.to_numeric(minus, errors="coerce")
    if pd.isna(p) or pd.isna(m):
        return None
    if float(p) > float(m):
        return "LONG"
    if float(m) > float(p):
        return "SHORT"
    return None


def _trend_filter_allows_open_side(
    *,
    trend_filter: str,
    open_side: str | None,
    vortex_side: str | None,
    dmi_side: str | None,
) -> bool:
    tf = str(trend_filter).strip().lower()
    if tf in {"", "none", "off", "0"}:
        return True
    if open_side is None:
        return False
    if tf == "vortex":
        return vortex_side is not None and str(vortex_side).upper() == str(open_side).upper()
    if tf == "dmi":
        return dmi_side is not None and str(dmi_side).upper() == str(open_side).upper()
    if tf == "both":
        return (
            vortex_side is not None
            and dmi_side is not None
            and str(vortex_side).upper() == str(open_side).upper()
            and str(dmi_side).upper() == str(open_side).upper()
        )
    raise ValueError(f"Unexpected trend_filter: {trend_filter}")


def get_current_tranche_extreme_zone_confluence_signal(
    df: pd.DataFrame,
    *,
    ts_col: str = "ts",
    hist_col: str = "macd_hist",
    close_col: str = "close",
    series_cols: list[str] | None = None,
    target_close_extreme_kind: str | None = "LOW",
    cci_fast_threshold: float | None = None,
    cci_medium_threshold: float | None = None,
    cci_slow_threshold: float | None = None,
    cci_fast_col: str = "cci_30",
    cci_medium_col: str = "cci_120",
    cci_slow_col: str = "cci_300",
    min_confirmed: int | None = None,
    trend_filter: str | None = None,
    vortex_plus_col: str = "vi_plus",
    vortex_minus_col: str = "vi_minus",
    dmi_plus_col: str = "di_plus",
    dmi_minus_col: str = "di_minus",
) -> dict[str, object]:
    series_cols_eff = list(series_cols) if series_cols is not None else [
        close_col,
        "macd_hist",
        "macd_line",
        str(cci_fast_col),
        str(cci_medium_col),
        str(cci_slow_col),
        "vwma_4",
        "vwma_12",
    ]

    for c in (ts_col, hist_col, close_col):
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    for c in series_cols_eff:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if len(df) < 2:
        return {
            "is_zone": False,
            "confirmed_series": [],
            "confirmed_count": 0,
            "required_count": 0,
            "cand_ts": None,
            "now_ts": None,
            "open_side": None,
            "close_extreme_kind": None,
            "tranche_sign": None,
            "tranche_start_ts": None,
            "tranche_len": 0,
            "trend_filter": str(trend_filter).lower() if trend_filter is not None else "none",
            "trend_vortex_side": None,
            "trend_dmi_side": None,
            "trend_ok": True,
        }

    cand_ts = int(pd.to_numeric(df[ts_col].iloc[-2], errors="coerce"))
    now_ts = int(pd.to_numeric(df[ts_col].iloc[-1], errors="coerce"))

    confirmed_series: list[str] = []
    tranche_sign: str | None = None
    tranche_start_ts: int | None = None
    tranche_len: int = 0
    open_side: str | None = None
    close_extreme_kind: str | None = None

    # Always evaluate close first (primary signal)
    close_sig = get_current_tranche_series_extreme_signal(
        df,
        series_col=close_col,
        ts_col=ts_col,
        hist_col=hist_col,
    )

    tranche_sign = close_sig.get("tranche_sign")  # '+' / '-'
    tranche_start_ts = close_sig.get("tranche_start_ts")
    tranche_len = int(close_sig.get("tranche_len") or 0)
    close_extreme_kind = (
        str(close_sig.get("extreme_kind")) if close_sig.get("extreme_kind") is not None else None
    )

    if not bool(close_sig.get("is_extreme_confirmed_now")):
        return {
            "is_zone": False,
            "confirmed_series": [],
            "confirmed_count": 0,
            "required_count": len(series_cols_eff),
            "cand_ts": cand_ts,
            "now_ts": now_ts,
            "open_side": None,
            "close_extreme_kind": close_extreme_kind,
            "tranche_sign": tranche_sign,
            "tranche_start_ts": tranche_start_ts,
            "tranche_len": tranche_len,
            "trend_filter": str(trend_filter).lower() if trend_filter is not None else "none",
            "trend_vortex_side": None,
            "trend_dmi_side": None,
            "trend_ok": True,
        }

    if target_close_extreme_kind is not None:
        target = str(target_close_extreme_kind).upper()
        if close_extreme_kind is None or close_extreme_kind.upper() != target:
            return {
                "is_zone": False,
                "confirmed_series": [],
                "confirmed_count": 0,
                "required_count": len(series_cols_eff),
                "cand_ts": cand_ts,
                "now_ts": now_ts,
                "open_side": None,
                "close_extreme_kind": close_extreme_kind,
                "tranche_sign": tranche_sign,
                "tranche_start_ts": tranche_start_ts,
                "tranche_len": tranche_len,
                "trend_filter": str(trend_filter).lower() if trend_filter is not None else "none",
                "trend_vortex_side": None,
                "trend_dmi_side": None,
                "trend_ok": True,
            }

    open_side = str(close_sig.get("open_side")) if close_sig.get("open_side") is not None else None
    confirmed_series.append(close_col)

    tranche_len = int(close_sig.get("tranche_len") or 0)
    tranche_df = df
    if tranche_len > 0 and tranche_len <= len(df):
        tranche_df = df.iloc[int(len(df) - tranche_len) :]

    for col in series_cols_eff:
        if col == close_col:
            continue

        thr = _cci_threshold_for_col(
            str(col),
            cci_fast_col=cci_fast_col,
            cci_medium_col=cci_medium_col,
            cci_slow_col=cci_slow_col,
            cci_fast_threshold=cci_fast_threshold,
            cci_medium_threshold=cci_medium_threshold,
            cci_slow_threshold=cci_slow_threshold,
        )

        # CCI columns are validated as tranche-level extremes (threshold hit anywhere in tranche).
        if thr is not None:
            if open_side is None:
                continue
            if str(col) not in tranche_df.columns:
                continue
            if not _cci_filter_ok_tranche(side=str(open_side), tranche_series=tranche_df[str(col)], threshold=float(thr)):
                continue
            confirmed_series.append(col)
            continue

        s = get_current_tranche_series_extreme_signal(df, series_col=col, ts_col=ts_col, hist_col=hist_col)
        if bool(s.get("is_extreme_confirmed_now")):
            # All series are validated on the same candle (cand = n-2). If a mismatch happens, ignore.
            if s.get("extreme_ts") != cand_ts:
                continue
            if open_side is not None and s.get("open_side") != open_side:
                continue
            confirmed_series.append(col)

    confirmed_count = int(len(confirmed_series))
    required_count = int(len(series_cols_eff))
    required_min = required_count if min_confirmed is None else int(min_confirmed)

    trend_filter_eff = str(trend_filter).strip().lower() if trend_filter is not None else "none"
    vortex_side: str | None = None
    dmi_side: str | None = None
    if trend_filter_eff in {"vortex", "both"}:
        for c in (vortex_plus_col, vortex_minus_col):
            if str(c) not in df.columns:
                raise ValueError(f"Missing required column for vortex trend filter: {c}")
        now_row = df.iloc[-1]
        vortex_side = _trend_side_from_pair(plus=now_row[str(vortex_plus_col)], minus=now_row[str(vortex_minus_col)])
    if trend_filter_eff in {"dmi", "both"}:
        for c in (dmi_plus_col, dmi_minus_col):
            if str(c) not in df.columns:
                raise ValueError(f"Missing required column for dmi trend filter: {c}")
        now_row = df.iloc[-1]
        dmi_side = _trend_side_from_pair(plus=now_row[str(dmi_plus_col)], minus=now_row[str(dmi_minus_col)])

    trend_ok = _trend_filter_allows_open_side(
        trend_filter=trend_filter_eff,
        open_side=open_side,
        vortex_side=vortex_side,
        dmi_side=dmi_side,
    )

    return {
        "is_zone": bool((confirmed_count >= required_min) and trend_ok),
        "confirmed_series": confirmed_series,
        "confirmed_count": confirmed_count,
        "required_count": required_min,
        "cand_ts": cand_ts,
        "now_ts": now_ts,
        "open_side": open_side,
        "close_extreme_kind": close_extreme_kind,
        "tranche_sign": tranche_sign,
        "tranche_start_ts": tranche_start_ts,
        "tranche_len": tranche_len,
        "trend_filter": trend_filter_eff,
        "trend_vortex_side": vortex_side,
        "trend_dmi_side": dmi_side,
        "trend_ok": trend_ok,
    }


def get_current_tranche_extreme_zone_confluence_tranche_last_signal(
    df: pd.DataFrame,
    *,
    ts_col: str = "ts",
    hist_col: str = "macd_hist",
    close_col: str = "close",
    series_cols: list[str] | None = None,
    target_close_extreme_kind: str | None = "LOW",
    cci_fast_threshold: float | None = None,
    cci_medium_threshold: float | None = None,
    cci_slow_threshold: float | None = None,
    cci_fast_col: str = "cci_30",
    cci_medium_col: str = "cci_120",
    cci_slow_col: str = "cci_300",
    min_confirmed: int | None = None,
    trend_filter: str | None = None,
    vortex_plus_col: str = "vi_plus",
    vortex_minus_col: str = "vi_minus",
    dmi_plus_col: str = "di_plus",
    dmi_minus_col: str = "di_minus",
) -> dict[str, object]:
    series_cols_eff = list(series_cols) if series_cols is not None else [
        close_col,
        "macd_hist",
        "macd_line",
        str(cci_fast_col),
        str(cci_medium_col),
        str(cci_slow_col),
        "vwma_4",
        "vwma_12",
    ]

    for c in (ts_col, hist_col, close_col):
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    for c in series_cols_eff:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if len(df) < 2:
        return {
            "is_zone": False,
            "confirmed_series": [],
            "newly_confirmed_series": [],
            "confirmed_count": 0,
            "required_count": 0,
            "cand_ts": None,
            "now_ts": None,
            "open_side": None,
            "close_extreme_kind": None,
            "tranche_sign": None,
            "tranche_start_ts": None,
            "tranche_len": 0,
            "trend_filter": str(trend_filter).lower() if trend_filter is not None else "none",
            "trend_vortex_side": None,
            "trend_dmi_side": None,
            "trend_ok": True,
        }

    cand_ts = int(pd.to_numeric(df[ts_col].iloc[-2], errors="coerce"))
    now_ts = int(pd.to_numeric(df[ts_col].iloc[-1], errors="coerce"))

    start_i, current_sign = _get_current_tranche_start_index_and_sign(df, ts_col=ts_col, hist_col=hist_col)
    if start_i is None or current_sign == 0:
        return {
            "is_zone": False,
            "confirmed_series": [],
            "newly_confirmed_series": [],
            "confirmed_count": 0,
            "required_count": len(series_cols_eff),
            "cand_ts": cand_ts,
            "now_ts": now_ts,
            "open_side": None,
            "close_extreme_kind": None,
            "tranche_sign": None,
            "tranche_start_ts": None,
            "tranche_len": 0,
            "trend_filter": str(trend_filter).lower() if trend_filter is not None else "none",
            "trend_vortex_side": None,
            "trend_dmi_side": None,
            "trend_ok": True,
        }

    tranche = df.iloc[int(start_i) :].reset_index(drop=True)
    tranche_len = int(len(tranche))
    tranche_start_ts = int(pd.to_numeric(tranche[ts_col].iloc[0], errors="coerce"))
    tranche_sign = "+" if int(current_sign) > 0 else "-"

    close_extreme_kind = "HIGH" if int(current_sign) > 0 else "LOW"
    if target_close_extreme_kind is not None:
        if str(close_extreme_kind).upper() != str(target_close_extreme_kind).upper():
            return {
                "is_zone": False,
                "confirmed_series": [],
                "newly_confirmed_series": [],
                "confirmed_count": 0,
                "required_count": len(series_cols_eff),
                "cand_ts": cand_ts,
                "now_ts": now_ts,
                "open_side": None,
                "close_extreme_kind": close_extreme_kind,
                "tranche_sign": tranche_sign,
                "tranche_start_ts": tranche_start_ts,
                "tranche_len": tranche_len,
                "trend_filter": str(trend_filter).lower() if trend_filter is not None else "none",
                "trend_vortex_side": None,
                "trend_dmi_side": None,
                "trend_ok": True,
            }

    direction = "MAX" if int(current_sign) > 0 else "MIN"
    open_side = "SHORT" if int(current_sign) > 0 else "LONG"

    first_confirm_pos_by_col: dict[str, int] = {}
    for col in series_cols_eff:
        thr = _cci_threshold_for_col(
            str(col),
            cci_fast_col=cci_fast_col,
            cci_medium_col=cci_medium_col,
            cci_slow_col=cci_slow_col,
            cci_fast_threshold=cci_fast_threshold,
            cci_medium_threshold=cci_medium_threshold,
            cci_slow_threshold=cci_slow_threshold,
        )

        # CCI columns are validated as tranche-level extremes (threshold hit anywhere in tranche).
        if thr is not None:
            if open_side is None:
                continue
            pos = _first_threshold_hit_pos_in_tranche(tranche[str(col)], side=str(open_side), threshold=float(thr))
            if pos is None:
                continue
            first_confirm_pos_by_col[str(col)] = int(pos)
            continue

        pos = _first_confirm_pos_in_tranche(tranche[col], direction=direction)
        if pos is not None:
            first_confirm_pos_by_col[str(col)] = int(pos)

    confirmed_series = sorted(list(first_confirm_pos_by_col.keys()), key=lambda c: series_cols_eff.index(c))
    confirmed_count = int(len(confirmed_series))
    required_count = int(len(series_cols_eff))
    required_min = required_count if min_confirmed is None else int(min_confirmed)

    if confirmed_count < required_min:
        return {
            "is_zone": False,
            "confirmed_series": confirmed_series,
            "newly_confirmed_series": [],
            "confirmed_count": confirmed_count,
            "required_count": required_min,
            "cand_ts": cand_ts,
            "now_ts": now_ts,
            "open_side": open_side,
            "close_extreme_kind": close_extreme_kind,
            "tranche_sign": tranche_sign,
            "tranche_start_ts": tranche_start_ts,
            "tranche_len": tranche_len,
            "trend_filter": str(trend_filter).lower() if trend_filter is not None else "none",
            "trend_vortex_side": None,
            "trend_dmi_side": None,
            "trend_ok": True,
        }

    positions_sorted = sorted([int(p) for p in first_confirm_pos_by_col.values()])
    trigger_pos = int(positions_sorted[int(required_min) - 1])
    now_pos = int(tranche_len - 1)
    newly_confirmed_series = [c for c, p in first_confirm_pos_by_col.items() if int(p) == now_pos]

    trend_filter_eff = str(trend_filter).strip().lower() if trend_filter is not None else "none"
    vortex_side: str | None = None
    dmi_side: str | None = None
    if trend_filter_eff in {"vortex", "both"}:
        for c in (vortex_plus_col, vortex_minus_col):
            if str(c) not in df.columns:
                raise ValueError(f"Missing required column for vortex trend filter: {c}")
        now_row = df.iloc[-1]
        vortex_side = _trend_side_from_pair(plus=now_row[str(vortex_plus_col)], minus=now_row[str(vortex_minus_col)])
    if trend_filter_eff in {"dmi", "both"}:
        for c in (dmi_plus_col, dmi_minus_col):
            if str(c) not in df.columns:
                raise ValueError(f"Missing required column for dmi trend filter: {c}")
        now_row = df.iloc[-1]
        dmi_side = _trend_side_from_pair(plus=now_row[str(dmi_plus_col)], minus=now_row[str(dmi_minus_col)])

    trend_ok = _trend_filter_allows_open_side(
        trend_filter=trend_filter_eff,
        open_side=open_side,
        vortex_side=vortex_side,
        dmi_side=dmi_side,
    )

    is_zone = bool((trigger_pos == now_pos) and trend_ok)

    return {
        "is_zone": is_zone,
        "confirmed_series": confirmed_series,
        "newly_confirmed_series": newly_confirmed_series,
        "confirmed_count": confirmed_count,
        "required_count": required_min,
        "cand_ts": cand_ts,
        "now_ts": now_ts,
        "open_side": open_side,
        "close_extreme_kind": close_extreme_kind,
        "tranche_sign": tranche_sign,
        "tranche_start_ts": tranche_start_ts,
        "tranche_len": tranche_len,
        "trend_filter": trend_filter_eff,
        "trend_vortex_side": vortex_side,
        "trend_dmi_side": dmi_side,
        "trend_ok": trend_ok,
    }
