from __future__ import annotations

import math
import pandas as pd

from libs.blocks.get_current_tranche_extreme_signal import get_current_tranche_extreme_signal


def _ms_to_iso_utc(ms: int) -> str:
    return pd.to_datetime(int(ms), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")


def _finite_float(x: object) -> float | None:
    v = pd.to_numeric(x, errors="coerce")
    if pd.isna(v):
        return None
    f = float(v)
    if not math.isfinite(f):
        return None
    return f


def _cci_category_for_row(
    row: pd.Series,
    *,
    kind: str,
    cci_fast_col: str,
    cci_medium_col: str,
    cci_slow_col: str,
    cci_fast_threshold: float,
    cci_medium_threshold: float,
    cci_slow_threshold: float,
) -> tuple[str, float | None]:
    k = str(kind).upper()
    if k not in {"LOW", "HIGH"}:
        return "none", None

    def ok(v: float, thr: float) -> bool:
        if thr <= 0:
            return False
        if k == "LOW":
            return v <= (-thr)
        return v >= (+thr)

    slow_v = _finite_float(row.get(str(cci_slow_col)))
    if slow_v is not None and ok(slow_v, float(cci_slow_threshold)):
        return "slow", slow_v

    med_v = _finite_float(row.get(str(cci_medium_col)))
    if med_v is not None and ok(med_v, float(cci_medium_threshold)):
        return "medium", med_v

    fast_v = _finite_float(row.get(str(cci_fast_col)))
    if fast_v is not None and ok(fast_v, float(cci_fast_threshold)):
        return "fast", fast_v

    return "none", None


def _cci_category_for_tranche(
    tranche: pd.DataFrame,
    *,
    kind: str,
    cci_fast_col: str,
    cci_medium_col: str,
    cci_slow_col: str,
    cci_fast_threshold: float,
    cci_medium_threshold: float,
    cci_slow_threshold: float,
) -> dict[str, float]:
    k = str(kind).upper()
    if k not in {"LOW", "HIGH"}:
        return {}

    def _best_value(s: pd.Series, thr: float) -> float | None:
        if thr <= 0:
            return None
        v = pd.to_numeric(s, errors="coerce").astype(float)
        if k == "LOW":
            m = v <= (-thr)
            if not bool(m.any()):
                return None
            return _finite_float(v[m].min())
        m = v >= (+thr)
        if not bool(m.any()):
            return None
        return _finite_float(v[m].max())

    hits: dict[str, float] = {}

    if str(cci_slow_col) in tranche.columns:
        vv = _best_value(tranche[str(cci_slow_col)], float(cci_slow_threshold))
        if vv is not None:
            hits["slow"] = float(vv)

    if str(cci_medium_col) in tranche.columns:
        vv = _best_value(tranche[str(cci_medium_col)], float(cci_medium_threshold))
        if vv is not None:
            hits["medium"] = float(vv)

    if str(cci_fast_col) in tranche.columns:
        vv = _best_value(tranche[str(cci_fast_col)], float(cci_fast_threshold))
        if vv is not None:
            hits["fast"] = float(vv)

    return hits


def extract_window_close_extremes(
    df: pd.DataFrame,
    *,
    ts_col: str = "ts",
    close_col: str = "close",
    hist_col: str = "macd_hist",
    cci_fast_col: str = "cci_30",
    cci_medium_col: str = "cci_120",
    cci_slow_col: str = "cci_300",
    cci_fast_threshold: float = 100.0,
    cci_medium_threshold: float = 100.0,
    cci_slow_threshold: float = 100.0,
    zone_radius_pct: float = 0.01,
    max_bars_ago: int | None = None,
) -> pd.DataFrame:
    for c in (ts_col, close_col, hist_col):
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    for c in (cci_fast_col, cci_medium_col, cci_slow_col):
        if str(c) not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if len(df) < 3:
        return pd.DataFrame([])

    ts = pd.to_numeric(df[ts_col], errors="coerce").astype("Int64")
    close = pd.to_numeric(df[close_col], errors="coerce")

    work = df.copy()
    work[ts_col] = ts
    work[close_col] = close
    work = work.dropna(subset=[ts_col, close_col]).reset_index(drop=True)

    if len(work) < 3:
        return pd.DataFrame([])

    now_ts = int(work[ts_col].iloc[-1])
    now_close = float(work[close_col].iloc[-1])

    events: list[dict[str, object]] = []
    confirmed_by_tranche: dict[tuple[int, str], dict[str, object]] = {}

    for end_idx in range(3, len(work) + 1):
        w = work.iloc[:end_idx]
        sig = get_current_tranche_extreme_signal(w, ts_col=ts_col, close_col=close_col, hist_col=hist_col)

        tranche_start_ts = sig.get("tranche_start_ts")
        if tranche_start_ts is None:
            continue
        tranche_start_ts_i = int(tranche_start_ts)

        kind = str(sig.get("extreme_kind") or "")
        if kind not in {"LOW", "HIGH"}:
            continue

        tranche_len = int(sig.get("tranche_len") or 0)
        if tranche_len <= 0:
            continue
        start_i = int(end_idx - tranche_len)
        if start_i < 0:
            start_i = 0
        tranche_df = w.iloc[start_i:end_idx]

        hits = _cci_category_for_tranche(
            tranche_df,
            kind=kind,
            cci_fast_col=cci_fast_col,
            cci_medium_col=cci_medium_col,
            cci_slow_col=cci_slow_col,
            cci_fast_threshold=cci_fast_threshold,
            cci_medium_threshold=cci_medium_threshold,
            cci_slow_threshold=cci_slow_threshold,
        )

        if bool(sig.get("is_extreme_confirmed_now")):
            cand_i = int(end_idx - 2)
            cand_row = work.iloc[int(cand_i)]

            ev_ts = sig.get("extreme_ts")
            if ev_ts is None:
                continue
            ev_ts_i = int(ev_ts)
            if int(cand_row[ts_col]) != ev_ts_i:
                continue

            ev_close = _finite_float(cand_row[close_col])
            if ev_close is None:
                continue

            confirmed_by_tranche[(int(tranche_start_ts_i), str(kind))] = {
                "ts": int(ev_ts_i),
                "close": float(ev_close),
                "index": int(cand_i),
            }

        best = confirmed_by_tranche.get((int(tranche_start_ts_i), str(kind)))
        if best is None:
            continue

        if not hits:
            continue

        ev_ts_i = int(best["ts"])
        ev_close = float(best["close"])
        ev_index = int(best["index"])

        bars_ago = int(len(work) - 1 - ev_index)
        if max_bars_ago is not None and bars_ago > int(max_bars_ago):
            continue
        pct_from_now = float("nan")
        if math.isfinite(now_close) and now_close != 0:
            pct_from_now = float((ev_close / now_close) - 1.0)

        for cat in ("slow", "medium", "fast"):
            if cat not in hits:
                continue
            events.append(
                {
                    "ts": ev_ts_i,
                    "dt": _ms_to_iso_utc(ev_ts_i),
                    "index": int(ev_index),
                    "kind": kind,
                    "close": float(ev_close),
                    "bars_ago": int(bars_ago),
                    "pct_from_now": float(pct_from_now),
                    "now_ts": int(now_ts),
                    "now_dt": _ms_to_iso_utc(int(now_ts)),
                    "now_close": float(now_close),
                    "tranche_sign": sig.get("tranche_sign"),
                    "tranche_start_ts": sig.get("tranche_start_ts"),
                    "tranche_start_dt": (
                        _ms_to_iso_utc(int(sig.get("tranche_start_ts")))
                        if sig.get("tranche_start_ts") is not None
                        else ""
                    ),
                    "tranche_len": int(tranche_len),
                    "cci_category": str(cat),
                    "cci_value": float(hits[cat]),
                    "zone_radius_pct": float(zone_radius_pct),
                }
            )

    if not events:
        return pd.DataFrame([])

    out = pd.DataFrame(events).drop_duplicates(subset=["tranche_start_ts", "kind", "cci_category"], keep="last")
    rows = out.to_dict("records")
    for r in rows:
        c = float(r["close"])
        if not math.isfinite(c) or c == 0:
            lo = float("nan")
            hi = float("nan")
        else:
            lo = c * (1.0 - float(zone_radius_pct))
            hi = c * (1.0 + float(zone_radius_pct))

        same_type = 0
        other_type = 0
        same_type_same_cat = 0
        any_type_same_cat = 0

        for q in rows:
            if int(q.get("ts")) == int(r.get("ts")) and str(q.get("kind")) == str(r.get("kind")):
                continue
            qc = float(q["close"])
            if not math.isfinite(qc):
                continue
            if math.isfinite(lo) and math.isfinite(hi) and not (lo <= qc <= hi):
                continue

            if str(q["kind"]) == str(r["kind"]):
                same_type += 1
                if str(q["cci_category"]) == str(r["cci_category"]):
                    same_type_same_cat += 1
            else:
                other_type += 1

            if str(q["cci_category"]) == str(r["cci_category"]):
                any_type_same_cat += 1

        r["zone_same_type_count"] = int(same_type)
        r["zone_other_type_count"] = int(other_type)
        r["zone_same_type_same_category_count"] = int(same_type_same_cat)
        r["zone_any_type_same_category_count"] = int(any_type_same_cat)

    out2 = pd.DataFrame(rows)
    out2 = out2.sort_values(["ts", "kind"]).reset_index(drop=True)
    return out2
