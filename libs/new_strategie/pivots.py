from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from libs.blocks.segment_macd_hist_tranches_df import segment_macd_hist_tranches_df
from libs.new_strategie.config import NewStrategieConfig


@dataclass(frozen=True)
class PivotPoint:
    pivot_id: int
    level: float
    zone_low: float
    zone_high: float

    side: str  # LONG => pivot low, SHORT => pivot high
    strength_kind: str  # strong | medium | weak

    dx_max_last: float

    first_ts: int
    last_ts: int
    n_tests: int


def _safe_float(v: object) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return float(x)


def _dt(ts: int) -> str:
    if int(ts) <= 0:
        return ""
    return str(pd.to_datetime(int(ts), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC"))


def _strength_kind(*, dx_max: float, di_sup: float, di_min: float) -> str:
    if float(dx_max) > float(di_sup):
        return "strong"
    if float(dx_max) < float(di_min):
        return "weak"
    return "medium"


def _pivot_level_for_extreme(
    df: pd.DataFrame,
    *,
    extreme_idx: int,
    side: str,
    high_col: str,
    low_col: str,
    close_col: str,
) -> float | None:
    if extreme_idx < 0 or extreme_idx >= int(len(df)):
        return None

    row = df.iloc[int(extreme_idx)]

    high_v = _safe_float(row.get(str(high_col)))
    low_v = _safe_float(row.get(str(low_col)))
    close_v = _safe_float(row.get(str(close_col)))

    if str(side) == "SHORT":
        # pivot high
        if high_v is None:
            return None
        if close_v is not None and float(close_v) == float(high_v):
            return float(close_v)
        return float(high_v)

    # pivot low
    if low_v is None:
        return None
    if close_v is not None and float(close_v) == float(low_v):
        return float(close_v)
    return float(low_v)


def extract_pivot_candidates(df: pd.DataFrame, *, cfg: NewStrategieConfig) -> list[PivotPoint]:
    """Extract one pivot candidate per MACD_hist tranche (if DI-aligned DX max exists).

    Output list is chronological by tranche.
    """

    for c in (
        cfg.ts_col,
        cfg.high_col,
        cfg.low_col,
        cfg.close_col,
        cfg.macd_hist_col,
        cfg.dx_col,
        cfg.adx_col,
        cfg.plus_di_col,
        cfg.minus_di_col,
    ):
        if str(c) not in df.columns:
            raise ValueError(f"Missing required column for pivots: {c}")

    # Segmentation gives tranche_id / tranche_sign and tranche_high_ts / tranche_low_ts
    work = segment_macd_hist_tranches_df(
        df,
        ts_col=str(cfg.ts_col),
        high_col=str(cfg.high_col),
        low_col=str(cfg.low_col),
        close_col=str(cfg.close_col),
        hist_col=str(cfg.macd_hist_col),
        extremes_on="high_low",
    )

    tids = pd.to_numeric(work["tranche_id"], errors="coerce").dropna().astype(int)
    if tids.empty:
        return []

    out: list[PivotPoint] = []
    next_id = 0

    for tid in sorted(set(tids.tolist())):
        w = work.loc[work["tranche_id"].astype("Int64") == int(tid)]
        if w.empty:
            continue

        sign = str(w["tranche_sign"].iloc[0])
        tranche_side = "LONG" if sign == "+" else "SHORT"

        # DI alignment mask
        plus_di = pd.to_numeric(w[str(cfg.plus_di_col)], errors="coerce").astype(float)
        minus_di = pd.to_numeric(w[str(cfg.minus_di_col)], errors="coerce").astype(float)
        dx = pd.to_numeric(w[str(cfg.dx_col)], errors="coerce").astype(float)

        if sign == "+":
            aligned = (plus_di > minus_di)
        else:
            aligned = (minus_di > plus_di)

        aligned = aligned.fillna(False).astype(bool)
        if not bool(aligned.any()):
            # No pivot for this tranche
            continue

        dx_aligned = dx.where(aligned)
        if dx_aligned.dropna().empty:
            continue

        # index within w (absolute index from df because w keeps original index)
        dx_max_idx = int(dx_aligned.idxmax())
        dx_max_val = _safe_float(work.loc[dx_max_idx, str(cfg.dx_col)])
        if dx_max_val is None:
            continue

        di_sup = max(
            float(_safe_float(work.loc[dx_max_idx, str(cfg.plus_di_col)]) or float("nan")),
            float(_safe_float(work.loc[dx_max_idx, str(cfg.minus_di_col)]) or float("nan")),
        )
        di_min = min(
            float(_safe_float(work.loc[dx_max_idx, str(cfg.plus_di_col)]) or float("nan")),
            float(_safe_float(work.loc[dx_max_idx, str(cfg.minus_di_col)]) or float("nan")),
        )
        if not (math.isfinite(di_sup) and math.isfinite(di_min)):
            continue

        # Pick pivot extreme for the tranche (high for SHORT, low for LONG)
        extreme_ts_col = "tranche_high_ts" if tranche_side == "SHORT" else "tranche_low_ts"
        extreme_ts = int(pd.to_numeric(w[extreme_ts_col], errors="coerce").dropna().iloc[0])
        if extreme_ts <= 0:
            continue

        # map ts->index
        ts_map = pd.to_numeric(work[str(cfg.ts_col)], errors="coerce").astype("Int64")
        idx_matches = ts_map[ts_map == int(extreme_ts)].index.tolist()
        if not idx_matches:
            continue
        extreme_idx = int(idx_matches[-1])

        level = _pivot_level_for_extreme(
            work,
            extreme_idx=int(extreme_idx),
            side=str(tranche_side),
            high_col=str(cfg.high_col),
            low_col=str(cfg.low_col),
            close_col=str(cfg.close_col),
        )
        if level is None:
            continue

        level_f = float(level)
        zone_low = float(level_f * (1.0 - float(cfg.pivot_zone_pct)))
        zone_high = float(level_f * (1.0 + float(cfg.pivot_zone_pct)))

        ts0 = int(pd.to_numeric(w[str(cfg.ts_col)], errors="coerce").dropna().iloc[0])
        ts1 = int(pd.to_numeric(w[str(cfg.ts_col)], errors="coerce").dropna().iloc[-1])

        out.append(
            PivotPoint(
                pivot_id=int(next_id),
                level=float(level_f),
                zone_low=float(zone_low),
                zone_high=float(zone_high),
                side=str(tranche_side),
                strength_kind=str(_strength_kind(dx_max=float(dx_max_val), di_sup=float(di_sup), di_min=float(di_min))),
                dx_max_last=float(dx_max_val),
                first_ts=int(ts0),
                last_ts=int(ts1),
                n_tests=1,
            )
        )
        next_id += 1

    out.sort(key=lambda p: int(p.first_ts))
    return out


def build_top_pivots(df: pd.DataFrame, *, cfg: NewStrategieConfig) -> list[PivotPoint]:
    """Build pivot list from candidates and merge nearby pivots.

    Returns the top cfg.max_pivots by dx_max_last (descending), but keeps chronological info.
    """

    candidates = extract_pivot_candidates(df, cfg=cfg)
    if not candidates:
        return []

    merged: list[PivotPoint] = []

    for p in candidates:
        merged_into = False
        for j, q in enumerate(merged):
            # merge by price proximity (Δ% between levels)
            if float(q.level) == 0.0:
                continue
            dist = abs(float(p.level) - float(q.level)) / abs(float(q.level))
            if dist <= float(cfg.pivot_merge_pct):
                # merge: keep most recent dx_max_last, update timestamps, tests
                dx_last = float(p.dx_max_last)
                kind = str(p.strength_kind)
                merged[j] = PivotPoint(
                    pivot_id=int(q.pivot_id),
                    level=float(q.level),
                    zone_low=float(q.level * (1.0 - float(cfg.pivot_zone_pct))),
                    zone_high=float(q.level * (1.0 + float(cfg.pivot_zone_pct))),
                    side=str(q.side),
                    strength_kind=str(kind),
                    dx_max_last=float(dx_last),
                    first_ts=int(q.first_ts),
                    last_ts=int(max(int(q.last_ts), int(p.last_ts))),
                    n_tests=int(q.n_tests + 1),
                )
                merged_into = True
                break

        if not merged_into:
            merged.append(p)

    # rank by dx_max_last desc
    merged_sorted = sorted(merged, key=lambda x: float(x.dx_max_last), reverse=True)
    top = merged_sorted[: int(cfg.max_pivots)]
    # keep stable ids
    return top


def pivots_to_chrono_rows(pivots: list[PivotPoint]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for p in sorted(pivots, key=lambda x: int(x.first_ts)):
        rows.append(
            {
                "pivot_id": int(p.pivot_id),
                "side": str(p.side),
                "kind": str(p.strength_kind),
                "level": float(p.level),
                "zone_low": float(p.zone_low),
                "zone_high": float(p.zone_high),
                "dx_max_last": float(p.dx_max_last),
                "n_tests": int(p.n_tests),
                "first_ts": int(p.first_ts),
                "first_dt": _dt(int(p.first_ts)),
                "last_ts": int(p.last_ts),
                "last_dt": _dt(int(p.last_ts)),
            }
        )
    return rows


def is_in_any_pivot_zone(*, price: float, pivots: list[PivotPoint]) -> bool:
    p = float(price)
    for z in pivots:
        if float(z.zone_low) <= p <= float(z.zone_high):
            return True
    return False
