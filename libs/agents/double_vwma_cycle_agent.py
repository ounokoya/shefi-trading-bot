from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from libs.agents.impact_bar_agent import compute_impact_bar
from libs.agents.micro_direction_agent import compute_micro_direction


def _safe_float(v: object) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return float(x)


def _pct_delta(a0: float | None, a1: float | None) -> float | None:
    if a0 is None or a1 is None:
        return None
    if (not math.isfinite(float(a0))) or float(a0) == 0.0:
        return None
    if not math.isfinite(float(a1)):
        return None
    return float((float(a1) - float(a0)) / float(a0))


@dataclass(frozen=True)
class DoubleVwmaEvent:
    kind: str
    pos: int
    ts: int
    dt: str
    meta: dict[str, object]


@dataclass(frozen=True)
class DoubleVwmaCycleMetrics:
    cycle_id: int

    start_i: int
    end_i: int
    start_ts: int
    end_ts: int
    start_dt: str
    end_dt: str

    vwma_fast_col: str
    vwma_slow_col: str
    zone_fast_radius_pct: float
    zone_slow_radius_pct: float

    trend_side: str

    spread_abs_max_pct: float | None
    spread_abs_end_pct: float | None
    spread_abs_slope_mean_pct: float | None

    vwma_fast_slope_mean_pct: float | None
    vwma_slow_slope_mean_pct: float | None
    vwma_slope_harmony_ratio: float | None

    pullback_weak_count: int
    pullback_medium_count: int
    pullback_strong_count: int

    last_pullback_kind: str | None
    last_pullback_end_pos: int | None
    last_pullback_recency: int | None

    break_confirm_bars: int
    break_slow_confirmed: bool

    score: float
    is_interesting: bool

    events: list[dict[str, object]]


@dataclass(frozen=True)
class DoubleVwmaCycleAgentConfig:
    ts_col: str = "ts"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"

    vwma_fast_col: str = "vwma_4"
    vwma_slow_col: str = "vwma_12"

    zone_fast_radius_pct: float = 0.001
    zone_slow_radius_pct: float = 0.001

    zone_large_mult: float = 2.0

    break_confirm_bars: int = 3

    reversal_confirm_bars: int = 3
    reversal_impact_filter_enabled: bool = False
    reversal_impact_agg_len: int = 0
    reversal_impact_body_pct_min: float = 60.0
    reversal_impact_body_pct_max: float = 100.0
    reversal_impact_require_same_color: bool = True

    micro_filter_enabled: bool = False
    micro_slope_bars: int = 2
    micro_vwma_col: str = "vwma_4"
    micro_min_abs_slope: float = 0.0

    spread_ref_pct: float = 0.002

    min_cycle_len: int = 20
    min_score: float = 0.05


class DoubleVwmaCycleAgent:
    def __init__(self, *, cfg: DoubleVwmaCycleAgentConfig | None = None):
        self.cfg = cfg or DoubleVwmaCycleAgentConfig()

    def _dt(self, ts: int) -> str:
        if int(ts) <= 0:
            return ""
        return str(pd.to_datetime(int(ts), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC"))

    def _segment_cycles(self, df: pd.DataFrame) -> list[tuple[int, int]]:
        cfg = self.cfg
        for c in (cfg.ts_col, cfg.vwma_fast_col, cfg.vwma_slow_col):
            if str(c) not in df.columns:
                raise ValueError(f"Missing required column: {c}")

        fast = pd.to_numeric(df[str(cfg.vwma_fast_col)], errors="coerce").astype(float).to_numpy()
        slow = pd.to_numeric(df[str(cfg.vwma_slow_col)], errors="coerce").astype(float).to_numpy()

        zf = float(cfg.zone_fast_radius_pct)
        zs = float(cfg.zone_slow_radius_pct)
        if zf < 0:
            zf = 0.0
        if zs < 0:
            zs = 0.0

        fast_upper = fast * (1.0 + zf)
        fast_lower = fast * (1.0 - zf)
        slow_upper = slow * (1.0 + zs)
        slow_lower = slow * (1.0 - zs)

        finite_mask = np.isfinite(fast_upper) & np.isfinite(fast_lower) & np.isfinite(slow_upper) & np.isfinite(slow_lower)
        sep_state = np.zeros(int(len(df)), dtype=int)
        above = finite_mask & (fast_lower > slow_upper)
        below = finite_mask & (fast_upper < slow_lower)
        sep_state = np.where(above, 1, sep_state)
        sep_state = np.where(below, -1, sep_state)

        if int(len(sep_state)) == 0:
            return []

        in_collision = False
        last_sep_sign = 0
        cross_pos: list[int] = []
        for i in range(int(len(sep_state))):
            s = int(sep_state[i])
            if s == 0:
                if (not in_collision) and last_sep_sign != 0:
                    in_collision = True
                continue

            if last_sep_sign == 0:
                last_sep_sign = int(s)
                in_collision = False
                continue

            if in_collision:
                if int(s) != int(last_sep_sign):
                    cross_pos.append(int(i))
                in_collision = False

            last_sep_sign = int(s)

        if not cross_pos:
            return [(0, int(len(df) - 1))] if len(df) else []

        starts = [0] + cross_pos
        ends = [p - 1 for p in cross_pos] + [int(len(df) - 1)]
        out: list[tuple[int, int]] = []
        for a, b in zip(starts, ends):
            if int(b) >= int(a):
                out.append((int(a), int(b)))
        return out

    def _analyze_cycle(self, df: pd.DataFrame, *, cycle_id: int, start: int, end: int) -> DoubleVwmaCycleMetrics | None:
        cfg = self.cfg
        w = df.iloc[int(start) : int(end) + 1].copy()
        if len(w) == 0:
            return None

        for c in (
            cfg.ts_col,
            cfg.open_col,
            cfg.high_col,
            cfg.low_col,
            cfg.close_col,
            cfg.vwma_fast_col,
            cfg.vwma_slow_col,
        ):
            if str(c) not in w.columns:
                raise ValueError(f"Missing required column: {c}")

        ts = pd.to_numeric(w[str(cfg.ts_col)], errors="coerce").astype("Int64").to_numpy()
        o = pd.to_numeric(w[str(cfg.open_col)], errors="coerce").astype(float).to_numpy()
        h = pd.to_numeric(w[str(cfg.high_col)], errors="coerce").astype(float).to_numpy()
        l = pd.to_numeric(w[str(cfg.low_col)], errors="coerce").astype(float).to_numpy()
        c = pd.to_numeric(w[str(cfg.close_col)], errors="coerce").astype(float).to_numpy()
        fast = pd.to_numeric(w[str(cfg.vwma_fast_col)], errors="coerce").astype(float).to_numpy()
        slow = pd.to_numeric(w[str(cfg.vwma_slow_col)], errors="coerce").astype(float).to_numpy()

        start_ts = int(ts[0]) if len(ts) else 0
        end_ts = int(ts[-1]) if len(ts) else 0

        zf = float(cfg.zone_fast_radius_pct)
        zs = float(cfg.zone_slow_radius_pct)
        if zf < 0:
            zf = 0.0
        if zs < 0:
            zs = 0.0

        zone_large_mult = float(cfg.zone_large_mult)
        if zone_large_mult < 1.0:
            zone_large_mult = 1.0

        fast_upper = fast * (1.0 + zf)
        fast_lower = fast * (1.0 - zf)
        slow_upper = slow * (1.0 + zs)
        slow_lower = slow * (1.0 - zs)

        fast_upper_l = fast * (1.0 + zf * zone_large_mult)
        fast_lower_l = fast * (1.0 - zf * zone_large_mult)
        slow_upper_l = slow * (1.0 + zs * zone_large_mult)
        slow_lower_l = slow * (1.0 - zs * zone_large_mult)

        def _sep_sign_from_bounds(*, fu: float, fl: float, su: float, sl: float) -> int:
            if float(fl) > float(su):
                return 1
            if float(fu) < float(sl):
                return -1
            return 0

        trend_side = "SHORT"
        fu0 = _safe_float(fast_upper[0])
        fl0 = _safe_float(fast_lower[0])
        su0 = _safe_float(slow_upper[0])
        sl0 = _safe_float(slow_lower[0])
        if fu0 is not None and fl0 is not None and su0 is not None and sl0 is not None:
            ss0 = _sep_sign_from_bounds(fu=float(fu0), fl=float(fl0), su=float(su0), sl=float(sl0))
            if int(ss0) == 1:
                trend_side = "LONG"
            elif int(ss0) == -1:
                trend_side = "SHORT"
            else:
                diff0 = _safe_float(fast[0])
                diff1 = _safe_float(slow[0])
                trend_side = "LONG" if (diff0 is not None and diff1 is not None and float(diff0) > float(diff1)) else "SHORT"
        else:
            diff0 = _safe_float(fast[0])
            diff1 = _safe_float(slow[0])
            trend_side = "LONG" if (diff0 is not None and diff1 is not None and float(diff0) > float(diff1)) else "SHORT"

        dir_sign = 1.0 if trend_side == "LONG" else -1.0

        spread_abs = None
        if len(w):
            denom = np.where(np.isfinite(slow) & (slow != 0.0), slow, np.nan)
            sp = np.abs((fast - slow) / denom)
            sp = sp[np.isfinite(sp)]
            if sp.size:
                spread_abs = sp

        spread_abs_max_pct = float(np.nanmax(spread_abs)) if spread_abs is not None else None
        spread_abs_end_pct = None
        if len(w):
            d0 = _safe_float(slow[-1])
            if d0 is not None and d0 != 0.0:
                spread_abs_end_pct = float(abs(float(fast[-1]) - float(slow[-1])) / float(d0))

        spread_abs_slope_mean_pct = None
        if len(w) >= 3:
            denom = np.where(np.isfinite(slow[:-1]) & (slow[:-1] != 0.0), slow[:-1], np.nan)
            sp0 = np.abs((fast[:-1] - slow[:-1]) / denom)
            denom1 = np.where(np.isfinite(slow[1:]) & (slow[1:] != 0.0), slow[1:], np.nan)
            sp1 = np.abs((fast[1:] - slow[1:]) / denom1)
            d = sp1 - sp0
            d = d[np.isfinite(d)]
            if d.size:
                spread_abs_slope_mean_pct = float(np.mean(d))

        vwma_fast_slope_mean_pct = None
        vwma_slow_slope_mean_pct = None
        vwma_slope_harmony_ratio = None
        if len(w) >= 3:
            f0 = fast[:-1]
            f1 = fast[1:]
            s0 = slow[:-1]
            s1 = slow[1:]
            fden = np.where(np.isfinite(f0) & (f0 != 0.0), f0, np.nan)
            sden = np.where(np.isfinite(s0) & (s0 != 0.0), s0, np.nan)
            fs = (f1 - f0) / fden
            ss = (s1 - s0) / sden
            fs_ok = fs[np.isfinite(fs)]
            ss_ok = ss[np.isfinite(ss)]
            if fs_ok.size:
                vwma_fast_slope_mean_pct = float(np.mean(fs_ok))
            if ss_ok.size:
                vwma_slow_slope_mean_pct = float(np.mean(ss_ok))
            if fs.size and ss.size:
                ok = (dir_sign * fs > 0.0) & (dir_sign * ss > 0.0)
                ok = ok[np.isfinite(fs) & np.isfinite(ss)]
                if ok.size:
                    vwma_slope_harmony_ratio = float(np.mean(ok.astype(float)))

        events: list[DoubleVwmaEvent] = []

        def _ev(kind: str, pos: int, meta: dict[str, object] | None = None) -> None:
            meta2 = meta or {}
            events.append(
                DoubleVwmaEvent(
                    kind=str(kind),
                    pos=int(pos),
                    ts=int(ts[int(pos)]) if 0 <= int(pos) < len(ts) else 0,
                    dt=self._dt(int(ts[int(pos)]) if 0 <= int(pos) < len(ts) else 0),
                    meta=dict(meta2),
                )
            )

        def _sep_sign_from_values(fv: float, sv: float) -> int | None:
            if (not math.isfinite(float(fv))) or (not math.isfinite(float(sv))):
                return None
            fu = float(fv) * (1.0 + float(zf))
            fl = float(fv) * (1.0 - float(zf))
            su = float(sv) * (1.0 + float(zs))
            sl = float(sv) * (1.0 - float(zs))
            return int(_sep_sign_from_bounds(fu=float(fu), fl=float(fl), su=float(su), sl=float(sl)))

        _ev(
            "cycle_start",
            0,
            {
                "trend_side": str(trend_side),
                "start_pos": 0,
                "end_pos": int(len(w) - 1),
            },
        )

        if int(start) > 0:
            prev = df.iloc[int(start) - 1]
            try:
                pf = float(pd.to_numeric(prev[str(cfg.vwma_fast_col)], errors="coerce"))
                ps = float(pd.to_numeric(prev[str(cfg.vwma_slow_col)], errors="coerce"))
                cf = float(pd.to_numeric(w[str(cfg.vwma_fast_col)].iloc[0], errors="coerce"))
                cs = float(pd.to_numeric(w[str(cfg.vwma_slow_col)].iloc[0], errors="coerce"))
                prev_sep = _sep_sign_from_values(float(pf), float(ps))
                cur_sep = _sep_sign_from_values(float(cf), float(cs))
                if cur_sep is not None and int(cur_sep) != 0 and (prev_sep is None or int(prev_sep) == 0 or int(prev_sep) != int(cur_sep)):
                    _ev(
                        "vwma_cross",
                        0,
                        {
                            "trend_side": trend_side,
                            "mode": "zone_collision",
                            "prev_sep": prev_sep,
                            "sep": cur_sep,
                        },
                    )
            except Exception:
                pass

        finite_mask = np.isfinite(fast_upper) & np.isfinite(fast_lower) & np.isfinite(slow_upper) & np.isfinite(slow_lower)
        sep_state = np.zeros(int(len(w)), dtype=int)
        above = finite_mask & (fast_lower > slow_upper)
        below = finite_mask & (fast_upper < slow_lower)
        sep_state = np.where(above, 1, sep_state)
        sep_state = np.where(below, -1, sep_state)
        is_collision = finite_mask & (sep_state == 0)

        i = 0
        while i < int(len(w)):
            if not bool(is_collision[i]):
                i += 1
                continue

            j = int(i)
            while j < int(len(w)) and bool(is_collision[j]):
                j += 1

            start_pos = int(i)
            end_pos = int(j - 1)
            start_sep = int(sep_state[int(start_pos - 1)]) if int(start_pos) > 0 else 0
            end_sep = int(sep_state[int(end_pos + 1)]) if int(end_pos + 1) < int(len(w)) else 0

            outcome = "open"
            if start_sep != 0 and end_sep != 0:
                outcome = "reject" if int(end_sep) == int(start_sep) else "confirmed_cross"

            direction = None
            if int(start_sep) == 1:
                direction = "DOWN"
            elif int(start_sep) == -1:
                direction = "UP"

            extreme_pos = int(start_pos)
            extreme_spread_abs: float | None = None
            if int(end_pos) >= int(start_pos):
                seg = np.abs(pd.to_numeric(w[str(cfg.vwma_fast_col)].iloc[int(start_pos) : int(end_pos) + 1], errors="coerce").astype(float).to_numpy() - pd.to_numeric(w[str(cfg.vwma_slow_col)].iloc[int(start_pos) : int(end_pos) + 1], errors="coerce").astype(float).to_numpy())
                if seg.size and np.isfinite(seg).any():
                    rel = int(np.nanargmin(seg))
                    extreme_pos = int(start_pos + rel)
                    extreme_spread_abs = float(seg[rel])

            _ev(
                "vwma_zone_collision",
                int(end_pos),
                {
                    "start_pos": int(start_pos),
                    "end_pos": int(end_pos),
                    "extreme_pos": int(extreme_pos),
                    "start_ts": int(ts[int(start_pos)]) if 0 <= int(start_pos) < len(ts) else 0,
                    "end_ts": int(ts[int(end_pos)]) if 0 <= int(end_pos) < len(ts) else 0,
                    "extreme_ts": int(ts[int(extreme_pos)]) if 0 <= int(extreme_pos) < len(ts) else 0,
                    "start_dt": self._dt(int(ts[int(start_pos)]) if 0 <= int(start_pos) < len(ts) else 0),
                    "end_dt": self._dt(int(ts[int(end_pos)]) if 0 <= int(end_pos) < len(ts) else 0),
                    "extreme_dt": self._dt(int(ts[int(extreme_pos)]) if 0 <= int(extreme_pos) < len(ts) else 0),
                    "start_sep": int(start_sep),
                    "end_sep": int(end_sep),
                    "direction": direction,
                    "outcome": str(outcome),
                    "extreme_spread_abs": extreme_spread_abs,
                },
            )

            i = int(j)

        def _band_state(pos: int, *, lower: np.ndarray, upper: np.ndarray) -> str | None:
            hp = _safe_float(h[pos])
            lp = _safe_float(l[pos])
            lo = _safe_float(lower[pos])
            up = _safe_float(upper[pos])
            if hp is None or lp is None or lo is None or up is None:
                return None
            if float(lp) > float(up):
                return "above"
            if float(hp) < float(lo):
                return "below"
            return "in_zone"

        def _band_intersects(pos: int, *, lower: np.ndarray, upper: np.ndarray) -> bool:
            st = _band_state(int(pos), lower=lower, upper=upper)
            return bool(st == "in_zone")

        def _emit_zone_tests(
            *,
            zone: str,
            layer: str,
            lower: np.ndarray,
            upper: np.ndarray,
            inner_lower: np.ndarray | None = None,
            inner_upper: np.ndarray | None = None,
            require_outer_visit: bool = False,
        ) -> None:
            in_test = False
            start_pos: int | None = None
            start_side: str | None = None
            direction: str | None = None
            extreme_pos: int | None = None
            extreme_price: float | None = None
            outer_visit = False
            for k in range(int(len(w))):
                st = _band_state(int(k), lower=lower, upper=upper)
                if st is None:
                    if in_test:
                        in_test = False
                        start_pos = None
                        start_side = None
                        direction = None
                        extreme_pos = None
                        extreme_price = None
                        outer_visit = False
                    continue

                if not in_test:
                    if st != "in_zone":
                        continue
                    if int(k) <= 0:
                        continue
                    prev_st = _band_state(int(k - 1), lower=lower, upper=upper)
                    if prev_st not in {"above", "below"}:
                        continue
                    in_test = True
                    start_pos = int(k)
                    start_side = str(prev_st)
                    direction = "DOWN" if str(prev_st) == "above" else "UP"
                    if direction == "DOWN":
                        lp = _safe_float(l[int(k)])
                        extreme_price = float(lp) if lp is not None else None
                        extreme_pos = int(k)
                    else:
                        hp = _safe_float(h[int(k)])
                        extreme_price = float(hp) if hp is not None else None
                        extreme_pos = int(k)
                    outer_visit = False
                    if inner_lower is not None and inner_upper is not None:
                        outer_visit = bool(_band_intersects(int(k), lower=lower, upper=upper) and (not _band_intersects(int(k), lower=inner_lower, upper=inner_upper)))
                    continue

                if st == "in_zone":
                    if direction == "DOWN":
                        lp = _safe_float(l[int(k)])
                        if lp is not None and (extreme_price is None or float(lp) < float(extreme_price)):
                            extreme_price = float(lp)
                            extreme_pos = int(k)
                    elif direction == "UP":
                        hp = _safe_float(h[int(k)])
                        if hp is not None and (extreme_price is None or float(hp) > float(extreme_price)):
                            extreme_price = float(hp)
                            extreme_pos = int(k)
                    if inner_lower is not None and inner_upper is not None:
                        if bool(_band_intersects(int(k), lower=lower, upper=upper) and (not _band_intersects(int(k), lower=inner_lower, upper=inner_upper))):
                            outer_visit = True
                    continue

                end_pos = int(k - 1)
                end_side = str(st)
                traversed = bool(start_side in {"above", "below"} and end_side in {"above", "below"} and end_side != start_side)
                outcome = "open"
                if start_side in {"above", "below"} and end_side in {"above", "below"}:
                    outcome = "traversed" if bool(traversed) else "rejection"
                if require_outer_visit and (not bool(outer_visit)):
                    in_test = False
                    start_pos = None
                    start_side = None
                    direction = None
                    extreme_pos = None
                    extreme_price = None
                    outer_visit = False
                    continue

                is_contra = bool((trend_side == "LONG" and direction == "DOWN") or (trend_side == "SHORT" and direction == "UP"))
                _ev(
                    "zone_test",
                    int(end_pos),
                    {
                        "zone": str(zone),
                        "layer": str(layer),
                        "direction": direction,
                        "trend_side": str(trend_side),
                        "is_contra_trend": bool(is_contra),
                        "start_pos": int(start_pos) if start_pos is not None else None,
                        "end_pos": int(end_pos),
                        "extreme_pos": int(extreme_pos) if extreme_pos is not None else None,
                        "start_ts": int(ts[int(start_pos)]) if start_pos is not None and 0 <= int(start_pos) < len(ts) else 0,
                        "end_ts": int(ts[int(end_pos)]) if 0 <= int(end_pos) < len(ts) else 0,
                        "extreme_ts": int(ts[int(extreme_pos)]) if extreme_pos is not None and 0 <= int(extreme_pos) < len(ts) else 0,
                        "start_dt": self._dt(int(ts[int(start_pos)]) if start_pos is not None and 0 <= int(start_pos) < len(ts) else 0),
                        "end_dt": self._dt(int(ts[int(end_pos)]) if 0 <= int(end_pos) < len(ts) else 0),
                        "extreme_dt": self._dt(int(ts[int(extreme_pos)]) if extreme_pos is not None and 0 <= int(extreme_pos) < len(ts) else 0),
                        "start_side": start_side,
                        "end_side": end_side,
                        "traversed": bool(traversed),
                        "extreme_price": extreme_price,
                        "outcome": str(outcome),
                    },
                )

                in_test = False
                start_pos = None
                start_side = None
                direction = None
                extreme_pos = None
                extreme_price = None
                outer_visit = False

            if in_test and start_pos is not None:
                end_pos = int(len(w) - 1)
                if (not require_outer_visit) or bool(outer_visit):
                    is_contra = bool((trend_side == "LONG" and direction == "DOWN") or (trend_side == "SHORT" and direction == "UP"))
                    _ev(
                        "zone_test",
                        int(end_pos),
                        {
                            "zone": str(zone),
                            "layer": str(layer),
                            "direction": direction,
                            "trend_side": str(trend_side),
                            "is_contra_trend": bool(is_contra),
                            "start_pos": int(start_pos),
                            "end_pos": int(end_pos),
                            "extreme_pos": int(extreme_pos) if extreme_pos is not None else None,
                            "start_ts": int(ts[int(start_pos)]) if 0 <= int(start_pos) < len(ts) else 0,
                            "end_ts": int(ts[int(end_pos)]) if 0 <= int(end_pos) < len(ts) else 0,
                            "extreme_ts": int(ts[int(extreme_pos)]) if extreme_pos is not None and 0 <= int(extreme_pos) < len(ts) else 0,
                            "start_dt": self._dt(int(ts[int(start_pos)]) if 0 <= int(start_pos) < len(ts) else 0),
                            "end_dt": self._dt(int(ts[int(end_pos)]) if 0 <= int(end_pos) < len(ts) else 0),
                            "extreme_dt": self._dt(int(ts[int(extreme_pos)]) if extreme_pos is not None and 0 <= int(extreme_pos) < len(ts) else 0),
                            "start_side": start_side,
                            "end_side": None,
                            "traversed": False,
                            "extreme_price": extreme_price,
                            "outcome": "open",
                        },
                    )

        _emit_zone_tests(zone="fast", layer="security", lower=fast_lower, upper=fast_upper)
        _emit_zone_tests(zone="slow", layer="security", lower=slow_lower, upper=slow_upper)
        _emit_zone_tests(
            zone="fast",
            layer="large",
            lower=fast_lower_l,
            upper=fast_upper_l,
            inner_lower=fast_lower,
            inner_upper=fast_upper,
            require_outer_visit=True,
        )
        _emit_zone_tests(
            zone="slow",
            layer="large",
            lower=slow_lower_l,
            upper=slow_upper_l,
            inner_lower=slow_lower,
            inner_upper=slow_upper,
            require_outer_visit=True,
        )

        break_confirm_bars = int(cfg.break_confirm_bars)
        confirm_slow_break = bool(int(break_confirm_bars) >= 1)
        if break_confirm_bars < 0:
            break_confirm_bars = 0

        pullback_weak_count = 0
        pullback_medium_count = 0
        pullback_strong_count = 0
        last_pullback_kind = None
        last_pullback_end_pos = None
        last_pullback_recency = None

        in_pullback = False
        pb_start = None
        pb_deep = 0
        pb_break_confirmed = False
        slow_confirm_run = 0

        pb_extreme_pos: int | None = None
        pb_extreme_price: float | None = None

        reversal_confirm_bars = int(cfg.reversal_confirm_bars)
        if reversal_confirm_bars < 1:
            reversal_confirm_bars = 1
        reversal_run = 0
        reversal_confirmed = False
        reversal_confirm_pos: int | None = None

        def _on_trend_side(pos: int) -> bool:
            if trend_side == "LONG":
                return bool(_safe_float(l[pos]) is not None and _safe_float(fast_upper[pos]) is not None and float(l[pos]) >= float(fast_upper[pos]))
            return bool(_safe_float(h[pos]) is not None and _safe_float(fast_lower[pos]) is not None and float(h[pos]) <= float(fast_lower[pos]))

        def _touch_zone(pos: int, *, upper: float, lower: float) -> bool:
            hp = _safe_float(h[pos])
            lp = _safe_float(l[pos])
            if hp is None or lp is None:
                return False
            return bool(float(hp) >= float(lower) and float(lp) <= float(upper))

        for i in range(int(len(w))):
            fu = _safe_float(fast_upper[i])
            fl = _safe_float(fast_lower[i])
            su = _safe_float(slow_upper[i])
            sl = _safe_float(slow_lower[i])

            if fu is None or fl is None or su is None or sl is None:
                continue

            on_trend = _on_trend_side(int(i))

            touched_fast = _touch_zone(int(i), upper=float(fu), lower=float(fl))
            touched_slow = _touch_zone(int(i), upper=float(su), lower=float(sl))

            broke_fast = False
            broke_slow = False
            hi = _safe_float(h[i])
            li = _safe_float(l[i])
            if hi is not None and li is not None:
                if trend_side == "LONG":
                    broke_fast = bool(float(li) < float(fl))
                    broke_slow = bool(float(li) < float(sl))
                else:
                    broke_fast = bool(float(hi) > float(fu))
                    broke_slow = bool(float(hi) > float(su))

            if (not in_pullback) and (not on_trend) and (touched_fast or broke_fast or broke_slow):
                in_pullback = True
                pb_start = int(i)
                pb_deep = 0
                pb_break_confirmed = False
                slow_confirm_run = 0
                pb_extreme_pos = int(i)
                if trend_side == "LONG":
                    pb_extreme_price = (None if li is None else float(li))
                else:
                    pb_extreme_price = (None if hi is None else float(hi))
                reversal_run = 0
                reversal_confirmed = False
                reversal_confirm_pos = None
                _ev("pullback_start", int(i))

            if in_pullback:
                if trend_side == "LONG":
                    if li is not None and (pb_extreme_price is None or float(li) < float(pb_extreme_price)):
                        pb_extreme_price = float(li)
                        pb_extreme_pos = int(i)
                        reversal_run = 0
                else:
                    if hi is not None and (pb_extreme_price is None or float(hi) > float(pb_extreme_price)):
                        pb_extreme_price = float(hi)
                        pb_extreme_pos = int(i)
                        reversal_run = 0

                if touched_fast:
                    pb_deep = max(int(pb_deep), 1)
                if broke_fast:
                    pb_deep = max(int(pb_deep), 2)
                if broke_slow:
                    pb_deep = max(int(pb_deep), 3)

                if (not reversal_confirmed) and pb_extreme_pos is not None and int(i) > int(pb_extreme_pos):
                    oi = _safe_float(o[i])
                    ci = _safe_float(c[i])
                    if oi is None or ci is None:
                        reversal_run = 0
                    else:
                        candle_ok = bool((float(ci) > float(oi))) if trend_side == "LONG" else bool((float(ci) < float(oi)))
                        if candle_ok:
                            reversal_run += 1
                        else:
                            reversal_run = 0

                    if reversal_run >= int(reversal_confirm_bars):
                        agg_len = int(cfg.reversal_impact_agg_len)
                        if agg_len <= 0:
                            agg_len = int(reversal_confirm_bars)
                        run_start_pos = int(i - int(reversal_confirm_bars) + 1)

                        meta: dict[str, object] = {
                            "start_pos": int(pb_start) if pb_start is not None else None,
                            "extreme_pos": int(pb_extreme_pos),
                            "extreme_price": (None if pb_extreme_price is None else float(pb_extreme_price)),
                            "confirm_bars": int(reversal_confirm_bars),
                            "run_start_pos": int(run_start_pos),
                            "trend_side": str(trend_side),
                        }

                        if bool(cfg.reversal_impact_filter_enabled):
                            target_color = "GREEN" if str(trend_side) == "LONG" else "RED"
                            z = compute_impact_bar(
                                w,
                                end_pos=int(i),
                                agg_len=int(agg_len),
                                ts_col=str(cfg.ts_col),
                                open_col=str(cfg.open_col),
                                high_col=str(cfg.high_col),
                                low_col=str(cfg.low_col),
                                close_col=str(cfg.close_col),
                                target_color=str(target_color),
                                body_pct_min=float(cfg.reversal_impact_body_pct_min),
                                body_pct_max=float(cfg.reversal_impact_body_pct_max),
                                require_same_color=bool(cfg.reversal_impact_require_same_color),
                            )
                            if not bool(z.get("is_impact")):
                                continue
                            meta["impact"] = dict(z)

                        if bool(cfg.micro_filter_enabled):
                            if str(cfg.micro_vwma_col) not in set(w.columns):
                                raise ValueError(f"Missing required column for micro filter: {cfg.micro_vwma_col}")
                            z2 = compute_micro_direction(
                                w,
                                pos=int(i),
                                ts_col=str(cfg.ts_col),
                                slope_bars=int(cfg.micro_slope_bars),
                                vwma_col=str(cfg.micro_vwma_col),
                                min_abs_slope=float(cfg.micro_min_abs_slope),
                            )
                            if str(z2.get("side")) != str(trend_side):
                                continue
                            meta["micro"] = dict(z2)

                        reversal_confirmed = True
                        reversal_confirm_pos = int(i)
                        _ev("pullback_reversal_confirmed", int(i), dict(meta))

                if confirm_slow_break and broke_slow and (not pb_break_confirmed):
                    oi = _safe_float(o[i])
                    ci = _safe_float(c[i])
                    if oi is None or ci is None:
                        slow_confirm_run = 0
                    else:
                        candle_ok = bool((float(ci) < float(oi))) if trend_side == "LONG" else bool((float(ci) > float(oi)))
                        if candle_ok:
                            slow_confirm_run += 1
                        else:
                            slow_confirm_run = 0
                    if slow_confirm_run >= int(break_confirm_bars):
                        pb_break_confirmed = True
                        _ev(
                            "slow_break_confirmed",
                            int(i),
                            {
                                "confirm_bars": int(break_confirm_bars),
                                "break_mode": "wick",
                                "trend_side": str(trend_side),
                                "broke_fast": bool(broke_fast),
                                "broke_slow": bool(broke_slow),
                                "bar_high": float(hi) if hi is not None else None,
                                "bar_low": float(li) if li is not None else None,
                                "fast_upper": float(fu) if fu is not None else None,
                                "fast_lower": float(fl) if fl is not None else None,
                                "slow_upper": float(su) if su is not None else None,
                                "slow_lower": float(sl) if sl is not None else None,
                            },
                        )

                if on_trend:
                    pb_end = int(i)
                    kind = None
                    if pb_deep <= 1:
                        kind = "pullback_weak"
                        pullback_weak_count += 1
                    elif pb_deep == 2:
                        kind = "pullback_medium"
                        pullback_medium_count += 1
                    else:
                        kind = "pullback_strong"
                        pullback_strong_count += 1

                    _ev(str(kind), int(pb_end), {"start_pos": int(pb_start) if pb_start is not None else None, "break_confirmed": bool(pb_break_confirmed)})

                    if pb_extreme_pos is not None:
                        _ev(
                            "pullback_summary",
                            int(pb_end),
                            {
                                "start_pos": int(pb_start) if pb_start is not None else None,
                                "end_pos": int(pb_end),
                                "trend_side": str(trend_side),
                                "extreme_pos": int(pb_extreme_pos),
                                "extreme_price": (None if pb_extreme_price is None else float(pb_extreme_price)),
                                "reversal_confirmed": bool(reversal_confirmed),
                                "reversal_pos": (None if reversal_confirm_pos is None else int(reversal_confirm_pos)),
                                "confirm_bars": int(reversal_confirm_bars),
                                "break_confirmed": bool(pb_break_confirmed),
                            },
                        )

                    last_pullback_kind = str(kind)
                    last_pullback_end_pos = int(pb_end)
                    last_pullback_recency = int(len(w) - 1 - int(pb_end))

                    in_pullback = False
                    pb_start = None
                    pb_deep = 0
                    pb_break_confirmed = False
                    slow_confirm_run = 0
                    pb_extreme_pos = None
                    pb_extreme_price = None
                    reversal_run = 0
                    reversal_confirmed = False
                    reversal_confirm_pos = None

        break_slow_confirmed = bool(any(e.kind == "slow_break_confirmed" for e in events))

        _ev(
            "cycle_end",
            int(len(w) - 1),
            {
                "trend_side": str(trend_side),
                "start_pos": 0,
                "end_pos": int(len(w) - 1),
                "break_slow_confirmed": bool(break_slow_confirmed),
            },
        )

        spread_ref = float(cfg.spread_ref_pct)
        if spread_ref <= 0:
            spread_ref = 0.002
        spread_factor = float(min(1.0, float(spread_abs_max_pct or 0.0) / spread_ref))

        harmony_factor = float(vwma_slope_harmony_ratio or 0.0)

        pb_weight = 0.0
        if last_pullback_kind == "pullback_weak":
            pb_weight = 0.2
        elif last_pullback_kind == "pullback_medium":
            pb_weight = 0.5
        elif last_pullback_kind == "pullback_strong":
            pb_weight = 0.8
        if break_slow_confirmed:
            pb_weight = max(float(pb_weight), 1.0)

        pb_recency_factor = 0.0
        if last_pullback_recency is not None and len(w) > 1:
            pb_recency_factor = float(1.0 - float(min(int(last_pullback_recency), int(len(w) - 1))) / float(max(1, int(len(w) - 1))))

        score = float(0.10 + 0.90 * float(pb_weight))
        score *= float(0.25 + 0.75 * float(pb_recency_factor))
        score *= float(0.25 + 0.75 * float(spread_factor))
        score *= float(0.25 + 0.75 * float(harmony_factor))

        is_interesting = bool(int(len(w)) >= int(cfg.min_cycle_len) and float(score) >= float(cfg.min_score))

        events_sorted = sorted(events, key=lambda e: (int(e.pos), int(e.ts)))
        events_out = [asdict(e) for e in events_sorted]

        return DoubleVwmaCycleMetrics(
            cycle_id=int(cycle_id),
            start_i=int(start),
            end_i=int(end),
            start_ts=int(start_ts),
            end_ts=int(end_ts),
            start_dt=self._dt(int(start_ts)),
            end_dt=self._dt(int(end_ts)),
            vwma_fast_col=str(cfg.vwma_fast_col),
            vwma_slow_col=str(cfg.vwma_slow_col),
            zone_fast_radius_pct=float(zf),
            zone_slow_radius_pct=float(zs),
            trend_side=str(trend_side),
            spread_abs_max_pct=(None if spread_abs_max_pct is None else float(spread_abs_max_pct)),
            spread_abs_end_pct=(None if spread_abs_end_pct is None else float(spread_abs_end_pct)),
            spread_abs_slope_mean_pct=(None if spread_abs_slope_mean_pct is None else float(spread_abs_slope_mean_pct)),
            vwma_fast_slope_mean_pct=(None if vwma_fast_slope_mean_pct is None else float(vwma_fast_slope_mean_pct)),
            vwma_slow_slope_mean_pct=(None if vwma_slow_slope_mean_pct is None else float(vwma_slow_slope_mean_pct)),
            vwma_slope_harmony_ratio=(None if vwma_slope_harmony_ratio is None else float(vwma_slope_harmony_ratio)),
            pullback_weak_count=int(pullback_weak_count),
            pullback_medium_count=int(pullback_medium_count),
            pullback_strong_count=int(pullback_strong_count),
            last_pullback_kind=(None if last_pullback_kind is None else str(last_pullback_kind)),
            last_pullback_end_pos=(None if last_pullback_end_pos is None else int(last_pullback_end_pos)),
            last_pullback_recency=(None if last_pullback_recency is None else int(last_pullback_recency)),
            break_confirm_bars=int(break_confirm_bars),
            break_slow_confirmed=bool(break_slow_confirmed),
            score=float(score),
            is_interesting=bool(is_interesting),
            events=list(events_out),
        )

    def analyze_df(self, df: pd.DataFrame, *, max_cycles: int = 0) -> list[DoubleVwmaCycleMetrics]:
        cycles = self._segment_cycles(df)
        if not cycles:
            return []

        selected = cycles[-int(max_cycles) :] if int(max_cycles) > 0 else cycles
        base_id = int(max(0, int(len(cycles)) - int(len(selected))))

        out: list[DoubleVwmaCycleMetrics] = []
        for i, (a, b) in enumerate(selected):
            m = self._analyze_cycle(df, cycle_id=int(base_id + int(i)), start=int(a), end=int(b))
            if m is not None:
                out.append(m)

        out.sort(key=lambda x: int(x.cycle_id))
        return out

    def current_df(self, df: pd.DataFrame) -> DoubleVwmaCycleMetrics | None:
        cycles = self._segment_cycles(df)
        if not cycles:
            return None
        last_start, last_end = cycles[-1]
        return self._analyze_cycle(df, cycle_id=int(len(cycles) - 1), start=int(last_start), end=int(last_end))

    def answer(self, *, question: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
        kind = str(question.get("kind") or "").strip().lower()
        top_n = int(question.get("top_n") or 6)

        if kind in {"analyze", "analyze_double_vwma_cycles"}:
            max_cycles = int(question.get("max_cycles") or 0)
            cycles = self.analyze_df(df, max_cycles=int(max_cycles))
            return {
                "kind": "analyze",
                "max_cycles": int(max_cycles),
                "metrics": [asdict(c) for c in cycles],
            }

        if kind in {"current", "current_double_vwma_cycle"}:
            m = self.current_df(df)
            return {
                "kind": "current",
                "metric": (None if m is None else asdict(m)),
            }

        if kind in {"rank_double_vwma_cycles", "rank", ""}:
            max_cycles_raw = question.get("max_cycles")
            max_cycles = int(8 if max_cycles_raw is None else int(max_cycles_raw))
            cycles = self.analyze_df(df, max_cycles=int(max_cycles))

            ranked_all = sorted(cycles, key=lambda x: (float(x.score), int(x.end_ts)), reverse=True)
            ranked = ranked_all[: int(top_n)]

            interesting_all = [c for c in cycles if bool(c.is_interesting)]
            interesting_all.sort(key=lambda x: (float(x.score), int(x.end_ts)), reverse=True)
            interesting = interesting_all[: int(top_n)]

            return {
                "kind": "rank_double_vwma_cycles",
                "top_n": int(top_n),
                "max_cycles": int(max_cycles),
                "ranked": [asdict(c) for c in ranked],
                "interesting": [asdict(c) for c in interesting],
            }

        raise ValueError(f"Unsupported question.kind: {question.get('kind')}")
