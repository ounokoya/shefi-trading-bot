from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


def _safe_float(v: object) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return float(x)


def _cross_up(a0: float, a1: float, b0: float, b1: float) -> bool:
    return bool(float(a0) <= float(b0) and float(a1) > float(b1))


def _cross_down(a0: float, a1: float, b0: float, b1: float) -> bool:
    return bool(float(a0) >= float(b0) and float(a1) < float(b1))


def _rolling_linreg_slope(y: np.ndarray, *, window: int) -> np.ndarray:
    n = int(y.size)
    w = int(window)
    out = np.full(n, np.nan, dtype=float)
    if n <= 0 or w <= 1 or n < w:
        return out

    x = np.arange(w, dtype=float)
    x_mean = float((w - 1) / 2.0)
    x_centered = x - float(x_mean)
    var_x = float(np.sum(x_centered * x_centered))
    if not math.isfinite(var_x) or var_x <= 0.0:
        return out

    for end in range(w - 1, n):
        start = int(end - w + 1)
        win = y[start : end + 1]
        if int(win.size) != w:
            continue
        if not np.isfinite(win).all():
            continue
        cov = float(np.sum(x_centered * win))
        out[end] = float(cov / var_x)

    return out


@dataclass(frozen=True)
class DoubleStochGlobalEvent:
    kind: str
    pos: int
    ts: int
    dt: str
    side: str
    meta: dict[str, object]


@dataclass(frozen=True)
class DoubleStochGlobalMetrics:
    pos: int
    ts: int
    dt: str

    k_global: float | None
    d_global: float | None
    spread: float | None

    k_mid: float | None
    d_mid: float | None

    k_slope: float | None
    d_slope: float | None
    spread_slope: float | None

    k_extreme_high: bool
    k_extreme_low: bool
    d_extreme_high: bool
    d_extreme_low: bool

    global_extreme_high: bool
    global_extreme_low: bool

    bull_cross: bool
    bear_cross: bool

    weight_fast: float
    weight_slow: float
    weight_sum: float


@dataclass(frozen=True)
class DoubleStochGlobalAgentConfig:
    ts_col: str = "ts"
    dt_col: str = "dt"

    k_fast_col: str = "stoch_k_fast"
    d_fast_col: str = "stoch_d_fast"
    k_slow_col: str = "stoch_k_slow"
    d_slow_col: str = "stoch_d_slow"

    k_period_fast: int = 14
    k_period_slow: int = 56

    out_k_col: str = "stoch_k_global"
    out_d_col: str = "stoch_d_global"
    out_spread_col: str = "stoch_spread_global"
    out_k_mid_col: str = "stoch_k_global_mid"
    out_d_mid_col: str = "stoch_d_global_mid"

    out_k_slope_col: str = "stoch_k_global_slope"
    out_d_slope_col: str = "stoch_d_global_slope"
    out_spread_slope_col: str = "stoch_spread_global_slope"

    out_k_extreme_high_col: str = "stoch_k_global_extreme_high"
    out_k_extreme_low_col: str = "stoch_k_global_extreme_low"
    out_d_extreme_high_col: str = "stoch_d_global_extreme_high"
    out_d_extreme_low_col: str = "stoch_d_global_extreme_low"

    out_global_extreme_high_col: str = "stoch_global_extreme_high"
    out_global_extreme_low_col: str = "stoch_global_extreme_low"

    out_bull_cross_col: str = "stoch_global_bull_cross"
    out_bear_cross_col: str = "stoch_global_bear_cross"

    extreme_high: float = 80.0
    extreme_low: float = 20.0

    extreme_mode: str = "k_only"  # k_only | d_only | k_and_d

    min_cross_gap_global: float = 0.0

    slope_mode: str = "delta"  # delta | linreg
    slope_len: int = 5


class DoubleStochGlobalAgent:
    def __init__(self, *, cfg: DoubleStochGlobalAgentConfig | None = None):
        self.cfg = cfg or DoubleStochGlobalAgentConfig()

    def _dt(self, ts: int) -> str:
        if int(ts) <= 0:
            return ""
        return str(pd.to_datetime(int(ts), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC"))

    def _weights(self) -> tuple[float, float, float] | None:
        cfg = self.cfg
        p_fast = int(cfg.k_period_fast)
        p_slow = int(cfg.k_period_slow)
        if p_fast <= 0 or p_slow <= 0:
            return None

        w_fast = 1.0
        w_slow = float(p_slow) / float(p_fast)
        w_sum = float(w_fast + w_slow)
        if not math.isfinite(w_sum) or w_sum <= 0.0:
            return None
        return float(w_fast), float(w_slow), float(w_sum)

    def enrich_df(self, df: pd.DataFrame, *, in_place: bool = False) -> pd.DataFrame:
        cfg = self.cfg
        need = {
            str(cfg.ts_col),
            str(cfg.k_fast_col),
            str(cfg.d_fast_col),
            str(cfg.k_slow_col),
            str(cfg.d_slow_col),
        }
        miss = sorted([c for c in need if c not in df.columns])
        if miss:
            raise ValueError(f"Missing required columns: {miss}")

        w = self._weights()
        if w is None:
            raise ValueError("Invalid k_periods: cannot compute weights")
        w_fast, w_slow, w_sum = w

        ts_s = pd.to_numeric(df[str(cfg.ts_col)], errors="coerce").astype("Int64").to_numpy()
        k_fast = pd.to_numeric(df[str(cfg.k_fast_col)], errors="coerce").astype(float).to_numpy()
        d_fast = pd.to_numeric(df[str(cfg.d_fast_col)], errors="coerce").astype(float).to_numpy()
        k_slow = pd.to_numeric(df[str(cfg.k_slow_col)], errors="coerce").astype(float).to_numpy()
        d_slow = pd.to_numeric(df[str(cfg.d_slow_col)], errors="coerce").astype(float).to_numpy()

        ok_k = np.isfinite(k_fast) & np.isfinite(k_slow)
        ok_d = np.isfinite(d_fast) & np.isfinite(d_slow)

        k_global = np.full(int(len(df)), np.nan, dtype=float)
        d_global = np.full(int(len(df)), np.nan, dtype=float)

        k_global[ok_k] = (k_fast[ok_k] * float(w_fast) + k_slow[ok_k] * float(w_slow)) / float(w_sum)
        d_global[ok_d] = (d_fast[ok_d] * float(w_fast) + d_slow[ok_d] * float(w_slow)) / float(w_sum)

        spread = np.where(np.isfinite(k_global) & np.isfinite(d_global), k_global - d_global, np.nan)
        k_mid = np.where(np.isfinite(k_global), k_global - 50.0, np.nan)
        d_mid = np.where(np.isfinite(d_global), d_global - 50.0, np.nan)

        high = float(cfg.extreme_high)
        low = float(cfg.extreme_low)
        k_ext_hi = np.where(np.isfinite(k_global), k_global >= high, False)
        k_ext_lo = np.where(np.isfinite(k_global), k_global <= low, False)
        d_ext_hi = np.where(np.isfinite(d_global), d_global >= high, False)
        d_ext_lo = np.where(np.isfinite(d_global), d_global <= low, False)

        extreme_mode = str(cfg.extreme_mode or "k_only").strip().lower()
        if extreme_mode not in {"k_only", "d_only", "k_and_d"}:
            raise ValueError(f"Unsupported extreme_mode: {cfg.extreme_mode}")

        if extreme_mode == "k_only":
            g_ext_hi = k_ext_hi
            g_ext_lo = k_ext_lo
        elif extreme_mode == "d_only":
            g_ext_hi = d_ext_hi
            g_ext_lo = d_ext_lo
        else:
            g_ext_hi = k_ext_hi & d_ext_hi
            g_ext_lo = k_ext_lo & d_ext_lo

        bull_cross = np.full(int(len(df)), False, dtype=bool)
        bear_cross = np.full(int(len(df)), False, dtype=bool)
        min_gap = float(cfg.min_cross_gap_global)

        for i in range(1, int(len(df))):
            s0 = _safe_float(spread[i - 1])
            s1 = _safe_float(spread[i])
            if s0 is None or s1 is None:
                continue
            if float(min_gap) > 0.0 and float(abs(s1)) < float(min_gap):
                continue

            if _cross_up(float(s0), float(s1), 0.0, 0.0):
                bull_cross[i] = True
            elif _cross_down(float(s0), float(s1), 0.0, 0.0):
                bear_cross[i] = True

        slope_mode = str(cfg.slope_mode or "delta").strip().lower()
        if slope_mode not in {"delta", "linreg"}:
            raise ValueError(f"Unsupported slope_mode: {cfg.slope_mode}")

        n_slope = int(cfg.slope_len)
        k_slope = np.full(int(len(df)), np.nan, dtype=float)
        d_slope = np.full(int(len(df)), np.nan, dtype=float)
        spread_slope = np.full(int(len(df)), np.nan, dtype=float)

        if slope_mode == "delta":
            if n_slope > 0:
                for i in range(n_slope, int(len(df))):
                    a0 = _safe_float(k_global[i - n_slope])
                    a1 = _safe_float(k_global[i])
                    b0 = _safe_float(d_global[i - n_slope])
                    b1 = _safe_float(d_global[i])
                    s0 = _safe_float(spread[i - n_slope])
                    s1 = _safe_float(spread[i])

                    if a0 is not None and a1 is not None:
                        k_slope[i] = float(a1 - a0)
                    if b0 is not None and b1 is not None:
                        d_slope[i] = float(b1 - b0)
                    if s0 is not None and s1 is not None:
                        spread_slope[i] = float(s1 - s0)
        else:
            k_slope = _rolling_linreg_slope(k_global.astype(float), window=int(n_slope))
            d_slope = _rolling_linreg_slope(d_global.astype(float), window=int(n_slope))
            spread_slope = _rolling_linreg_slope(spread.astype(float), window=int(n_slope))

        out = df if bool(in_place) else df.copy()

        out[str(cfg.out_k_col)] = k_global
        out[str(cfg.out_d_col)] = d_global
        out[str(cfg.out_spread_col)] = spread
        out[str(cfg.out_k_mid_col)] = k_mid
        out[str(cfg.out_d_mid_col)] = d_mid

        out[str(cfg.out_k_slope_col)] = k_slope
        out[str(cfg.out_d_slope_col)] = d_slope
        out[str(cfg.out_spread_slope_col)] = spread_slope

        out[str(cfg.out_k_extreme_high_col)] = k_ext_hi
        out[str(cfg.out_k_extreme_low_col)] = k_ext_lo
        out[str(cfg.out_d_extreme_high_col)] = d_ext_hi
        out[str(cfg.out_d_extreme_low_col)] = d_ext_lo

        out[str(cfg.out_global_extreme_high_col)] = g_ext_hi
        out[str(cfg.out_global_extreme_low_col)] = g_ext_lo

        out[str(cfg.out_bull_cross_col)] = bull_cross
        out[str(cfg.out_bear_cross_col)] = bear_cross

        if str(cfg.dt_col) not in out.columns:
            try:
                out[str(cfg.dt_col)] = pd.to_datetime(ts_s, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                out[str(cfg.dt_col)] = ""

        return out

    def current_metrics(self, df: pd.DataFrame) -> DoubleStochGlobalMetrics | None:
        cfg = self.cfg
        work = self.enrich_df(df, in_place=False)
        if len(work) <= 0:
            return None

        i = int(len(work) - 1)
        ts0 = pd.to_numeric(work[str(cfg.ts_col)], errors="coerce").astype("Int64")
        ts_i = int(ts0.iloc[i]) if not ts0.empty and ts0.iloc[i] is not None else 0

        dt = ""
        if str(cfg.dt_col) in work.columns:
            try:
                dt = str(work[str(cfg.dt_col)].iloc[i])
            except Exception:
                dt = ""
        if not dt:
            dt = self._dt(int(ts_i))

        w = self._weights()
        if w is None:
            return None
        w_fast, w_slow, w_sum = w

        k_g = _safe_float(work[str(cfg.out_k_col)].iloc[i]) if str(cfg.out_k_col) in work.columns else None
        d_g = _safe_float(work[str(cfg.out_d_col)].iloc[i]) if str(cfg.out_d_col) in work.columns else None
        spr = _safe_float(work[str(cfg.out_spread_col)].iloc[i]) if str(cfg.out_spread_col) in work.columns else None

        k_mid = _safe_float(work[str(cfg.out_k_mid_col)].iloc[i]) if str(cfg.out_k_mid_col) in work.columns else None
        d_mid = _safe_float(work[str(cfg.out_d_mid_col)].iloc[i]) if str(cfg.out_d_mid_col) in work.columns else None

        k_sl = _safe_float(work[str(cfg.out_k_slope_col)].iloc[i]) if str(cfg.out_k_slope_col) in work.columns else None
        d_sl = _safe_float(work[str(cfg.out_d_slope_col)].iloc[i]) if str(cfg.out_d_slope_col) in work.columns else None
        s_sl = (
            _safe_float(work[str(cfg.out_spread_slope_col)].iloc[i]) if str(cfg.out_spread_slope_col) in work.columns else None
        )

        def _b(col: str) -> bool:
            if col not in work.columns:
                return False
            try:
                return bool(work[str(col)].iloc[i])
            except Exception:
                return False

        return DoubleStochGlobalMetrics(
            pos=int(i),
            ts=int(ts_i),
            dt=str(dt),
            k_global=(None if k_g is None else float(k_g)),
            d_global=(None if d_g is None else float(d_g)),
            spread=(None if spr is None else float(spr)),
            k_mid=(None if k_mid is None else float(k_mid)),
            d_mid=(None if d_mid is None else float(d_mid)),
            k_slope=(None if k_sl is None else float(k_sl)),
            d_slope=(None if d_sl is None else float(d_sl)),
            spread_slope=(None if s_sl is None else float(s_sl)),
            k_extreme_high=bool(_b(str(cfg.out_k_extreme_high_col))),
            k_extreme_low=bool(_b(str(cfg.out_k_extreme_low_col))),
            d_extreme_high=bool(_b(str(cfg.out_d_extreme_high_col))),
            d_extreme_low=bool(_b(str(cfg.out_d_extreme_low_col))),
            global_extreme_high=bool(_b(str(cfg.out_global_extreme_high_col))),
            global_extreme_low=bool(_b(str(cfg.out_global_extreme_low_col))),
            bull_cross=bool(_b(str(cfg.out_bull_cross_col))),
            bear_cross=bool(_b(str(cfg.out_bear_cross_col))),
            weight_fast=float(w_fast),
            weight_slow=float(w_slow),
            weight_sum=float(w_sum),
        )

    def find_cross_events(self, df: pd.DataFrame, *, max_events: int = 200) -> list[DoubleStochGlobalEvent]:
        cfg = self.cfg
        work = self.enrich_df(df, in_place=False)

        ts_s = pd.to_numeric(work[str(cfg.ts_col)], errors="coerce").astype("Int64").to_numpy()

        out: list[DoubleStochGlobalEvent] = []
        for i in range(1, int(len(work))):
            is_bull = bool(work[str(cfg.out_bull_cross_col)].iloc[i])
            is_bear = bool(work[str(cfg.out_bear_cross_col)].iloc[i])
            if (not is_bull) and (not is_bear):
                continue

            side = "LONG" if is_bull else "SHORT"

            ts_i = int(ts_s[i]) if 0 <= int(i) < len(ts_s) and ts_s[i] is not None else 0
            dt = ""
            if str(cfg.dt_col) in work.columns:
                try:
                    dt = str(work[str(cfg.dt_col)].iloc[i])
                except Exception:
                    dt = ""
            if not dt:
                dt = self._dt(int(ts_i))

            meta: dict[str, object] = {
                "k_global": _safe_float(work[str(cfg.out_k_col)].iloc[i]),
                "d_global": _safe_float(work[str(cfg.out_d_col)].iloc[i]),
                "spread": _safe_float(work[str(cfg.out_spread_col)].iloc[i]),
                "k_slope": _safe_float(work[str(cfg.out_k_slope_col)].iloc[i]),
                "d_slope": _safe_float(work[str(cfg.out_d_slope_col)].iloc[i]),
                "spread_slope": _safe_float(work[str(cfg.out_spread_slope_col)].iloc[i]),
                "global_extreme_high": bool(work[str(cfg.out_global_extreme_high_col)].iloc[i]),
                "global_extreme_low": bool(work[str(cfg.out_global_extreme_low_col)].iloc[i]),
            }

            out.append(
                DoubleStochGlobalEvent(
                    kind="stoch_global_cross",
                    pos=int(i),
                    ts=int(ts_i),
                    dt=str(dt),
                    side=str(side),
                    meta=dict(meta),
                )
            )

        out.sort(key=lambda x: (int(x.pos), int(x.ts)))
        if int(max_events) > 0:
            out = out[-int(max_events) :]
        return out

    def answer(self, *, question: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
        kind = str(question.get("kind") or "").strip().lower()

        if kind in {"", "current", "current_metrics"}:
            m = self.current_metrics(df)
            return {
                "kind": "current",
                "metric": (None if m is None else asdict(m)),
            }

        if kind in {"find_cross", "find_global_cross"}:
            max_events = int(question.get("max_events") or 200)
            events = self.find_cross_events(df, max_events=int(max_events))
            return {
                "kind": "find_cross",
                "max_events": int(max_events),
                "events": [asdict(e) for e in events],
            }

        if kind in {"enrich", "enrich_df"}:
            max_rows = int(question.get("max_rows") or 200)
            work = self.enrich_df(df, in_place=False)
            if int(max_rows) > 0 and int(len(work)) > int(max_rows):
                work = work.iloc[-int(max_rows) :]
            cols = [
                str(self.cfg.ts_col),
                str(self.cfg.dt_col),
                str(self.cfg.out_k_col),
                str(self.cfg.out_d_col),
                str(self.cfg.out_spread_col),
                str(self.cfg.out_k_mid_col),
                str(self.cfg.out_d_mid_col),
                str(self.cfg.out_k_slope_col),
                str(self.cfg.out_d_slope_col),
                str(self.cfg.out_spread_slope_col),
                str(self.cfg.out_global_extreme_high_col),
                str(self.cfg.out_global_extreme_low_col),
                str(self.cfg.out_bull_cross_col),
                str(self.cfg.out_bear_cross_col),
            ]
            cols = [c for c in cols if c in work.columns]
            return {
                "kind": "enrich",
                "max_rows": int(max_rows),
                "columns": cols,
                "rows": work[cols].to_dict(orient="records"),
            }

        raise ValueError(f"Unsupported question.kind: {question.get('kind')}")
