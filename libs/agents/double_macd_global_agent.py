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
class DoubleMacdGlobalEvent:
    kind: str
    pos: int
    ts: int
    dt: str
    side: str
    meta: dict[str, object]


@dataclass(frozen=True)
class DoubleMacdGlobalMetrics:
    pos: int
    ts: int
    dt: str

    hist_global: float | None
    hist_slope: float | None

    above_zero: bool
    below_zero: bool

    bull_cross: bool
    bear_cross: bool

    weight_fast: float
    weight_slow: float
    weight_sum: float


@dataclass(frozen=True)
class DoubleMacdGlobalAgentConfig:
    ts_col: str = "ts"
    dt_col: str = "dt"

    hist_fast_col: str = "macd_hist_fast"
    hist_slow_col: str = "macd_hist_slow"

    slow_period_fast: int = 26
    slow_period_slow: int = 60

    out_hist_col: str = "macd_hist_global"
    out_above_zero_col: str = "macd_hist_global_above_zero"
    out_below_zero_col: str = "macd_hist_global_below_zero"

    out_slope_col: str = "macd_hist_global_slope"

    out_bull_cross_col: str = "macd_hist_global_bull_cross"
    out_bear_cross_col: str = "macd_hist_global_bear_cross"

    min_cross_abs: float = 0.0

    slope_mode: str = "delta"
    slope_len: int = 5


class DoubleMacdGlobalAgent:
    def __init__(self, *, cfg: DoubleMacdGlobalAgentConfig | None = None):
        self.cfg = cfg or DoubleMacdGlobalAgentConfig()

    def _dt(self, ts: int) -> str:
        if int(ts) <= 0:
            return ""
        return str(pd.to_datetime(int(ts), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC"))

    def _weights(self) -> tuple[float, float, float] | None:
        cfg = self.cfg
        l_fast = int(cfg.slow_period_fast)
        l_slow = int(cfg.slow_period_slow)
        if l_fast <= 0 or l_slow <= 0:
            return None

        w_fast = 1.0
        w_slow = float(l_slow) / float(l_fast)
        w_sum = float(w_fast + w_slow)
        if not math.isfinite(w_sum) or w_sum <= 0.0:
            return None
        return float(w_fast), float(w_slow), float(w_sum)

    def enrich_df(self, df: pd.DataFrame, *, in_place: bool = False) -> pd.DataFrame:
        cfg = self.cfg
        need = {str(cfg.ts_col), str(cfg.hist_fast_col), str(cfg.hist_slow_col)}
        miss = sorted([c for c in need if c not in df.columns])
        if miss:
            raise ValueError(f"Missing required columns: {miss}")

        w = self._weights()
        if w is None:
            raise ValueError("Invalid slow_periods: cannot compute weights")
        w_fast, w_slow, w_sum = w

        ts_s = pd.to_numeric(df[str(cfg.ts_col)], errors="coerce").astype("Int64").to_numpy()
        h_fast = pd.to_numeric(df[str(cfg.hist_fast_col)], errors="coerce").astype(float).to_numpy()
        h_slow = pd.to_numeric(df[str(cfg.hist_slow_col)], errors="coerce").astype(float).to_numpy()

        ok = np.isfinite(h_fast) & np.isfinite(h_slow)
        h_global = np.full(int(len(df)), np.nan, dtype=float)
        h_global[ok] = (h_fast[ok] * float(w_fast) + h_slow[ok] * float(w_slow)) / float(w_sum)

        above_zero = np.where(np.isfinite(h_global), h_global > 0.0, False)
        below_zero = np.where(np.isfinite(h_global), h_global < 0.0, False)

        bull_cross = np.full(int(len(df)), False, dtype=bool)
        bear_cross = np.full(int(len(df)), False, dtype=bool)
        min_abs = float(cfg.min_cross_abs)

        for i in range(1, int(len(df))):
            a0 = _safe_float(h_global[i - 1])
            a1 = _safe_float(h_global[i])
            if a0 is None or a1 is None:
                continue
            if float(min_abs) > 0.0 and float(abs(a1)) < float(min_abs):
                continue

            if _cross_up(float(a0), float(a1), 0.0, 0.0):
                bull_cross[i] = True
            elif _cross_down(float(a0), float(a1), 0.0, 0.0):
                bear_cross[i] = True

        slope_mode = str(cfg.slope_mode or "delta").strip().lower()
        if slope_mode not in {"delta", "linreg"}:
            raise ValueError(f"Unsupported slope_mode: {cfg.slope_mode}")

        n_slope = int(cfg.slope_len)
        slope = np.full(int(len(df)), np.nan, dtype=float)

        if slope_mode == "delta":
            if n_slope > 0:
                for i in range(n_slope, int(len(df))):
                    y0 = _safe_float(h_global[i - n_slope])
                    y1 = _safe_float(h_global[i])
                    if y0 is None or y1 is None:
                        continue
                    slope[i] = float(y1 - y0)
        else:
            slope = _rolling_linreg_slope(h_global.astype(float), window=int(n_slope))

        out = df if bool(in_place) else df.copy()

        out[str(cfg.out_hist_col)] = h_global
        out[str(cfg.out_above_zero_col)] = above_zero
        out[str(cfg.out_below_zero_col)] = below_zero
        out[str(cfg.out_bull_cross_col)] = bull_cross
        out[str(cfg.out_bear_cross_col)] = bear_cross
        out[str(cfg.out_slope_col)] = slope

        if str(cfg.dt_col) not in out.columns:
            try:
                out[str(cfg.dt_col)] = pd.to_datetime(ts_s, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                out[str(cfg.dt_col)] = ""

        return out

    def current_metrics(self, df: pd.DataFrame) -> DoubleMacdGlobalMetrics | None:
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

        hist_g = _safe_float(work[str(cfg.out_hist_col)].iloc[i]) if str(cfg.out_hist_col) in work.columns else None
        slope_g = _safe_float(work[str(cfg.out_slope_col)].iloc[i]) if str(cfg.out_slope_col) in work.columns else None

        def _b(col: str) -> bool:
            if col not in work.columns:
                return False
            try:
                return bool(work[str(col)].iloc[i])
            except Exception:
                return False

        return DoubleMacdGlobalMetrics(
            pos=int(i),
            ts=int(ts_i),
            dt=str(dt),
            hist_global=(None if hist_g is None else float(hist_g)),
            hist_slope=(None if slope_g is None else float(slope_g)),
            above_zero=bool(_b(str(cfg.out_above_zero_col))),
            below_zero=bool(_b(str(cfg.out_below_zero_col))),
            bull_cross=bool(_b(str(cfg.out_bull_cross_col))),
            bear_cross=bool(_b(str(cfg.out_bear_cross_col))),
            weight_fast=float(w_fast),
            weight_slow=float(w_slow),
            weight_sum=float(w_sum),
        )

    def find_zero_cross_events(self, df: pd.DataFrame, *, max_events: int = 200) -> list[DoubleMacdGlobalEvent]:
        cfg = self.cfg
        work = self.enrich_df(df, in_place=False)
        ts_s = pd.to_numeric(work[str(cfg.ts_col)], errors="coerce").astype("Int64").to_numpy()

        out: list[DoubleMacdGlobalEvent] = []
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
                "hist_global": _safe_float(work[str(cfg.out_hist_col)].iloc[i]),
                "hist_slope": _safe_float(work[str(cfg.out_slope_col)].iloc[i]),
            }

            out.append(
                DoubleMacdGlobalEvent(
                    kind="macd_hist_global_zero_cross",
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

        if kind in {"find_cross", "find_zero_cross", "find_hist_cross"}:
            max_events = int(question.get("max_events") or 200)
            events = self.find_zero_cross_events(df, max_events=int(max_events))
            return {
                "kind": "find_zero_cross",
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
                str(self.cfg.out_hist_col),
                str(self.cfg.out_slope_col),
                str(self.cfg.out_above_zero_col),
                str(self.cfg.out_below_zero_col),
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
