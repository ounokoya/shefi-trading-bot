from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from libs.blocks.segment_macd_hist_tranches_df import segment_macd_hist_tranches_df


@dataclass(frozen=True)
class TripleCciTrancheMetrics:
    tranche_id: int
    tranche_sign: str
    tranche_type: str

    tranche_len: int
    tranche_start_i: int
    tranche_end_i: int
    tranche_start_ts: int
    tranche_end_ts: int
    tranche_start_dt: str
    tranche_end_dt: str

    cci_fast_col: str
    cci_medium_col: str
    cci_slow_col: str
    cci_fast_period: int
    cci_medium_period: int
    cci_slow_period: int

    cci_weight_fast: float
    cci_weight_medium: float
    cci_weight_slow: float
    cci_weight_sum: float

    cci_global_extreme: float | None
    cci_global_extreme_abs: float | None
    cci_global_extreme_pos: int | None
    cci_global_extreme_ts: int | None

    cci_fast_at_extreme: float | None
    cci_medium_at_extreme: float | None
    cci_slow_at_extreme: float | None

    cci_global_last_extreme: float | None
    cci_global_last_extreme_abs: float | None
    cci_global_last_extreme_pos: int | None
    cci_global_last_extreme_ts: int | None

    cci_fast_at_last_extreme: float | None
    cci_medium_at_last_extreme: float | None
    cci_slow_at_last_extreme: float | None

    score: float
    is_interesting: bool


@dataclass(frozen=True)
class TripleCciTrancheAgentConfig:
    ts_col: str = "ts"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    hist_col: str = "macd_hist"
    extremes_on: str = "close"

    cci_fast_col: str = "cci_30"
    cci_medium_col: str = "cci_120"
    cci_slow_col: str = "cci_300"

    cci_fast_period: int = 30
    cci_medium_period: int = 120
    cci_slow_period: int = 300

    min_tranche_len: int = 5
    min_abs_cci_global: float = 100.0


class TripleCciTrancheAgent:
    def __init__(self, *, cfg: TripleCciTrancheAgentConfig | None = None):
        self.cfg = cfg or TripleCciTrancheAgentConfig()

    def _weights(self) -> tuple[float, float, float, float] | None:
        cfg = self.cfg
        p_fast = int(cfg.cci_fast_period)
        p_med = int(cfg.cci_medium_period)
        p_slow = int(cfg.cci_slow_period)
        if p_fast <= 0 or p_med <= 0 or p_slow <= 0:
            return None

        w_fast = 1.0
        w_med = float(p_med) / float(p_fast)
        w_slow = float(p_slow) / float(p_fast)
        w_sum = float(w_fast + w_med + w_slow)
        if w_sum <= 0.0:
            return None
        return float(w_fast), float(w_med), float(w_slow), float(w_sum)

    def _global_cci(self, tranche: pd.DataFrame) -> tuple[np.ndarray, tuple[float, float, float, float] | None]:
        cfg = self.cfg
        w = self._weights()
        if w is None:
            return np.array([], dtype=float), None
        w_fast, w_med, w_slow, w_sum = w

        fast = pd.to_numeric(tranche[cfg.cci_fast_col], errors="coerce").astype(float).to_numpy()
        med = pd.to_numeric(tranche[cfg.cci_medium_col], errors="coerce").astype(float).to_numpy()
        slow = pd.to_numeric(tranche[cfg.cci_slow_col], errors="coerce").astype(float).to_numpy()

        ok = np.isfinite(fast) & np.isfinite(med) & np.isfinite(slow)
        g = np.full(int(len(tranche)), np.nan, dtype=float)
        g[ok] = (fast[ok] * float(w_fast) + med[ok] * float(w_med) + slow[ok] * float(w_slow)) / float(w_sum)
        return g, w

    def _global_extreme(self, g: np.ndarray, *, dir_sign: float) -> tuple[float | None, int | None]:
        if g.size == 0:
            return None, None
        g2 = np.where(np.isfinite(g), g, np.nan)
        if not np.isfinite(g2).any():
            return None, None
        if float(dir_sign) > 0:
            pos = int(np.nanargmax(g2))
        else:
            pos = int(np.nanargmin(g2))
        v = float(g2[pos]) if math.isfinite(float(g2[pos])) else None
        return v, pos

    def _last_local_extreme(self, g: np.ndarray, *, dir_sign: float) -> tuple[float | None, int | None]:
        n = int(g.size)
        if n < 3:
            return None, None

        last_pos: int | None = None
        for i in range(1, n - 1):
            a = float(g[i - 1])
            b = float(g[i])
            c = float(g[i + 1])
            if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
                continue

            if float(dir_sign) > 0:
                if (b >= a and b >= c) and (b > a or b > c):
                    last_pos = int(i)
            else:
                if (b <= a and b <= c) and (b < a or b < c):
                    last_pos = int(i)

        if last_pos is None:
            return None, None
        v = float(g[last_pos])
        return (v if math.isfinite(v) else None), int(last_pos)

    def _to_metrics(self, tranche: pd.DataFrame) -> TripleCciTrancheMetrics | None:
        cfg = self.cfg
        if len(tranche) == 0:
            return None

        need = {cfg.ts_col, cfg.cci_fast_col, cfg.cci_medium_col, cfg.cci_slow_col}
        if not need.issubset(set(tranche.columns)):
            return None

        try:
            tranche_start_i = int(min(tranche.index))
            tranche_end_i = int(max(tranche.index))
        except Exception:
            tranche_start_i = 0
            tranche_end_i = int(len(tranche) - 1)

        tid0 = tranche.get("tranche_id")
        if tid0 is None:
            return None
        try:
            tid = int(pd.to_numeric(tid0.iloc[0], errors="coerce"))
        except Exception:
            return None

        sign = str(tranche.get("tranche_sign").iloc[0])
        tranche_type = "haussier" if sign == "+" else "baissier"
        dir_sign = 1.0 if str(sign) == "+" else -1.0

        ts0 = pd.to_numeric(tranche[cfg.ts_col], errors="coerce").astype("Int64")
        tranche_start_ts = int(ts0.iloc[0])
        tranche_end_ts = int(ts0.iloc[-1])
        tranche_start_dt = (
            pd.to_datetime(int(tranche_start_ts), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")
            if tranche_start_ts > 0
            else ""
        )
        tranche_end_dt = (
            pd.to_datetime(int(tranche_end_ts), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")
            if tranche_end_ts > 0
            else ""
        )

        g, w = self._global_cci(tranche)
        if w is None:
            return None
        w_fast, w_med, w_slow, w_sum = w

        ext_v, ext_p = self._global_extreme(g, dir_sign=float(dir_sign))
        ext_abs = (None if ext_v is None else float(abs(float(ext_v))))

        last_v, last_p = self._last_local_extreme(g, dir_sign=float(dir_sign))
        if last_v is None or last_p is None:
            last_v, last_p = ext_v, ext_p
        last_abs = (None if last_v is None else float(abs(float(last_v))))

        ext_ts = None
        if ext_p is not None:
            try:
                ext_ts = int(pd.to_numeric(tranche[cfg.ts_col], errors="coerce").astype("Int64").iloc[int(ext_p)])
            except Exception:
                ext_ts = None

        last_ts = None
        if last_p is not None:
            try:
                last_ts = int(pd.to_numeric(tranche[cfg.ts_col], errors="coerce").astype("Int64").iloc[int(last_p)])
            except Exception:
                last_ts = None

        cci_fast_at_ext = None
        cci_med_at_ext = None
        cci_slow_at_ext = None
        if ext_p is not None:
            try:
                v = float(pd.to_numeric(tranche[cfg.cci_fast_col], errors="coerce").astype(float).iloc[int(ext_p)])
                cci_fast_at_ext = (v if math.isfinite(v) else None)
            except Exception:
                cci_fast_at_ext = None
            try:
                v = float(pd.to_numeric(tranche[cfg.cci_medium_col], errors="coerce").astype(float).iloc[int(ext_p)])
                cci_med_at_ext = (v if math.isfinite(v) else None)
            except Exception:
                cci_med_at_ext = None
            try:
                v = float(pd.to_numeric(tranche[cfg.cci_slow_col], errors="coerce").astype(float).iloc[int(ext_p)])
                cci_slow_at_ext = (v if math.isfinite(v) else None)
            except Exception:
                cci_slow_at_ext = None

        cci_fast_at_last = None
        cci_med_at_last = None
        cci_slow_at_last = None
        if last_p is not None:
            try:
                v = float(pd.to_numeric(tranche[cfg.cci_fast_col], errors="coerce").astype(float).iloc[int(last_p)])
                cci_fast_at_last = (v if math.isfinite(v) else None)
            except Exception:
                cci_fast_at_last = None
            try:
                v = float(pd.to_numeric(tranche[cfg.cci_medium_col], errors="coerce").astype(float).iloc[int(last_p)])
                cci_med_at_last = (v if math.isfinite(v) else None)
            except Exception:
                cci_med_at_last = None
            try:
                v = float(pd.to_numeric(tranche[cfg.cci_slow_col], errors="coerce").astype(float).iloc[int(last_p)])
                cci_slow_at_last = (v if math.isfinite(v) else None)
            except Exception:
                cci_slow_at_last = None

        score = float(ext_abs or 0.0)
        is_interesting = bool(int(len(tranche)) >= int(cfg.min_tranche_len) and score >= float(cfg.min_abs_cci_global))

        return TripleCciTrancheMetrics(
            tranche_id=int(tid),
            tranche_sign=str(sign),
            tranche_type=str(tranche_type),
            tranche_len=int(len(tranche)),
            tranche_start_i=int(tranche_start_i),
            tranche_end_i=int(tranche_end_i),
            tranche_start_ts=int(tranche_start_ts),
            tranche_end_ts=int(tranche_end_ts),
            tranche_start_dt=str(tranche_start_dt),
            tranche_end_dt=str(tranche_end_dt),
            cci_fast_col=str(cfg.cci_fast_col),
            cci_medium_col=str(cfg.cci_medium_col),
            cci_slow_col=str(cfg.cci_slow_col),
            cci_fast_period=int(cfg.cci_fast_period),
            cci_medium_period=int(cfg.cci_medium_period),
            cci_slow_period=int(cfg.cci_slow_period),
            cci_weight_fast=float(w_fast),
            cci_weight_medium=float(w_med),
            cci_weight_slow=float(w_slow),
            cci_weight_sum=float(w_sum),
            cci_global_extreme=(None if ext_v is None else float(ext_v)),
            cci_global_extreme_abs=(None if ext_abs is None else float(ext_abs)),
            cci_global_extreme_pos=(None if ext_p is None else int(ext_p)),
            cci_global_extreme_ts=(None if ext_ts is None else int(ext_ts)),

            cci_fast_at_extreme=(None if cci_fast_at_ext is None else float(cci_fast_at_ext)),
            cci_medium_at_extreme=(None if cci_med_at_ext is None else float(cci_med_at_ext)),
            cci_slow_at_extreme=(None if cci_slow_at_ext is None else float(cci_slow_at_ext)),

            cci_global_last_extreme=(None if last_v is None else float(last_v)),
            cci_global_last_extreme_abs=(None if last_abs is None else float(last_abs)),
            cci_global_last_extreme_pos=(None if last_p is None else int(last_p)),
            cci_global_last_extreme_ts=(None if last_ts is None else int(last_ts)),

            cci_fast_at_last_extreme=(None if cci_fast_at_last is None else float(cci_fast_at_last)),
            cci_medium_at_last_extreme=(None if cci_med_at_last is None else float(cci_med_at_last)),
            cci_slow_at_last_extreme=(None if cci_slow_at_last is None else float(cci_slow_at_last)),

            score=float(score),
            is_interesting=bool(is_interesting),
        )

    def analyze_df(self, df: pd.DataFrame, *, max_tranches: int = 0) -> list[TripleCciTrancheMetrics]:
        cfg = self.cfg
        work = segment_macd_hist_tranches_df(
            df,
            ts_col=str(cfg.ts_col),
            high_col=str(cfg.high_col),
            low_col=str(cfg.low_col),
            close_col=str(cfg.close_col),
            hist_col=str(cfg.hist_col),
            extremes_on=str(cfg.extremes_on),
        )
        if "tranche_id" not in work.columns:
            return []

        tids = pd.to_numeric(work["tranche_id"], errors="coerce").dropna().astype(int).tolist()
        if not tids:
            return []

        uniq = sorted(set(tids))
        selected = uniq[-int(max_tranches) :] if int(max_tranches) > 0 else uniq

        out: list[TripleCciTrancheMetrics] = []
        for tid in selected:
            tdf = work.loc[work["tranche_id"].astype("Int64") == int(tid)]
            m = self._to_metrics(tdf)
            if m is not None:
                out.append(m)

        out.sort(key=lambda x: int(x.tranche_id))
        return out

    def current_df(self, df: pd.DataFrame) -> TripleCciTrancheMetrics | None:
        cfg = self.cfg
        work = segment_macd_hist_tranches_df(
            df,
            ts_col=str(cfg.ts_col),
            high_col=str(cfg.high_col),
            low_col=str(cfg.low_col),
            close_col=str(cfg.close_col),
            hist_col=str(cfg.hist_col),
            extremes_on=str(cfg.extremes_on),
        )
        if "tranche_id" not in work.columns:
            return None

        tids_s = pd.to_numeric(work["tranche_id"], errors="coerce").dropna().astype(int)
        if tids_s.empty:
            return None

        last_tid = int(tids_s.iloc[-1])
        tdf = work.loc[work["tranche_id"].astype("Int64") == int(last_tid)]
        return self._to_metrics(tdf)

    def answer(self, *, question: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
        kind = str(question.get("kind") or "").strip().lower()
        max_tranches = int(question.get("max_tranches") or 0)

        if kind in {"", "analyze", "analyze_triple_cci_tranches"}:
            metrics = self.analyze_df(df, max_tranches=int(max_tranches))
            return {
                "kind": "analyze",
                "max_tranches": int(max_tranches),
                "metrics": [asdict(m) for m in metrics],
            }

        if kind in {"current", "current_triple_cci_tranche"}:
            m = self.current_df(df)
            return {
                "kind": "current",
                "metric": (None if m is None else asdict(m)),
            }

        raise ValueError(f"Unsupported question.kind: {question.get('kind')}")
