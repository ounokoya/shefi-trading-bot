from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from libs.blocks.segment_macd_hist_tranches_df import segment_macd_hist_tranches_df


@dataclass(frozen=True)
class TrancheHistMetrics:
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

    force_mean_abs: float
    force_peak_abs: float

    score: float
    is_interesting: bool


@dataclass(frozen=True)
class HistTrancheAgentConfig:
    ts_col: str = "ts"
    close_col: str = "close"
    hist_col: str = "macd_hist"
    min_abs_force: float = 0.0


class MacdHistTrancheAgent:
    def __init__(self, *, cfg: HistTrancheAgentConfig | None = None):
        self.cfg = cfg or HistTrancheAgentConfig()

    def _to_metrics(self, tranche: pd.DataFrame) -> TrancheHistMetrics | None:
        cfg = self.cfg
        if len(tranche) == 0:
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

        hist_arr = pd.to_numeric(tranche[cfg.hist_col], errors="coerce").astype(float).to_numpy()
        close_arr = pd.to_numeric(tranche[cfg.close_col], errors="coerce").astype(float).to_numpy()
        if int(len(hist_arr)) <= 0 or int(len(close_arr)) <= 0:
            return None

        force: list[float] = []
        for h, c in zip(hist_arr.tolist(), close_arr.tolist()):
            if not math.isfinite(float(h)):
                continue
            if not math.isfinite(float(c)):
                continue
            if float(c) == 0.0:
                continue
            force.append(float(float(h) / float(c)))

        if not force:
            return None

        force_abs = [abs(float(x)) for x in force if math.isfinite(float(x))]
        if not force_abs:
            return None

        force_mean_abs = float(sum(force_abs) / float(len(force_abs)))
        force_peak_abs = float(max(force_abs))

        score = float(force_mean_abs)
        is_interesting = bool(float(score) >= float(cfg.min_abs_force))

        return TrancheHistMetrics(
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
            force_mean_abs=float(force_mean_abs),
            force_peak_abs=float(force_peak_abs),
            score=float(score),
            is_interesting=bool(is_interesting),
        )

    def analyze_df(self, df: pd.DataFrame, *, max_tranches: int = 0) -> list[TrancheHistMetrics]:
        cfg = self.cfg
        work = segment_macd_hist_tranches_df(
            df,
            ts_col=str(cfg.ts_col),
            close_col=str(cfg.close_col),
            hist_col=str(cfg.hist_col),
            extremes_on="close",
        )

        if "tranche_id" not in work.columns:
            return []

        tids = pd.to_numeric(work["tranche_id"], errors="coerce").dropna().astype(int).tolist()
        if not tids:
            return []

        uniq = sorted(set(tids))
        selected = uniq[-int(max_tranches) :] if int(max_tranches) > 0 else uniq

        out: list[TrancheHistMetrics] = []
        for tid in selected:
            tdf = work.loc[work["tranche_id"].astype("Int64") == int(tid)]
            m = self._to_metrics(tdf)
            if m is not None:
                out.append(m)

        out.sort(key=lambda x: int(x.tranche_id))
        return out

    def current_df(self, df: pd.DataFrame) -> TrancheHistMetrics | None:
        cfg = self.cfg
        work = segment_macd_hist_tranches_df(
            df,
            ts_col=str(cfg.ts_col),
            close_col=str(cfg.close_col),
            hist_col=str(cfg.hist_col),
            extremes_on="close",
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

        if kind in {"", "analyze", "analyze_hist_tranches", "classify_hist_tranches"}:
            metrics = self.analyze_df(df, max_tranches=int(max_tranches))
            return {
                "kind": "analyze",
                "max_tranches": int(max_tranches),
                "metrics": [asdict(m) for m in metrics],
            }

        if kind in {"current", "current_hist_tranche"}:
            m = self.current_df(df)
            return {
                "kind": "current",
                "metric": (None if m is None else asdict(m)),
            }

        raise ValueError(f"Unsupported question.kind: {question.get('kind')}")
