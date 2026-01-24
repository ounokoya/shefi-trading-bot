from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from libs.agents.macd_hist_tranche_agent import HistTrancheAgentConfig, MacdHistTrancheAgent, TrancheHistMetrics
from libs.agents.triple_cci_tranche_agent import TripleCciTrancheAgent, TripleCciTrancheAgentConfig


@dataclass(frozen=True)
class MacdMomentumTwoTFSignalMetrics:
    signal_kind: str
    signal_ts: int
    signal_dt: str

    exec_tranche_id: int
    exec_tranche_sign: str
    exec_tranche_type: str
    exec_tranche_len: int
    exec_tranche_start_i: int
    exec_tranche_end_i: int
    exec_tranche_start_ts: int
    exec_tranche_end_ts: int
    exec_tranche_start_dt: str
    exec_tranche_end_dt: str
    exec_force_mean_abs: float
    exec_force_peak_abs: float

    ctx_tranche_id: int | None
    ctx_tranche_sign: str | None
    ctx_tranche_type: str | None
    ctx_tranche_len: int | None
    ctx_tranche_start_ts: int | None
    ctx_tranche_end_ts: int | None
    ctx_force_mean_abs: float | None
    ctx_force_peak_abs: float | None

    cci_ctx_last_extreme: float | None
    cci_exec_last_extreme: float | None

    signal_i: int
    exec_i: int | None
    entry_ts: int | None
    entry_dt: str | None
    entry: float | None

    side: str
    status: str
    reason: str

    score: float
    is_interesting: bool


@dataclass(frozen=True)
class MacdMomentumTwoTFConfig:
    ts_col: str = "ts"
    dt_col: str = "dt"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    hist_col: str = "macd_hist"

    cci_fast_col: str = "cci_30"
    cci_medium_col: str = "cci_120"
    cci_slow_col: str = "cci_300"
    cci_fast_period: int = 30
    cci_medium_period: int = 120
    cci_slow_period: int = 300

    cci_ctx_fast_col: str = ""
    cci_ctx_medium_col: str = ""
    cci_ctx_slow_col: str = ""
    cci_ctx_fast_period: int = 0
    cci_ctx_medium_period: int = 0
    cci_ctx_slow_period: int = 0

    cci_exec_fast_col: str = ""
    cci_exec_medium_col: str = ""
    cci_exec_slow_col: str = ""
    cci_exec_fast_period: int = 0
    cci_exec_medium_period: int = 0
    cci_exec_slow_period: int = 0

    min_abs_force_ctx: float = 0.0
    min_abs_force_exec: float = 0.0

    cci_global_extreme_level_ctx: float = 100.0
    cci_global_extreme_level_exec: float = 100.0

    take_exec_cci_extreme_if_ctx_not_extreme: bool = False
    take_exec_and_ctx_cci_extreme: bool = False

    signal_on_ctx_flip_if_exec_aligned: bool = False


class MacdMomentiumTwoTFCycleAgent:
    def __init__(self, *, cfg: MacdMomentumTwoTFConfig | None = None):
        self.cfg = cfg or MacdMomentumTwoTFConfig()

    def _dt_from_ts(self, ts_ms: int | None) -> str:
        if ts_ms is None:
            return ""
        try:
            ts_i = int(ts_ms)
        except Exception:
            return ""
        if ts_i <= 0:
            return ""
        return pd.to_datetime(ts_i, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")

    def _is_extreme_dir(self, v: float | None, *, side: str, level: float) -> bool:
        if v is None:
            return False
        if not math.isfinite(float(v)):
            return False
        if not math.isfinite(float(level)) or float(level) <= 0.0:
            return False
        if str(side) == "LONG":
            return float(v) >= float(level)
        return float(v) <= -float(level)

    def _pick_ctx_for_ts(self, ctx_tranches: list[TrancheHistMetrics], *, ts_ms: int) -> TrancheHistMetrics | None:
        if not ctx_tranches:
            return None

        if int(ts_ms) < int(ctx_tranches[0].tranche_start_ts):
            return None

        for m in reversed(ctx_tranches):
            if int(m.tranche_start_ts) <= int(ts_ms) <= int(m.tranche_end_ts):
                return m
            if int(m.tranche_end_ts) < int(ts_ms):
                return m
        return None

    def _pick_exec_for_ts(self, exec_tranches: list[TrancheHistMetrics], *, ts_ms: int) -> TrancheHistMetrics | None:
        if not exec_tranches:
            return None
        if int(ts_ms) < int(exec_tranches[0].tranche_start_ts):
            return None

        for m in reversed(exec_tranches):
            if int(m.tranche_start_ts) <= int(ts_ms) <= int(m.tranche_end_ts):
                return m
            if int(m.tranche_end_ts) < int(ts_ms):
                return m
        return None

    def analyze_df(
        self,
        df: pd.DataFrame,
        *,
        df_ctx: pd.DataFrame,
        max_signals: int = 0,
    ) -> list[MacdMomentumTwoTFSignalMetrics]:
        cfg = self.cfg
        df_exec = df

        cci_ctx_fast_col = str(cfg.cci_ctx_fast_col or cfg.cci_fast_col)
        cci_ctx_medium_col = str(cfg.cci_ctx_medium_col or cfg.cci_medium_col)
        cci_ctx_slow_col = str(cfg.cci_ctx_slow_col or cfg.cci_slow_col)
        cci_ctx_fast_period = int(cfg.cci_ctx_fast_period or cfg.cci_fast_period)
        cci_ctx_medium_period = int(cfg.cci_ctx_medium_period or cfg.cci_medium_period)
        cci_ctx_slow_period = int(cfg.cci_ctx_slow_period or cfg.cci_slow_period)

        cci_exec_fast_col = str(cfg.cci_exec_fast_col or cfg.cci_fast_col)
        cci_exec_medium_col = str(cfg.cci_exec_medium_col or cfg.cci_medium_col)
        cci_exec_slow_col = str(cfg.cci_exec_slow_col or cfg.cci_slow_col)
        cci_exec_fast_period = int(cfg.cci_exec_fast_period or cfg.cci_fast_period)
        cci_exec_medium_period = int(cfg.cci_exec_medium_period or cfg.cci_medium_period)
        cci_exec_slow_period = int(cfg.cci_exec_slow_period or cfg.cci_slow_period)

        hist_exec_cfg = HistTrancheAgentConfig(
            ts_col=str(cfg.ts_col),
            close_col=str(cfg.close_col),
            hist_col=str(cfg.hist_col),
            min_abs_force=float(cfg.min_abs_force_exec),
        )
        hist_ctx_cfg = HistTrancheAgentConfig(
            ts_col=str(cfg.ts_col),
            close_col=str(cfg.close_col),
            hist_col=str(cfg.hist_col),
            min_abs_force=float(cfg.min_abs_force_ctx),
        )

        exec_tranches = MacdHistTrancheAgent(cfg=hist_exec_cfg).analyze_df(df_exec, max_tranches=0)
        ctx_tranches = MacdHistTrancheAgent(cfg=hist_ctx_cfg).analyze_df(df_ctx, max_tranches=0)

        cci_ctx_cfg = TripleCciTrancheAgentConfig(
            ts_col=str(cfg.ts_col),
            high_col=str(cfg.high_col),
            low_col=str(cfg.low_col),
            close_col=str(cfg.close_col),
            hist_col=str(cfg.hist_col),
            extremes_on="close",
            cci_fast_col=str(cci_ctx_fast_col),
            cci_medium_col=str(cci_ctx_medium_col),
            cci_slow_col=str(cci_ctx_slow_col),
            cci_fast_period=int(cci_ctx_fast_period),
            cci_medium_period=int(cci_ctx_medium_period),
            cci_slow_period=int(cci_ctx_slow_period),
            min_tranche_len=1,
            min_abs_cci_global=0.0,
        )
        cci_exec_cfg = TripleCciTrancheAgentConfig(
            ts_col=str(cfg.ts_col),
            high_col=str(cfg.high_col),
            low_col=str(cfg.low_col),
            close_col=str(cfg.close_col),
            hist_col=str(cfg.hist_col),
            extremes_on="close",
            cci_fast_col=str(cci_exec_fast_col),
            cci_medium_col=str(cci_exec_medium_col),
            cci_slow_col=str(cci_exec_slow_col),
            cci_fast_period=int(cci_exec_fast_period),
            cci_medium_period=int(cci_exec_medium_period),
            cci_slow_period=int(cci_exec_slow_period),
            min_tranche_len=1,
            min_abs_cci_global=0.0,
        )

        cci_ctx_tranches = TripleCciTrancheAgent(cfg=cci_ctx_cfg).analyze_df(df_ctx, max_tranches=0)
        cci_exec_tranches = TripleCciTrancheAgent(cfg=cci_exec_cfg).analyze_df(df_exec, max_tranches=0)

        cci_ctx_by_id = {int(m.tranche_id): m for m in cci_ctx_tranches}
        cci_exec_by_id = {int(m.tranche_id): m for m in cci_exec_tranches}

        if not exec_tranches:
            return []

        selected = exec_tranches
        if int(max_signals) > 0:
            selected = selected[-int(max_signals) :]

        selected_start_ts = 0
        if selected:
            selected_start_ts = int(selected[0].tranche_start_ts)

        try:
            open_arr = pd.to_numeric(df_exec[str(cfg.open_col)], errors="coerce").astype(float).to_numpy()
        except Exception:
            open_arr = np.full(int(len(df_exec)), np.nan, dtype=float)

        try:
            ts_arr = pd.to_numeric(df_exec[str(cfg.ts_col)], errors="coerce").astype("Int64").to_numpy()
        except Exception:
            ts_arr = np.full(int(len(df_exec)), np.nan, dtype=float)

        ts_exec = pd.to_numeric(df_exec[str(cfg.ts_col)], errors="coerce").astype("Int64").to_numpy()
        ts_exec_filled = np.where(pd.isna(ts_exec), -1, ts_exec.astype(np.int64))

        def _signal_i_for_ts(signal_ts: int) -> int | None:
            if int(len(ts_exec_filled)) == 0:
                return None
            if int(signal_ts) > int(ts_exec_filled[-1]):
                return None
            if int(signal_ts) <= int(ts_exec_filled[0]):
                return 0
            i = int(np.searchsorted(ts_exec_filled, int(signal_ts), side="left"))
            if i < 0 or i >= int(len(ts_exec_filled)):
                return None
            return i

        def _build_metric(
            *,
            signal_kind: str,
            signal_ts: int,
            exec_m: TrancheHistMetrics,
            ctx_m: TrancheHistMetrics | None,
            signal_i: int,
        ) -> MacdMomentumTwoTFSignalMetrics:
            side = "LONG" if str(exec_m.tranche_sign) == "+" else "SHORT"

            exec_i = int(signal_i + 1)
            if int(exec_i) < 0 or int(exec_i) >= int(len(df_exec)):
                exec_i2: int | None = None
                entry_ts = None
                entry_dt = None
                entry = None
                status = "REJECT"
                reason = "NO_EXEC_BAR"
            else:
                exec_i2 = int(exec_i)
                try:
                    entry_ts = int(ts_arr[int(exec_i2)]) if not pd.isna(ts_arr[int(exec_i2)]) else None
                except Exception:
                    entry_ts = None
                entry_dt = self._dt_from_ts(entry_ts)
                try:
                    entry = float(open_arr[int(exec_i2)])
                    if not math.isfinite(float(entry)):
                        entry = None
                except Exception:
                    entry = None

                if entry is None or float(entry) <= 0.0:
                    status = "REJECT"
                    reason = "MISSING_ENTRY_PRICE"
                else:
                    status = "PENDING"
                    reason = ""

            ctx_tid = None if ctx_m is None else int(ctx_m.tranche_id)
            ctx_sign = None if ctx_m is None else str(ctx_m.tranche_sign)

            cci_ctx_last = None
            cci_exec_last = None

            if ctx_tid is not None:
                cci_ctx = cci_ctx_by_id.get(int(ctx_tid))
                cci_ctx_last = None if cci_ctx is None else cci_ctx.cci_global_last_extreme

            cci_exec = cci_exec_by_id.get(int(exec_m.tranche_id))
            cci_exec_last = None if cci_exec is None else cci_exec.cci_global_last_extreme

            if status == "PENDING":
                if ctx_m is None:
                    status = "REJECT"
                    reason = "MISSING_CONTEXT"
                elif str(ctx_sign) != str(exec_m.tranche_sign):
                    status = "REJECT"
                    reason = "CONTEXT_NOT_ALIGNED"
                elif float(cfg.min_abs_force_ctx) > 0.0 and float(ctx_m.force_mean_abs) < float(cfg.min_abs_force_ctx):
                    status = "REJECT"
                    reason = "CONTEXT_FORCE_TOO_LOW"
                elif float(cfg.min_abs_force_exec) > 0.0 and float(exec_m.force_mean_abs) < float(cfg.min_abs_force_exec):
                    status = "REJECT"
                    reason = "EXEC_FORCE_TOO_LOW"
                elif cci_ctx_last is None or (not math.isfinite(float(cci_ctx_last))):
                    status = "REJECT"
                    reason = "MISSING_CCI_CONTEXT"
                elif cci_exec_last is None or (not math.isfinite(float(cci_exec_last))):
                    status = "REJECT"
                    reason = "MISSING_CCI_EXEC"
                else:
                    ctx_is_extreme = self._is_extreme_dir(
                        cci_ctx_last, side=str(side), level=float(cfg.cci_global_extreme_level_ctx)
                    )
                    exec_is_extreme = self._is_extreme_dir(
                        cci_exec_last, side=str(side), level=float(cfg.cci_global_extreme_level_exec)
                    )

                    if bool(cfg.take_exec_and_ctx_cci_extreme):
                        if not bool(ctx_is_extreme):
                            status = "REJECT"
                            reason = "CCI_CONTEXT_NOT_EXTREME"
                        elif not bool(exec_is_extreme):
                            status = "REJECT"
                            reason = "CCI_EXEC_NOT_EXTREME"
                        else:
                            status = "ACCEPT"
                            reason = "OK_EXEC_AND_CTX_CCI_EXTREME"
                    elif bool(cfg.take_exec_cci_extreme_if_ctx_not_extreme):
                        if bool(ctx_is_extreme):
                            status = "REJECT"
                            reason = "CCI_CONTEXT_TOO_EXTREME"
                        elif not bool(exec_is_extreme):
                            status = "REJECT"
                            reason = "CCI_EXEC_NOT_EXTREME"
                        else:
                            status = "ACCEPT"
                            reason = "OK_EXEC_CCI_EXTREME_CTX_NOT_EXTREME"
                    else:
                        if bool(ctx_is_extreme):
                            status = "REJECT"
                            reason = "CCI_CONTEXT_TOO_EXTREME"
                        elif bool(exec_is_extreme):
                            status = "REJECT"
                            reason = "CCI_EXEC_TOO_EXTREME"
                        else:
                            status = "ACCEPT"
                            reason = "OK_NO_CCI_EXTREMES"

            ctx_force_mean_abs = None if ctx_m is None else float(ctx_m.force_mean_abs)
            ctx_force_peak_abs = None if ctx_m is None else float(ctx_m.force_peak_abs)

            if status == "ACCEPT":
                score = float(min(float(exec_m.force_mean_abs), float(ctx_force_mean_abs or 0.0)))
            else:
                score = 0.0

            return MacdMomentumTwoTFSignalMetrics(
                signal_kind=str(signal_kind),
                signal_ts=int(signal_ts),
                signal_dt=str(self._dt_from_ts(int(signal_ts))),
                exec_tranche_id=int(exec_m.tranche_id),
                exec_tranche_sign=str(exec_m.tranche_sign),
                exec_tranche_type=str(exec_m.tranche_type),
                exec_tranche_len=int(exec_m.tranche_len),
                exec_tranche_start_i=int(exec_m.tranche_start_i),
                exec_tranche_end_i=int(exec_m.tranche_end_i),
                exec_tranche_start_ts=int(exec_m.tranche_start_ts),
                exec_tranche_end_ts=int(exec_m.tranche_end_ts),
                exec_tranche_start_dt=str(exec_m.tranche_start_dt),
                exec_tranche_end_dt=str(exec_m.tranche_end_dt),
                exec_force_mean_abs=float(exec_m.force_mean_abs),
                exec_force_peak_abs=float(exec_m.force_peak_abs),
                ctx_tranche_id=(None if ctx_m is None else int(ctx_m.tranche_id)),
                ctx_tranche_sign=(None if ctx_m is None else str(ctx_m.tranche_sign)),
                ctx_tranche_type=(None if ctx_m is None else str(ctx_m.tranche_type)),
                ctx_tranche_len=(None if ctx_m is None else int(ctx_m.tranche_len)),
                ctx_tranche_start_ts=(None if ctx_m is None else int(ctx_m.tranche_start_ts)),
                ctx_tranche_end_ts=(None if ctx_m is None else int(ctx_m.tranche_end_ts)),
                ctx_force_mean_abs=ctx_force_mean_abs,
                ctx_force_peak_abs=ctx_force_peak_abs,
                cci_ctx_last_extreme=(None if cci_ctx_last is None else float(cci_ctx_last)),
                cci_exec_last_extreme=(None if cci_exec_last is None else float(cci_exec_last)),
                signal_i=int(signal_i),
                exec_i=exec_i2,
                entry_ts=entry_ts,
                entry_dt=(None if not entry_dt else str(entry_dt)),
                entry=entry,
                side=str(side),
                status=str(status),
                reason=str(reason),
                score=float(score),
                is_interesting=bool(status == "ACCEPT"),
            )

        out: list[MacdMomentumTwoTFSignalMetrics] = []

        for m in selected:
            if int(m.tranche_id) <= 0:
                continue
            signal_ts = int(m.tranche_start_ts)
            ctx_m = self._pick_ctx_for_ts(ctx_tranches, ts_ms=int(signal_ts))
            out.append(
                _build_metric(
                    signal_kind="EXEC_TRANCHE_START",
                    signal_ts=int(signal_ts),
                    exec_m=m,
                    ctx_m=ctx_m,
                    signal_i=int(m.tranche_start_i),
                )
            )

        if bool(cfg.signal_on_ctx_flip_if_exec_aligned):
            for idx, ctx_m in enumerate(ctx_tranches):
                if idx == 0:
                    continue
                ctx_flip_ts = int(ctx_m.tranche_start_ts)
                if int(selected_start_ts) > 0 and int(ctx_flip_ts) < int(selected_start_ts):
                    continue
                exec_m = self._pick_exec_for_ts(exec_tranches, ts_ms=int(ctx_flip_ts))
                if exec_m is None:
                    continue
                if int(exec_m.tranche_id) <= 0:
                    continue
                if str(exec_m.tranche_sign) != str(ctx_m.tranche_sign):
                    continue
                if int(ctx_flip_ts) <= int(exec_m.tranche_start_ts):
                    continue
                sig_i = _signal_i_for_ts(int(ctx_flip_ts))
                if sig_i is None:
                    continue
                out.append(
                    _build_metric(
                        signal_kind="CTX_TRANCHE_FLIP",
                        signal_ts=int(ctx_flip_ts),
                        exec_m=exec_m,
                        ctx_m=ctx_m,
                        signal_i=int(sig_i),
                    )
                )

        out.sort(key=lambda x: (int(x.signal_ts), int(x.exec_tranche_id), str(x.signal_kind)))
        return out

    def current_df(
        self,
        df: pd.DataFrame,
        *,
        df_ctx: pd.DataFrame,
    ) -> MacdMomentumTwoTFSignalMetrics | None:
        metrics = self.analyze_df(df, df_ctx=df_ctx, max_signals=1)
        if not metrics:
            return None
        return metrics[-1]

    def answer(
        self,
        *,
        question: dict[str, Any],
        df: pd.DataFrame,
        df_ctx: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        kind = str(question.get("kind") or "").strip().lower()
        max_signals = int(question.get("max_signals") or 0)

        if df_ctx is None:
            df_ctx = df

        if kind in {"", "analyze"}:
            metrics = self.analyze_df(df, df_ctx=df_ctx, max_signals=int(max_signals))
            return {
                "kind": "analyze",
                "max_signals": int(max_signals),
                "metrics": [asdict(m) for m in metrics],
            }

        if kind in {"current"}:
            m = self.current_df(df, df_ctx=df_ctx)
            return {
                "kind": "current",
                "metric": (None if m is None else asdict(m)),
            }

        raise ValueError(f"Unsupported question.kind: {question.get('kind')}")
