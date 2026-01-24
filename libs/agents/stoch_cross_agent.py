from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

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


def _cross_up(a0: float, a1: float, b0: float, b1: float) -> bool:
    return bool(float(a0) <= float(b0) and float(a1) > float(b1))


def _cross_down(a0: float, a1: float, b0: float, b1: float) -> bool:
    return bool(float(a0) >= float(b0) and float(a1) < float(b1))


@dataclass(frozen=True)
class StochCrossEvent:
    kind: str
    pos: int
    ts: int
    dt: str
    side: str
    meta: dict[str, object]


@dataclass(frozen=True)
class StochCrossAgentConfig:
    ts_col: str = "ts"

    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: str = "volume"

    stoch_k_col: str = "stoch_k"
    stoch_d_col: str = "stoch_d"

    min_k: float | None = None
    max_k: float | None = None

    impact_filter_enabled: bool = False
    impact_agg_len: int = 1
    impact_body_pct_min: float = 60.0
    impact_body_pct_max: float = 100.0
    impact_require_same_color: bool = True

    micro_filter_enabled: bool = False
    micro_slope_bars: int = 2
    micro_vwma_col: str = "vwma_4"
    micro_min_abs_slope: float = 0.0


class StochCrossAgent:
    def __init__(self, *, cfg: StochCrossAgentConfig | None = None):
        self.cfg = cfg or StochCrossAgentConfig()

    def _dt(self, ts: int) -> str:
        if int(ts) <= 0:
            return ""
        return str(pd.to_datetime(int(ts), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC"))

    def analyze_df(self, df: pd.DataFrame, *, max_events: int = 200) -> list[StochCrossEvent]:
        cfg = self.cfg
        for c in (cfg.ts_col, cfg.stoch_k_col, cfg.stoch_d_col):
            if str(c) not in df.columns:
                raise ValueError(f"Missing required column: {c}")

        ts_s = pd.to_numeric(df[str(cfg.ts_col)], errors="coerce").astype("Int64").to_numpy()
        k_s = pd.to_numeric(df[str(cfg.stoch_k_col)], errors="coerce").astype(float).to_numpy()
        d_s = pd.to_numeric(df[str(cfg.stoch_d_col)], errors="coerce").astype(float).to_numpy()

        out: list[StochCrossEvent] = []
        for i in range(1, int(len(df))):
            k0 = _safe_float(k_s[i - 1])
            k1 = _safe_float(k_s[i])
            d0 = _safe_float(d_s[i - 1])
            d1 = _safe_float(d_s[i])
            if k0 is None or k1 is None or d0 is None or d1 is None:
                continue

            if cfg.min_k is not None and float(k1) < float(cfg.min_k):
                continue
            if cfg.max_k is not None and float(k1) > float(cfg.max_k):
                continue

            side: str | None = None
            if _cross_up(float(k0), float(k1), float(d0), float(d1)):
                side = "LONG"
            elif _cross_down(float(k0), float(k1), float(d0), float(d1)):
                side = "SHORT"
            else:
                continue

            meta: dict[str, object] = {
                "k_prev": float(k0),
                "k": float(k1),
                "d_prev": float(d0),
                "d": float(d1),
            }

            if bool(cfg.impact_filter_enabled):
                for c in (cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col):
                    if str(c) not in df.columns:
                        raise ValueError(f"Missing required column for impact filter: {c}")

                target_color = "GREEN" if str(side) == "LONG" else "RED"
                z = compute_impact_bar(
                    df,
                    end_pos=int(i),
                    agg_len=int(cfg.impact_agg_len),
                    ts_col=str(cfg.ts_col),
                    open_col=str(cfg.open_col),
                    high_col=str(cfg.high_col),
                    low_col=str(cfg.low_col),
                    close_col=str(cfg.close_col),
                    target_color=str(target_color),
                    body_pct_min=float(cfg.impact_body_pct_min),
                    body_pct_max=float(cfg.impact_body_pct_max),
                    require_same_color=bool(cfg.impact_require_same_color),
                )
                if not bool(z.get("is_impact")):
                    continue
                meta["impact"] = dict(z)

            if bool(cfg.micro_filter_enabled):
                if str(cfg.micro_vwma_col) not in df.columns:
                    raise ValueError(f"Missing required column for micro filter: {cfg.micro_vwma_col}")
                z2 = compute_micro_direction(
                    df,
                    pos=int(i),
                    ts_col=str(cfg.ts_col),
                    slope_bars=int(cfg.micro_slope_bars),
                    vwma_col=str(cfg.micro_vwma_col),
                    min_abs_slope=float(cfg.micro_min_abs_slope),
                )
                if str(z2.get("side")) != str(side):
                    continue
                meta["micro"] = dict(z2)

            ts_i = int(ts_s[i]) if 0 <= int(i) < len(ts_s) and ts_s[i] is not None else 0
            out.append(
                StochCrossEvent(
                    kind="stoch_cross",
                    pos=int(i),
                    ts=int(ts_i),
                    dt=self._dt(int(ts_i)),
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
        max_events = int(question.get("max_events") or 200)

        if kind not in {"find_stoch_cross", "", "find"}:
            raise ValueError(f"Unsupported question.kind: {question.get('kind')}")

        events = self.analyze_df(df, max_events=int(max_events))
        return {
            "kind": "find_stoch_cross",
            "max_events": int(max_events),
            "events": [asdict(e) for e in events],
        }
