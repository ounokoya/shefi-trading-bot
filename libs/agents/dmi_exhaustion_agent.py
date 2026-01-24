from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


def _safe_float(v: object) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return float(x)


@dataclass(frozen=True)
class DmiExhaustionEvent:
    kind: str
    pos: int
    ts: int
    dt: str
    side: str
    meta: dict[str, object]


@dataclass(frozen=True)
class DmiExhaustionAgentConfig:
    ts_col: str = "ts"

    adx_col: str = "adx"
    dx_col: str = "dx"
    plus_di_col: str = "plus_di"
    minus_di_col: str = "minus_di"

    maturity_mode: str = "di_max"  # di_max | adx_threshold
    adx_min_threshold: float = 20.0


class DmiExhaustionAgent:
    def __init__(self, *, cfg: DmiExhaustionAgentConfig | None = None):
        self.cfg = cfg or DmiExhaustionAgentConfig()

        mm = str(self.cfg.maturity_mode or "").strip().lower()
        if mm not in {"di_max", "adx_threshold"}:
            raise ValueError(
                f"DmiExhaustionAgent: invalid maturity_mode={self.cfg.maturity_mode!r} (expected 'di_max' or 'adx_threshold')"
            )

    def _dt(self, ts: int) -> str:
        if int(ts) <= 0:
            return ""
        return str(pd.to_datetime(int(ts), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC"))

    def _is_mature(self, *, adx: float, plus_di: float, minus_di: float) -> bool:
        mm = str(self.cfg.maturity_mode or "").strip().lower()
        if mm == "adx_threshold":
            return bool(float(adx) >= float(self.cfg.adx_min_threshold))
        return bool(float(adx) > max(float(plus_di), float(minus_di)))

    def analyze_df(self, df: pd.DataFrame, *, max_events: int = 200) -> list[DmiExhaustionEvent]:
        cfg = self.cfg

        for c in (cfg.ts_col, cfg.adx_col, cfg.plus_di_col, cfg.minus_di_col):
            if str(c) not in df.columns:
                raise ValueError(f"Missing required column: {c}")

        ts_s = pd.to_numeric(df[str(cfg.ts_col)], errors="coerce").astype("Int64").to_numpy()
        adx_s = pd.to_numeric(df[str(cfg.adx_col)], errors="coerce").astype(float).to_numpy()
        plus_di_s = pd.to_numeric(df[str(cfg.plus_di_col)], errors="coerce").astype(float).to_numpy()
        minus_di_s = pd.to_numeric(df[str(cfg.minus_di_col)], errors="coerce").astype(float).to_numpy()

        if str(cfg.dx_col) in df.columns:
            dx_s = pd.to_numeric(df[str(cfg.dx_col)], errors="coerce").astype(float).to_numpy()
        else:
            di_sum = plus_di_s + minus_di_s
            dx_raw = (plus_di_s - minus_di_s)
            dx_s = [math.nan] * int(len(di_sum))
            for i in range(int(len(di_sum))):
                denom = float(di_sum[i])
                num = float(dx_raw[i])
                if not (math.isfinite(denom) and math.isfinite(num)):
                    dx_s[i] = math.nan
                    continue
                if denom == 0.0:
                    dx_s[i] = 0.0
                    continue
                dx_s[i] = abs(num) / denom * 100.0

        out: list[DmiExhaustionEvent] = []

        for i in range(1, int(len(df))):
            adx0 = _safe_float(adx_s[i - 1])
            adx1 = _safe_float(adx_s[i])
            dx0 = _safe_float(dx_s[i - 1])
            dx1 = _safe_float(dx_s[i])
            pdi1 = _safe_float(plus_di_s[i])
            mdi1 = _safe_float(minus_di_s[i])

            if adx0 is None or adx1 is None or dx0 is None or dx1 is None or pdi1 is None or mdi1 is None:
                continue

            if not self._is_mature(adx=float(adx1), plus_di=float(pdi1), minus_di=float(mdi1)):
                continue

            dx_cross_under = bool(float(dx0) > float(adx0) and float(dx1) <= float(adx1))
            if not dx_cross_under:
                continue

            side = "LONG" if float(pdi1) > float(mdi1) else "SHORT"

            ts_i = int(ts_s[i]) if 0 <= int(i) < len(ts_s) and ts_s[i] is not None else 0
            out.append(
                DmiExhaustionEvent(
                    kind="dmi_dx_cross_under_adx",
                    pos=int(i),
                    ts=int(ts_i),
                    dt=self._dt(int(ts_i)),
                    side=str(side),
                    meta={
                        "adx_prev": float(adx0),
                        "adx": float(adx1),
                        "dx_prev": float(dx0),
                        "dx": float(dx1),
                        "plus_di": float(pdi1),
                        "minus_di": float(mdi1),
                        "maturity_mode": str(cfg.maturity_mode),
                        "adx_min_threshold": float(cfg.adx_min_threshold),
                    },
                )
            )

        out.sort(key=lambda x: (int(x.pos), int(x.ts)))
        if int(max_events) > 0:
            out = out[-int(max_events) :]
        return out

    def answer(self, *, question: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
        kind = str(question.get("kind") or "").strip().lower()
        max_events = int(question.get("max_events") or 200)

        if kind not in {"find_dmi_exhaustion", "find", ""}:
            raise ValueError(f"Unsupported question.kind: {question.get('kind')}")

        events = self.analyze_df(df, max_events=int(max_events))
        return {
            "kind": "find_dmi_exhaustion",
            "max_events": int(max_events),
            "events": [asdict(e) for e in events],
        }
