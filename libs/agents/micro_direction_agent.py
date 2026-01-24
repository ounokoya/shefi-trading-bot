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


def compute_micro_direction(
    df: pd.DataFrame,
    *,
    pos: int,
    ts_col: str,
    slope_bars: int,
    vwma_col: str,
    min_abs_slope: float,
) -> dict[str, object]:
    i = int(pos)
    if i < 0 or i >= int(len(df)):
        return {"side": None}

    sb = int(slope_bars)
    if sb < 1:
        sb = 1
    if str(vwma_col) not in df.columns:
        return {"side": None}

    vw = pd.to_numeric(df[str(vwma_col)], errors="coerce").astype(float).to_numpy()
    vi = _safe_float(vw[i])
    if vi is None:
        return {"side": None}

    is0 = int(i - sb)
    if is0 < 0:
        return {"side": None}
    v0 = _safe_float(vw[is0])
    if v0 is None:
        return {"side": None}

    slope = float(float(vi) - float(v0))
    if float(abs(float(slope))) < float(min_abs_slope):
        return {"side": None}

    side: str | None = None
    if float(slope) > 0.0:
        side = "LONG"
    elif float(slope) < 0.0:
        side = "SHORT"
    else:
        side = None

    if side is None:
        return {"side": None}

    ts = 0
    if str(ts_col) in df.columns:
        try:
            ts = int(pd.to_numeric(df[str(ts_col)].iloc[int(i)], errors="coerce"))
        except Exception:
            ts = 0

    return {
        "side": str(side),
        "pos": int(i),
        "ts": int(ts),
        "slope": float(slope),
    }


@dataclass(frozen=True)
class MicroDirectionMetrics:
    side: str
    pos: int
    ts: int
    dt: str

    slope: float


@dataclass(frozen=True)
class MicroDirectionAgentConfig:
    ts_col: str = "ts"

    vwma_col: str = "vwma_4"
    slope_bars: int = 2
    min_abs_slope: float = 0.0


class MicroDirectionAgent:
    def __init__(self, *, cfg: MicroDirectionAgentConfig | None = None):
        self.cfg = cfg or MicroDirectionAgentConfig()

    def _dt(self, ts: int) -> str:
        if int(ts) <= 0:
            return ""
        return str(pd.to_datetime(int(ts), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC"))

    def analyze_df(self, df: pd.DataFrame, *, max_hits: int = 200) -> list[MicroDirectionMetrics]:
        cfg = self.cfg
        out: list[MicroDirectionMetrics] = []

        for i in range(int(len(df))):
            z = compute_micro_direction(
                df,
                pos=int(i),
                ts_col=str(cfg.ts_col),
                slope_bars=int(cfg.slope_bars),
                vwma_col=str(cfg.vwma_col),
                min_abs_slope=float(cfg.min_abs_slope),
            )
            side = z.get("side")
            if side not in {"LONG", "SHORT"}:
                continue

            ts = int(z.get("ts") or 0)
            out.append(
                MicroDirectionMetrics(
                    side=str(side),
                    pos=int(z.get("pos") or i),
                    ts=int(ts),
                    dt=self._dt(int(ts)),
                    slope=float(z.get("slope") or 0.0),
                )
            )

        out.sort(key=lambda x: (int(x.ts), abs(float(x.slope))), reverse=True)
        if int(max_hits) > 0:
            out = out[: int(max_hits)]
        out.sort(key=lambda x: (int(x.ts), int(x.pos)))
        return out

    def answer(self, *, question: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
        kind = str(question.get("kind") or "").strip().lower()
        max_hits = int(question.get("max_hits") or 200)

        if kind not in {"find_micro_directions", "", "find"}:
            raise ValueError(f"Unsupported question.kind: {question.get('kind')}")

        hits = self.analyze_df(df, max_hits=int(max_hits))
        return {
            "kind": "find_micro_directions",
            "max_hits": int(max_hits),
            "hits": [asdict(x) for x in hits],
        }
