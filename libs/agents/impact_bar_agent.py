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


def _candle_color(*, open_v: float, close_v: float) -> str | None:
    if not math.isfinite(float(open_v)) or not math.isfinite(float(close_v)):
        return None
    if float(close_v) > float(open_v):
        return "GREEN"
    if float(close_v) < float(open_v):
        return "RED"
    return None


def compute_impact_bar(
    df: pd.DataFrame,
    *,
    end_pos: int,
    agg_len: int,
    ts_col: str,
    open_col: str,
    high_col: str,
    low_col: str,
    close_col: str,
    target_color: str,
    body_pct_min: float,
    body_pct_max: float,
    require_same_color: bool,
) -> dict[str, object]:
    n = int(agg_len)
    if n < 1:
        return {"is_impact": False}
    i1 = int(end_pos)
    i0 = int(i1 - n + 1)
    if i0 < 0:
        return {"is_impact": False}

    w = df.iloc[int(i0) : int(i1) + 1]
    if len(w) != n:
        return {"is_impact": False}

    opens = pd.to_numeric(w[str(open_col)], errors="coerce").astype(float).to_numpy()
    highs = pd.to_numeric(w[str(high_col)], errors="coerce").astype(float).to_numpy()
    lows = pd.to_numeric(w[str(low_col)], errors="coerce").astype(float).to_numpy()
    closes = pd.to_numeric(w[str(close_col)], errors="coerce").astype(float).to_numpy()

    if (not opens.size) or (not highs.size) or (not lows.size) or (not closes.size):
        return {"is_impact": False}

    if not (math.isfinite(float(opens[0])) and math.isfinite(float(closes[-1]))):
        return {"is_impact": False}

    agg_open = float(opens[0])
    agg_close = float(closes[-1])

    h2 = highs[pd.notna(highs)]
    l2 = lows[pd.notna(lows)]
    if (not h2.size) or (not l2.size):
        return {"is_impact": False}

    agg_high = float(max(h2))
    agg_low = float(min(l2))

    denom = float(agg_high - agg_low)
    if denom <= 0 or (not math.isfinite(denom)):
        return {"is_impact": False}

    body = abs(float(agg_close) - float(agg_open))
    body_pct = float(100.0 * float(body) / float(denom))

    color = _candle_color(open_v=float(agg_open), close_v=float(agg_close))
    if color is None:
        return {"is_impact": False}

    target = str(target_color).upper().strip()
    if target not in {"GREEN", "RED", "ANY"}:
        raise ValueError(f"Unexpected target_color: {target_color}")

    if target != "ANY" and str(color) != target:
        return {"is_impact": False}

    if bool(require_same_color):
        for oi, ci in zip(opens.tolist(), closes.tolist()):
            if not (math.isfinite(float(oi)) and math.isfinite(float(ci))):
                return {"is_impact": False}
            ccol = _candle_color(open_v=float(oi), close_v=float(ci))
            if ccol is None:
                return {"is_impact": False}
            if target != "ANY" and str(ccol) != target:
                return {"is_impact": False}

    if not (math.isfinite(float(body_pct)) and float(body_pct) >= float(body_pct_min) and float(body_pct) <= float(body_pct_max)):
        return {"is_impact": False}

    ts = 0
    if str(ts_col) in df.columns:
        try:
            ts = int(pd.to_numeric(df[str(ts_col)].iloc[int(i1)], errors="coerce"))
        except Exception:
            ts = 0

    return {
        "is_impact": True,
        "agg_len": int(n),
        "start_pos": int(i0),
        "end_pos": int(i1),
        "ts": int(ts),
        "open": float(agg_open),
        "high": float(agg_high),
        "low": float(agg_low),
        "close": float(agg_close),
        "color": str(color),
        "body_pct": float(body_pct),
    }


@dataclass(frozen=True)
class ImpactBarMetrics:
    pos: int
    ts: int
    dt: str

    agg_len: int
    start_pos: int
    end_pos: int

    color: str
    body_pct: float

    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class ImpactBarAgentConfig:
    ts_col: str = "ts"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"

    agg_lens: tuple[int, ...] = (1,)
    target_color: str = "ANY"
    body_pct_min: float = 60.0
    body_pct_max: float = 100.0
    require_same_color: bool = True


class ImpactBarAgent:
    def __init__(self, *, cfg: ImpactBarAgentConfig | None = None):
        self.cfg = cfg or ImpactBarAgentConfig()

    def _dt(self, ts: int) -> str:
        if int(ts) <= 0:
            return ""
        return str(pd.to_datetime(int(ts), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC"))

    def analyze_df(self, df: pd.DataFrame, *, max_hits: int = 200) -> list[ImpactBarMetrics]:
        cfg = self.cfg
        for c in (cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col):
            if str(c) not in df.columns:
                raise ValueError(f"Missing required column: {c}")

        out: list[ImpactBarMetrics] = []
        for agg_len in cfg.agg_lens:
            n = int(agg_len)
            if n < 1:
                continue
            for i in range(int(n - 1), int(len(df))):
                z = compute_impact_bar(
                    df,
                    end_pos=int(i),
                    agg_len=int(n),
                    ts_col=str(cfg.ts_col),
                    open_col=str(cfg.open_col),
                    high_col=str(cfg.high_col),
                    low_col=str(cfg.low_col),
                    close_col=str(cfg.close_col),
                    target_color=str(cfg.target_color),
                    body_pct_min=float(cfg.body_pct_min),
                    body_pct_max=float(cfg.body_pct_max),
                    require_same_color=bool(cfg.require_same_color),
                )
                if not bool(z.get("is_impact")):
                    continue

                ts = int(z.get("ts") or 0)
                out.append(
                    ImpactBarMetrics(
                        pos=int(i),
                        ts=int(ts),
                        dt=self._dt(int(ts)),
                        agg_len=int(z.get("agg_len") or n),
                        start_pos=int(z.get("start_pos") or (int(i) - int(n) + 1)),
                        end_pos=int(z.get("end_pos") or int(i)),
                        color=str(z.get("color") or ""),
                        body_pct=float(z.get("body_pct") or 0.0),
                        open=float(z.get("open") or float("nan")),
                        high=float(z.get("high") or float("nan")),
                        low=float(z.get("low") or float("nan")),
                        close=float(z.get("close") or float("nan")),
                    )
                )

        out.sort(key=lambda x: (int(x.ts), int(x.agg_len), int(x.pos)))
        if int(max_hits) > 0:
            out = out[-int(max_hits) :]
        return out

    def answer(self, *, question: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
        kind = str(question.get("kind") or "").strip().lower()
        max_hits = int(question.get("max_hits") or 200)

        if kind not in {"find_impact_bars", "", "find"}:
            raise ValueError(f"Unsupported question.kind: {question.get('kind')}")

        hits = self.analyze_df(df, max_hits=int(max_hits))
        return {
            "kind": "find_impact_bars",
            "max_hits": int(max_hits),
            "hits": [asdict(x) for x in hits],
        }
