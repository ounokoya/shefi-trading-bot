from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from libs.blocks.segment_macd_hist_tranches_df import segment_macd_hist_tranches_df


def _safe_float(v: object) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return float(x)


@dataclass(frozen=True)
class VwmaBreakTouchTrancheMetrics:
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

    vwma_col: str
    zone_radius_pct: float

    vwma_start: float | None
    vwma_end: float | None
    vwma_delta_pct: float | None

    vwma_slope_last_pct: float | None
    vwma_slope_mean_pct: float | None

    vwma_pivot_kind: str | None
    vwma_pivot_pos: int | None
    vwma_pivot_ts: int | None
    vwma_pivot_val: float | None
    vwma_pivot_slope_before_pct: float | None
    vwma_pivot_slope_after_pct: float | None
    vwma_pivot_angle_deg: float | None
    vwma_pivot_move_pct_from_start: float | None
    price_move_pct_to_pivot: float | None

    touch_count: int
    last_touch_pos: int | None
    last_touch_ts: int | None
    last_touch_recency: int | None
    last_touch_dist_pct: float | None
    last_touch_side: str | None

    touch_reject_ok: bool | None
    touch_reject_pos: int | None
    touch_reject_ts: int | None
    touch_reject_move_pct: float | None

    score: float
    is_interesting: bool


@dataclass(frozen=True)
class VwmaBreakTouchTrancheAgentConfig:
    ts_col: str = "ts"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    hist_col: str = "macd_hist"

    vwma_col: str = "vwma_4"

    zone_radius_pct: float = 0.001

    pivot_slope_window: int = 2

    reject_lookahead: int = 6
    min_reject_move_pct: float = 0.002

    max_last_touch_recency: int = 6

    vwma_move_ref_pct: float = 0.002
    pivot_angle_ref_deg: float = 25.0

    min_pivot_angle_deg: float = 10.0

    min_tranche_len: int = 6
    min_score: float = 0.05


class VwmaBreakTouchTrancheAgent:
    def __init__(self, *, cfg: VwmaBreakTouchTrancheAgentConfig | None = None):
        self.cfg = cfg or VwmaBreakTouchTrancheAgentConfig()

    def _dt(self, ts: int) -> str:
        if int(ts) <= 0:
            return ""
        return str(pd.to_datetime(int(ts), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC"))

    def _slope_pct(self, a0: float, a1: float) -> float | None:
        if (not math.isfinite(float(a0))) or float(a0) == 0.0:
            return None
        if not math.isfinite(float(a1)):
            return None
        return float((float(a1) - float(a0)) / float(a0))

    def _pick_pivot(self, vwma: np.ndarray) -> tuple[str | None, int | None]:
        if vwma.size < 3:
            return None, None
        dv = np.diff(vwma)
        s = np.sign(dv)
        s = np.where(np.isfinite(s), s, 0.0)
        for i in range(1, len(s)):
            if s[i] == 0.0:
                s[i] = s[i - 1]
        if len(s) and s[0] == 0.0:
            nz = np.where(s != 0.0)[0]
            if nz.size:
                s[: int(nz[0]) + 1] = s[int(nz[0])]

        mins: list[int] = []
        maxs: list[int] = []
        for k in range(1, len(s)):
            if (s[k - 1] < 0.0) and (s[k] > 0.0):
                mins.append(int(k))
            elif (s[k - 1] > 0.0) and (s[k] < 0.0):
                maxs.append(int(k))

        last_min = int(mins[-1]) if mins else None
        last_max = int(maxs[-1]) if maxs else None
        if last_min is None and last_max is None:
            return None, None
        if last_max is None:
            return "MIN", int(last_min)
        if last_min is None:
            return "MAX", int(last_max)
        if int(last_min) > int(last_max):
            return "MIN", int(last_min)
        return "MAX", int(last_max)

    def _touch_reject(
        self,
        *,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        vwma: np.ndarray,
        start_pos: int,
        dir_sign: float,
        zone_radius_pct: float,
    ) -> tuple[bool | None, int | None, float | None]:
        if start_pos < 0 or start_pos >= int(len(close)):
            return None, None, None
        k = int(self.cfg.reject_lookahead)
        if k < 1:
            k = 1
        end = min(int(len(close)) - 1, int(start_pos) + int(k))
        c0 = _safe_float(close[int(start_pos)])
        if c0 is None or c0 <= 0:
            return None, None, None

        for j in range(int(start_pos) + 1, int(end) + 1):
            cj = _safe_float(close[int(j)])
            vj = _safe_float(vwma[int(j)])
            if cj is None or vj is None or vj <= 0:
                continue
            upper = vj * (1.0 + float(zone_radius_pct))
            lower = vj * (1.0 - float(zone_radius_pct))

            ok_side = bool(cj > upper) if float(dir_sign) > 0.0 else bool(cj < lower)
            if not ok_side:
                continue
            move = float(dir_sign) * (float(cj) - float(c0)) / float(c0)
            if move >= float(self.cfg.min_reject_move_pct):
                return True, int(j), float(move)

        return False, None, None

    def _to_metrics(self, tranche: pd.DataFrame) -> VwmaBreakTouchTrancheMetrics | None:
        cfg = self.cfg
        if len(tranche) == 0:
            return None

        for c in (cfg.ts_col, cfg.close_col, cfg.hist_col, cfg.vwma_col):
            if str(c) not in tranche.columns:
                raise ValueError(f"Missing required column: {c}")

        if cfg.high_col not in tranche.columns or cfg.low_col not in tranche.columns:
            raise ValueError(f"Missing required columns: {cfg.high_col}, {cfg.low_col}")

        tid0 = tranche.get("tranche_id")
        if tid0 is None:
            return None
        try:
            tranche_id = int(pd.to_numeric(tid0.iloc[0], errors="coerce"))
        except Exception:
            return None

        sign = str(tranche.get("tranche_sign").iloc[0])
        tranche_type = "haussier" if sign == "+" else "baissier"
        dir_sign = 1.0 if sign == "+" else -1.0

        tranche_start_i = int(min(tranche.index))
        tranche_end_i = int(max(tranche.index))

        ts_s = pd.to_numeric(tranche[cfg.ts_col], errors="coerce").astype("Int64")
        tranche_start_ts = int(ts_s.iloc[0])
        tranche_end_ts = int(ts_s.iloc[-1])

        tranche_start_dt = self._dt(int(tranche_start_ts))
        tranche_end_dt = self._dt(int(tranche_end_ts))

        close = pd.to_numeric(tranche[cfg.close_col], errors="coerce").astype(float).to_numpy()
        high = pd.to_numeric(tranche[cfg.high_col], errors="coerce").astype(float).to_numpy()
        low = pd.to_numeric(tranche[cfg.low_col], errors="coerce").astype(float).to_numpy()
        vwma = pd.to_numeric(tranche[cfg.vwma_col], errors="coerce").astype(float).to_numpy()

        zone_radius = float(cfg.zone_radius_pct)
        if zone_radius < 0:
            zone_radius = 0.0

        vwma_start = _safe_float(vwma[0]) if vwma.size else None
        vwma_end = _safe_float(vwma[-1]) if vwma.size else None

        vwma_delta_pct = None
        if vwma_start is not None and vwma_start > 0 and vwma_end is not None:
            vwma_delta_pct = float(dir_sign) * (float(vwma_end) - float(vwma_start)) / float(vwma_start)

        vwma_slope_last_pct = None
        if vwma.size >= 2:
            a0 = _safe_float(vwma[-2])
            a1 = _safe_float(vwma[-1])
            if a0 is not None and a1 is not None:
                vwma_slope_last_pct = self._slope_pct(float(a0), float(a1))

        vwma_slope_mean_pct = None
        if vwma.size >= 3:
            prev = vwma[:-1]
            nxt = vwma[1:]
            denom = np.where(np.isfinite(prev) & (prev != 0.0), prev, np.nan)
            d = (nxt - prev) / denom
            d = d[np.isfinite(d)]
            if d.size:
                vwma_slope_mean_pct = float(np.mean(d))

        pivot_kind, pivot_pos = self._pick_pivot(vwma)
        pivot_ts = None
        pivot_val = None

        pivot_slope_before = None
        pivot_slope_after = None
        pivot_angle = None
        vwma_pivot_move_pct_from_start = None
        price_move_pct_to_pivot = None

        if pivot_pos is not None and 0 <= int(pivot_pos) < int(len(tranche)):
            try:
                pivot_ts = int(pd.to_numeric(tranche[cfg.ts_col], errors="coerce").astype("Int64").iloc[int(pivot_pos)])
            except Exception:
                pivot_ts = None
            pivot_val = _safe_float(vwma[int(pivot_pos)])

            w = int(cfg.pivot_slope_window)
            if w < 1:
                w = 1

            before = []
            for i in range(max(1, int(pivot_pos) - w + 1), int(pivot_pos) + 1):
                a0 = _safe_float(vwma[int(i) - 1])
                a1 = _safe_float(vwma[int(i)])
                if a0 is None or a1 is None:
                    continue
                sp = self._slope_pct(float(a0), float(a1))
                if sp is not None:
                    before.append(float(sp))
            if before:
                pivot_slope_before = float(np.mean(before))

            after = []
            for i in range(int(pivot_pos) + 1, min(int(len(vwma)), int(pivot_pos) + 1 + w)):
                a0 = _safe_float(vwma[int(i) - 1])
                a1 = _safe_float(vwma[int(i)])
                if a0 is None or a1 is None:
                    continue
                sp = self._slope_pct(float(a0), float(a1))
                if sp is not None:
                    after.append(float(sp))
            if after:
                pivot_slope_after = float(np.mean(after))

            if pivot_slope_before is not None and pivot_slope_after is not None:
                zr = float(zone_radius)
                if zr <= 0.0:
                    zr = 1.0
                th1 = math.atan(float(pivot_slope_before) / float(zr))
                th2 = math.atan(float(pivot_slope_after) / float(zr))
                pivot_angle = float(abs(th2 - th1) * 180.0 / math.pi)

            if vwma_start is not None and vwma_start > 0 and pivot_val is not None:
                vwma_pivot_move_pct_from_start = float(dir_sign) * (float(pivot_val) - float(vwma_start)) / float(vwma_start)

            c0 = _safe_float(close[0])
            cp = _safe_float(close[int(pivot_pos)])
            if c0 is not None and c0 > 0 and cp is not None:
                price_move_pct_to_pivot = float(dir_sign) * (float(cp) - float(c0)) / float(c0)

        touch = None
        if len(tranche) and vwma.size and high.size and low.size:
            v = vwma
            upper = v * (1.0 + float(zone_radius))
            lower = v * (1.0 - float(zone_radius))
            touch = (high >= lower) & (low <= upper)
            touch = np.where(np.isfinite(touch), touch, False).astype(bool)
        if touch is None:
            touch = np.zeros(int(len(tranche)), dtype=bool)

        touch_idx = np.where(touch)[0]
        touch_count = int(len(touch_idx))

        last_touch_pos = int(touch_idx[-1]) if touch_idx.size else None
        last_touch_ts = None
        last_touch_recency = None
        last_touch_dist_pct = None
        last_touch_side = None

        if last_touch_pos is not None:
            try:
                last_touch_ts = int(pd.to_numeric(tranche[cfg.ts_col], errors="coerce").astype("Int64").iloc[int(last_touch_pos)])
            except Exception:
                last_touch_ts = None

            last_touch_recency = int(len(tranche) - 1 - int(last_touch_pos))

            ct = _safe_float(close[int(last_touch_pos)])
            vt = _safe_float(vwma[int(last_touch_pos)])
            if ct is not None and vt is not None and vt > 0:
                last_touch_dist_pct = float(abs(float(ct) - float(vt)) / float(vt))

            if int(last_touch_pos) > 0:
                c_prev = _safe_float(close[int(last_touch_pos) - 1])
                v_prev = _safe_float(vwma[int(last_touch_pos) - 1])
                if c_prev is not None and v_prev is not None:
                    last_touch_side = "from_above" if float(c_prev) > float(v_prev) else "from_below"

        max_rec = int(cfg.max_last_touch_recency)
        if max_rec < 0:
            max_rec = 0
        last_touch_recent_ok = bool(last_touch_recency is not None and int(last_touch_recency) <= int(max_rec))

        touch_reject_ok = None
        touch_reject_pos = None
        touch_reject_ts = None
        touch_reject_move_pct = None
        if last_touch_pos is not None:
            ok, pos, move = self._touch_reject(
                high=high,
                low=low,
                close=close,
                vwma=vwma,
                start_pos=int(last_touch_pos),
                dir_sign=float(dir_sign),
                zone_radius_pct=float(zone_radius),
            )
            touch_reject_ok = ok
            touch_reject_pos = pos
            touch_reject_move_pct = move
            if pos is not None:
                try:
                    touch_reject_ts = int(pd.to_numeric(tranche[cfg.ts_col], errors="coerce").astype("Int64").iloc[int(pos)])
                except Exception:
                    touch_reject_ts = None

        angle_ref = float(cfg.pivot_angle_ref_deg)
        if angle_ref <= 0:
            angle_ref = 25.0
        angle_factor = float(min(1.0, (float(pivot_angle) / angle_ref))) if pivot_angle is not None else 0.0

        move_ref = float(cfg.vwma_move_ref_pct)
        if move_ref <= 0:
            move_ref = 0.002
        vwma_move_pct = 0.0
        if vwma_pivot_move_pct_from_start is not None and math.isfinite(float(vwma_pivot_move_pct_from_start)):
            vwma_move_pct = float(max(vwma_move_pct, float(vwma_pivot_move_pct_from_start)))
        if vwma_delta_pct is not None and math.isfinite(float(vwma_delta_pct)):
            vwma_move_pct = float(max(vwma_move_pct, float(vwma_delta_pct)))
        vwma_move_factor = float(min(1.0, float(max(0.0, float(vwma_move_pct))) / move_ref))

        touch_factor = 1.0 if bool(touch_reject_ok) else 0.0
        touch_recency_factor = (
            float(1.0 - float(min(int(last_touch_recency), int(len(tranche) - 1))) / float(max(1, int(len(tranche) - 1))))
            if last_touch_recency is not None
            else 0.0
        )

        score = float(0.10 + 0.90 * touch_factor)
        score *= float(0.25 + 0.75 * touch_recency_factor)
        score *= float(0.25 + 0.75 * angle_factor)
        score *= float(0.25 + 0.75 * vwma_move_factor)

        min_pivot_angle = float(cfg.min_pivot_angle_deg)
        if min_pivot_angle < 0.0:
            min_pivot_angle = 0.0

        is_interesting = bool(
            int(len(tranche)) >= int(cfg.min_tranche_len)
            and float(score) >= float(cfg.min_score)
            and bool(last_touch_recent_ok)
            and (bool(touch_reject_ok) or (pivot_angle is not None and float(pivot_angle) >= float(min_pivot_angle)))
        )

        return VwmaBreakTouchTrancheMetrics(
            tranche_id=int(tranche_id),
            tranche_sign=str(sign),
            tranche_type=str(tranche_type),
            tranche_len=int(len(tranche)),
            tranche_start_i=int(tranche_start_i),
            tranche_end_i=int(tranche_end_i),
            tranche_start_ts=int(tranche_start_ts),
            tranche_end_ts=int(tranche_end_ts),
            tranche_start_dt=str(tranche_start_dt),
            tranche_end_dt=str(tranche_end_dt),
            vwma_col=str(cfg.vwma_col),
            zone_radius_pct=float(zone_radius),
            vwma_start=(None if vwma_start is None else float(vwma_start)),
            vwma_end=(None if vwma_end is None else float(vwma_end)),
            vwma_delta_pct=(None if vwma_delta_pct is None else float(vwma_delta_pct)),
            vwma_slope_last_pct=(None if vwma_slope_last_pct is None else float(vwma_slope_last_pct)),
            vwma_slope_mean_pct=(None if vwma_slope_mean_pct is None else float(vwma_slope_mean_pct)),
            vwma_pivot_kind=(None if pivot_kind is None else str(pivot_kind)),
            vwma_pivot_pos=(None if pivot_pos is None else int(pivot_pos)),
            vwma_pivot_ts=(None if pivot_ts is None else int(pivot_ts)),
            vwma_pivot_val=(None if pivot_val is None else float(pivot_val)),
            vwma_pivot_slope_before_pct=(None if pivot_slope_before is None else float(pivot_slope_before)),
            vwma_pivot_slope_after_pct=(None if pivot_slope_after is None else float(pivot_slope_after)),
            vwma_pivot_angle_deg=(None if pivot_angle is None else float(pivot_angle)),
            vwma_pivot_move_pct_from_start=(
                None if vwma_pivot_move_pct_from_start is None else float(vwma_pivot_move_pct_from_start)
            ),
            price_move_pct_to_pivot=(None if price_move_pct_to_pivot is None else float(price_move_pct_to_pivot)),
            touch_count=int(touch_count),
            last_touch_pos=(None if last_touch_pos is None else int(last_touch_pos)),
            last_touch_ts=(None if last_touch_ts is None else int(last_touch_ts)),
            last_touch_recency=(None if last_touch_recency is None else int(last_touch_recency)),
            last_touch_dist_pct=(None if last_touch_dist_pct is None else float(last_touch_dist_pct)),
            last_touch_side=(None if last_touch_side is None else str(last_touch_side)),
            touch_reject_ok=(None if touch_reject_ok is None else bool(touch_reject_ok)),
            touch_reject_pos=(None if touch_reject_pos is None else int(touch_reject_pos)),
            touch_reject_ts=(None if touch_reject_ts is None else int(touch_reject_ts)),
            touch_reject_move_pct=(None if touch_reject_move_pct is None else float(touch_reject_move_pct)),
            score=float(score),
            is_interesting=bool(is_interesting),
        )

    def analyze_df(self, df: pd.DataFrame, *, max_tranches: int = 0) -> list[VwmaBreakTouchTrancheMetrics]:
        cfg = self.cfg
        work = segment_macd_hist_tranches_df(
            df,
            ts_col=str(cfg.ts_col),
            high_col=str(cfg.high_col),
            low_col=str(cfg.low_col),
            close_col=str(cfg.close_col),
            hist_col=str(cfg.hist_col),
            extremes_on="high_low",
        )
        if "tranche_id" not in work.columns:
            return []

        tids = pd.to_numeric(work["tranche_id"], errors="coerce").dropna().astype(int).tolist()
        if not tids:
            return []

        uniq = sorted(set(tids))
        selected = uniq[-int(max_tranches) :] if int(max_tranches) > 0 else uniq

        out: list[VwmaBreakTouchTrancheMetrics] = []
        for tid in selected:
            tdf = work.loc[work["tranche_id"].astype("Int64") == int(tid)]
            m = self._to_metrics(tdf)
            if m is not None:
                out.append(m)

        out.sort(key=lambda x: int(x.tranche_id))
        return out

    def current_df(self, df: pd.DataFrame) -> VwmaBreakTouchTrancheMetrics | None:
        cfg = self.cfg
        work = segment_macd_hist_tranches_df(
            df,
            ts_col=str(cfg.ts_col),
            high_col=str(cfg.high_col),
            low_col=str(cfg.low_col),
            close_col=str(cfg.close_col),
            hist_col=str(cfg.hist_col),
            extremes_on="high_low",
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
        top_n = int(question.get("top_n") or 10)
        max_tranches = int(question.get("max_tranches") or 0)

        if kind in {"", "analyze", "analyze_vwma_break_touch_tranches"}:
            metrics = self.analyze_df(df, max_tranches=int(max_tranches))
            return {
                "kind": "analyze",
                "max_tranches": int(max_tranches),
                "metrics": [asdict(m) for m in metrics],
            }

        if kind in {"current", "current_vwma_break_touch_tranche"}:
            m = self.current_df(df)
            return {
                "kind": "current",
                "metric": (None if m is None else asdict(m)),
            }

        if kind in {"rank_vwma_break_touch_tranches", "rank"}:
            metrics = self.analyze_df(df, max_tranches=int(max_tranches))
            metrics_sorted = sorted(metrics, key=lambda x: (float(x.score), int(x.tranche_end_ts)), reverse=True)
            ranked = metrics_sorted[: int(top_n)]
            interesting = [m for m in metrics_sorted if bool(m.is_interesting)][: int(top_n)]
            return {
                "kind": "rank_vwma_break_touch_tranches",
                "top_n": int(top_n),
                "max_tranches": int(max_tranches),
                "ranked": [asdict(m) for m in ranked],
                "interesting": [asdict(m) for m in interesting],
            }

        raise ValueError(f"Unsupported question.kind: {question.get('kind')}")
