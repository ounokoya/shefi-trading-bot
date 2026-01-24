from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

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


def _sign(v: float | None) -> int:
    if v is None:
        return 0
    if not math.isfinite(float(v)):
        return 0
    if float(v) > 0.0:
        return 1
    if float(v) < 0.0:
        return -1
    return 0


def _bool(v: object) -> bool:
    try:
        return bool(v)
    except Exception:
        return False


def _rolling_force_rising(force: np.ndarray, *, bars: int) -> np.ndarray:
    n = int(force.size)
    out = np.full(n, False, dtype=bool)
    b = int(bars)
    if n <= 0 or b <= 1:
        return out

    for i in range(b - 1, n):
        ok = True
        for k in range(1, b):
            a0 = _safe_float(force[i - k])
            a1 = _safe_float(force[i - k + 1])
            if a0 is None or a1 is None:
                ok = False
                break
            if not (float(a1) > float(a0)):
                ok = False
                break
        out[i] = bool(ok)

    return out


def _hist_sign_series(hist: np.ndarray, *, zero_policy: str) -> np.ndarray:
    n = int(hist.size)
    out = np.full(n, 0, dtype=int)
    zp = str(zero_policy or "carry_prev_sign").strip().lower()
    if zp not in {"carry_prev_sign", "neutral_zero"}:
        raise ValueError(f"Unsupported hist_zero_policy: {zero_policy}")

    prev = 0
    for i in range(n):
        h = _safe_float(hist[i])
        if h is None:
            out[i] = 0
            prev = 0
            continue
        s = _sign(float(h))
        if s == 0 and zp == "carry_prev_sign":
            s = int(prev)
        out[i] = int(s)
        prev = int(s)

    return out


@dataclass(frozen=True)
class TripleMacdLevelConfig:
    enabled: bool = True

    line_col: str = ""
    signal_col: str = ""
    hist_col: str = ""

    reject_zone_transition: bool = True

    min_abs_force: float = 0.0
    require_force_rising: bool = True
    force_rising_bars: int = 2

    allow_trade_when_respiration: bool = True

    require_align_zone_to_macro: bool = True
    require_align_hist_to_macro: bool = False


@dataclass(frozen=True)
class TripleMacdRolesEvent:
    kind: str
    pos: int
    ts: int
    dt: str
    side: str
    status: str
    reason: str
    meta: dict[str, object]


@dataclass(frozen=True)
class TripleMacdRolesMetrics:
    pos: int
    ts: int
    dt: str

    macro_side: str
    macro_sign: int

    slow_zone_sign: int
    slow_hist_sign: int
    slow_is_respiration: bool
    slow_force: float | None
    slow_force_rising: bool

    medium_zone_sign: int
    medium_hist_sign: int
    medium_is_respiration: bool
    medium_force: float | None
    medium_force_rising: bool

    fast_zone_sign: int
    fast_hist_sign: int
    fast_is_respiration: bool
    fast_force: float | None
    fast_force_rising: bool

    trigger_level: str
    trigger_cross_up: bool
    trigger_cross_down: bool


@dataclass(frozen=True)
class TripleMacdRolesAgentConfig:
    ts_col: str = "ts"
    dt_col: str = "dt"
    close_col: str = "close"

    slow: TripleMacdLevelConfig = TripleMacdLevelConfig(
        line_col="macd_line_slow",
        signal_col="macd_signal_slow",
        hist_col="macd_hist_slow",
    )
    medium: TripleMacdLevelConfig = TripleMacdLevelConfig(
        line_col="macd_line_medium",
        signal_col="macd_signal_medium",
        hist_col="macd_hist_medium",
    )
    fast: TripleMacdLevelConfig = TripleMacdLevelConfig(
        line_col="macd_line_fast",
        signal_col="macd_signal_fast",
        hist_col="macd_hist_fast",
    )

    hist_zero_policy: str = "carry_prev_sign"

    macro_mode: Literal["slow_zone", "slow_hist", "slow_zone_and_hist"] = "slow_zone"

    entry_style: Literal["default", "simple_alignment"] = "default"
    require_hists_rising_on_entry: bool = True

    style: Literal["", "scalp", "swing", "position"] = ""
    entry_trigger_level: Literal["", "slow", "medium", "fast"] = ""
    require_trigger_in_macro_dir: bool = True

    reject_when_all_three_respire: bool = False


class TripleMacdRolesAgent:
    def __init__(self, *, cfg: TripleMacdRolesAgentConfig | None = None):
        self.cfg = cfg or TripleMacdRolesAgentConfig()

    def _dt(self, ts: int) -> str:
        if int(ts) <= 0:
            return ""
        return str(pd.to_datetime(int(ts), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC"))

    def _entry_trigger_level(self) -> str:
        cfg = self.cfg
        if str(cfg.entry_trigger_level).strip():
            return str(cfg.entry_trigger_level).strip().lower()
        style = str(cfg.style or "").strip().lower()
        if style == "position":
            return "slow"
        if style == "swing":
            return "medium"
        return "fast"

    def _zone_sign_arr(self, line: np.ndarray, signal: np.ndarray) -> np.ndarray:
        n = int(line.size)
        out = np.full(n, 0, dtype=int)
        for i in range(n):
            ml = _safe_float(line[i])
            ms = _safe_float(signal[i])
            if ml is None or ms is None:
                out[i] = 0
                continue
            sl = _sign(float(ml))
            ss = _sign(float(ms))
            if sl == 0 or ss == 0:
                out[i] = 0
                continue
            if sl == ss:
                out[i] = int(sl)
            else:
                out[i] = 0
        return out

    def _force_arr(self, hist: np.ndarray, close: np.ndarray) -> np.ndarray:
        n = int(hist.size)
        out = np.full(n, np.nan, dtype=float)
        for i in range(n):
            h = _safe_float(hist[i])
            c = _safe_float(close[i])
            if h is None or c is None:
                continue
            if float(c) == 0.0:
                continue
            out[i] = float(abs(float(h) / float(c)))
        return out

    def enrich_df(self, df: pd.DataFrame, *, in_place: bool = False) -> pd.DataFrame:
        cfg = self.cfg

        need = {str(cfg.ts_col), str(cfg.close_col)}
        for lv in (cfg.slow, cfg.medium, cfg.fast):
            need.add(str(lv.line_col))
            need.add(str(lv.signal_col))
            need.add(str(lv.hist_col))

        miss = sorted([c for c in need if c not in df.columns])
        if miss:
            raise ValueError(f"Missing required columns: {miss}")

        ts = pd.to_numeric(df[str(cfg.ts_col)], errors="coerce").astype("Int64").to_numpy()
        close = pd.to_numeric(df[str(cfg.close_col)], errors="coerce").astype(float).to_numpy()

        def _compute(prefix: str, lv: TripleMacdLevelConfig) -> dict[str, np.ndarray]:
            line = pd.to_numeric(df[str(lv.line_col)], errors="coerce").astype(float).to_numpy()
            sig = pd.to_numeric(df[str(lv.signal_col)], errors="coerce").astype(float).to_numpy()
            hist = pd.to_numeric(df[str(lv.hist_col)], errors="coerce").astype(float).to_numpy()

            zone_sign = self._zone_sign_arr(line, sig)
            hist_sign = _hist_sign_series(hist, zero_policy=str(cfg.hist_zero_policy))

            is_resp = (zone_sign != 0) & (hist_sign == (-zone_sign))
            is_impulse = (zone_sign != 0) & (hist_sign == zone_sign)

            cross_up = np.full(int(len(df)), False, dtype=bool)
            cross_down = np.full(int(len(df)), False, dtype=bool)
            for i in range(1, int(len(df))):
                h0 = _safe_float(hist[i - 1])
                h1 = _safe_float(hist[i])
                if h0 is None or h1 is None:
                    continue
                if float(h0) <= 0.0 and float(h1) > 0.0:
                    cross_up[i] = True
                elif float(h0) >= 0.0 and float(h1) < 0.0:
                    cross_down[i] = True

            force = self._force_arr(hist, close)
            force_rising = _rolling_force_rising(force, bars=int(max(2, int(lv.force_rising_bars))))

            return {
                f"{prefix}_zone_sign": zone_sign,
                f"{prefix}_hist_sign": hist_sign,
                f"{prefix}_is_respiration": is_resp.astype(bool),
                f"{prefix}_is_impulse": is_impulse.astype(bool),
                f"{prefix}_cross_up": cross_up,
                f"{prefix}_cross_down": cross_down,
                f"{prefix}_force": force,
                f"{prefix}_force_rising": force_rising,
            }

        slow = _compute("macd_slow", cfg.slow)
        medium = _compute("macd_medium", cfg.medium)
        fast = _compute("macd_fast", cfg.fast)

        out = df if bool(in_place) else df.copy()
        for d in (slow, medium, fast):
            for k, v in d.items():
                out[str(k)] = v

        if str(cfg.dt_col) not in out.columns:
            try:
                out[str(cfg.dt_col)] = pd.to_datetime(ts, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                out[str(cfg.dt_col)] = ""

        return out

    def _macro_sign_at(self, work: pd.DataFrame, *, i: int) -> int:
        cfg = self.cfg
        mode = str(cfg.macro_mode or "slow_zone").strip().lower()
        if mode not in {"slow_zone", "slow_hist", "slow_zone_and_hist"}:
            raise ValueError(f"Unsupported macro_mode: {cfg.macro_mode}")

        z = int(work["macd_slow_zone_sign"].iloc[i])
        h = int(work["macd_slow_hist_sign"].iloc[i])

        if mode == "slow_zone":
            return int(z)
        if mode == "slow_hist":
            return int(h)
        if z != 0 and h != 0 and int(z) == int(h):
            return int(z)
        return 0

    def _level_checks(
        self,
        work: pd.DataFrame,
        *,
        i: int,
        name: str,
        lv: TripleMacdLevelConfig,
        macro_sign: int,
    ) -> tuple[bool, str]:
        if not bool(lv.enabled):
            return True, ""

        zone_sign = int(work[f"macd_{name}_zone_sign"].iloc[i])
        hist_sign = int(work[f"macd_{name}_hist_sign"].iloc[i])
        is_resp = bool(work[f"macd_{name}_is_respiration"].iloc[i])

        if bool(lv.reject_zone_transition) and int(zone_sign) == 0:
            return False, f"{name}:zone_transition"

        if bool(lv.require_align_zone_to_macro) and int(macro_sign) != 0:
            if int(zone_sign) != int(macro_sign):
                return False, f"{name}:zone_not_aligned"

        if bool(lv.require_align_hist_to_macro) and int(macro_sign) != 0:
            if int(hist_sign) != int(macro_sign):
                return False, f"{name}:hist_not_aligned"

        if (not bool(lv.allow_trade_when_respiration)) and bool(is_resp):
            return False, f"{name}:respiration_forbidden"

        f = _safe_float(work[f"macd_{name}_force"].iloc[i])
        if f is None or not math.isfinite(float(f)):
            return False, f"{name}:force_nan"

        if float(f) < float(lv.min_abs_force):
            return False, f"{name}:force_too_low"

        if bool(lv.require_force_rising):
            if not bool(work[f"macd_{name}_force_rising"].iloc[i]):
                return False, f"{name}:force_not_rising"

        return True, ""

    def current_metrics(self, df: pd.DataFrame) -> TripleMacdRolesMetrics | None:
        cfg = self.cfg
        work = self.enrich_df(df, in_place=False)
        if len(work) <= 0:
            return None

        i = int(len(work) - 1)
        ts_s = pd.to_numeric(work[str(cfg.ts_col)], errors="coerce").astype("Int64")
        ts_i = int(ts_s.iloc[i]) if not ts_s.empty and ts_s.iloc[i] is not None else 0

        dt = ""
        if str(cfg.dt_col) in work.columns:
            try:
                dt = str(work[str(cfg.dt_col)].iloc[i])
            except Exception:
                dt = ""
        if not dt:
            dt = self._dt(int(ts_i))

        entry_style = str(cfg.entry_style or "default").strip().lower()
        if entry_style == "simple_alignment":
            slow_sign = int(work["macd_slow_hist_sign"].iloc[i])
            medium_sign = int(work["macd_medium_hist_sign"].iloc[i])
            macro_sign = int(slow_sign) if int(slow_sign) != 0 and int(medium_sign) == int(slow_sign) else 0
            trigger_level = "fast"
            fast_sign = int(work["macd_fast_hist_sign"].iloc[i])
            fast_sign_prev = int(work["macd_fast_hist_sign"].iloc[int(i - 1)]) if int(i) > 0 else 0

            slow_sign_prev = int(work["macd_slow_hist_sign"].iloc[int(i - 1)]) if int(i) > 0 else 0
            medium_sign_prev = int(work["macd_medium_hist_sign"].iloc[int(i - 1)]) if int(i) > 0 else 0
            slow_stable = bool(int(i) > 0 and int(slow_sign) == int(slow_sign_prev))
            medium_stable = bool(int(i) > 0 and int(medium_sign) == int(medium_sign_prev))

            changed = bool(int(i) > 0 and int(fast_sign) != int(fast_sign_prev) and int(fast_sign) != 0)
            aligned = bool(int(macro_sign) != 0 and int(fast_sign) == int(macro_sign))
            stable = bool(slow_stable and medium_stable)
            trig_up = bool(changed and aligned and stable and int(fast_sign_prev) <= 0 and int(fast_sign) > 0)
            trig_down = bool(changed and aligned and stable and int(fast_sign_prev) >= 0 and int(fast_sign) < 0)
        else:
            macro_sign = int(self._macro_sign_at(work, i=int(i)))
            trigger_level = str(self._entry_trigger_level())
            trig_up = bool(work[f"macd_{trigger_level}_cross_up"].iloc[i])
            trig_down = bool(work[f"macd_{trigger_level}_cross_down"].iloc[i])

        macro_side = "LONG" if macro_sign > 0 else "SHORT" if macro_sign < 0 else ""

        def _force(name: str) -> float | None:
            v = _safe_float(work[f"macd_{name}_force"].iloc[i])
            return None if v is None else float(v)

        def _rising(name: str) -> bool:
            return bool(work[f"macd_{name}_force_rising"].iloc[i])

        return TripleMacdRolesMetrics(
            pos=int(i),
            ts=int(ts_i),
            dt=str(dt),
            macro_side=str(macro_side),
            macro_sign=int(macro_sign),
            slow_zone_sign=int(work["macd_slow_zone_sign"].iloc[i]),
            slow_hist_sign=int(work["macd_slow_hist_sign"].iloc[i]),
            slow_is_respiration=bool(work["macd_slow_is_respiration"].iloc[i]),
            slow_force=_force("slow"),
            slow_force_rising=bool(_rising("slow")),
            medium_zone_sign=int(work["macd_medium_zone_sign"].iloc[i]),
            medium_hist_sign=int(work["macd_medium_hist_sign"].iloc[i]),
            medium_is_respiration=bool(work["macd_medium_is_respiration"].iloc[i]),
            medium_force=_force("medium"),
            medium_force_rising=bool(_rising("medium")),
            fast_zone_sign=int(work["macd_fast_zone_sign"].iloc[i]),
            fast_hist_sign=int(work["macd_fast_hist_sign"].iloc[i]),
            fast_is_respiration=bool(work["macd_fast_is_respiration"].iloc[i]),
            fast_force=_force("fast"),
            fast_force_rising=bool(_rising("fast")),
            trigger_level=str(trigger_level),
            trigger_cross_up=bool(trig_up),
            trigger_cross_down=bool(trig_down),
        )

    def find_entry_events(self, df: pd.DataFrame, *, max_events: int = 200) -> list[TripleMacdRolesEvent]:
        cfg = self.cfg
        work = self.enrich_df(df, in_place=False)
        if len(work) <= 0:
            return []

        entry_style = str(cfg.entry_style or "default").strip().lower()

        if entry_style == "simple_alignment":
            out: list[TripleMacdRolesEvent] = []
            ts_arr = pd.to_numeric(work[str(cfg.ts_col)], errors="coerce").astype("Int64").to_numpy()

            def _hists_rising(*, i: int, macro_sign: int) -> bool:
                if int(i) <= 0:
                    return False
                for lv in (cfg.slow, cfg.medium, cfg.fast):
                    col = str(lv.hist_col)
                    h0 = _safe_float(work[col].iloc[int(i - 1)])
                    h1 = _safe_float(work[col].iloc[int(i)])
                    if h0 is None or h1 is None:
                        return False
                    if float(macro_sign) * (float(h1) - float(h0)) <= 0.0:
                        return False
                return True

            for i in range(1, int(len(work))):
                slow_hist_prev = _safe_float(work[str(cfg.slow.hist_col)].iloc[int(i - 1)])
                slow_hist = _safe_float(work[str(cfg.slow.hist_col)].iloc[int(i)])
                medium_hist_prev = _safe_float(work[str(cfg.medium.hist_col)].iloc[int(i - 1)])
                medium_hist = _safe_float(work[str(cfg.medium.hist_col)].iloc[int(i)])
                fast_hist_prev = _safe_float(work[str(cfg.fast.hist_col)].iloc[int(i - 1)])
                fast_hist = _safe_float(work[str(cfg.fast.hist_col)].iloc[int(i)])
                if (
                    slow_hist_prev is None
                    or slow_hist is None
                    or medium_hist_prev is None
                    or medium_hist is None
                    or fast_hist_prev is None
                    or fast_hist is None
                ):
                    continue

                slow_sign = int(work["macd_slow_hist_sign"].iloc[i])
                medium_sign = int(work["macd_medium_hist_sign"].iloc[i])
                slow_sign_prev = int(work["macd_slow_hist_sign"].iloc[int(i - 1)])
                medium_sign_prev = int(work["macd_medium_hist_sign"].iloc[int(i - 1)])
                fast_sign_prev = int(work["macd_fast_hist_sign"].iloc[int(i - 1)])
                fast_sign = int(work["macd_fast_hist_sign"].iloc[i])

                if int(fast_sign) == 0:
                    continue
                if int(fast_sign) == int(fast_sign_prev):
                    continue
                if int(slow_sign) != int(fast_sign) or int(medium_sign) != int(fast_sign):
                    continue

                slow_stable = bool(int(slow_sign) == int(slow_sign_prev))
                medium_stable = bool(int(medium_sign) == int(medium_sign_prev))
                if (not bool(slow_stable)) or (not bool(medium_stable)):
                    continue

                macro_sign = int(fast_sign)
                trig_up = bool(int(fast_sign_prev) <= 0 and int(fast_sign) > 0)
                trig_down = bool(int(fast_sign_prev) >= 0 and int(fast_sign) < 0)

                if bool(cfg.require_hists_rising_on_entry) and (not _hists_rising(i=int(i), macro_sign=int(macro_sign))):
                    continue

                ok_slow, _ = self._level_checks(work, i=int(i), name="slow", lv=cfg.slow, macro_sign=int(macro_sign))
                if not ok_slow:
                    continue
                ok_med, _ = self._level_checks(work, i=int(i), name="medium", lv=cfg.medium, macro_sign=int(macro_sign))
                if not ok_med:
                    continue
                ok_fast, _ = self._level_checks(work, i=int(i), name="fast", lv=cfg.fast, macro_sign=int(macro_sign))
                if not ok_fast:
                    continue

                side = "LONG" if int(macro_sign) > 0 else "SHORT"
                ts_i = int(ts_arr[i]) if i < int(len(ts_arr)) and ts_arr[i] is not None else 0
                dt = ""
                if str(cfg.dt_col) in work.columns:
                    try:
                        dt = str(work[str(cfg.dt_col)].iloc[i])
                    except Exception:
                        dt = ""
                if not dt:
                    dt = self._dt(int(ts_i))

                out.append(
                    TripleMacdRolesEvent(
                        kind="entry",
                        pos=int(i),
                        ts=int(ts_i),
                        dt=str(dt),
                        side=str(side),
                        status="ACCEPT",
                        reason="",
                        meta={
                            "entry_style": str(entry_style),
                            "macro_sign": int(macro_sign),
                            "fast_hist_sign": int(fast_sign),
                            "fast_hist_sign_prev": int(fast_sign_prev),
                            "slow_hist_sign": int(slow_sign),
                            "medium_hist_sign": int(medium_sign),
                            "slow_hist_sign_prev": int(slow_sign_prev),
                            "medium_hist_sign_prev": int(medium_sign_prev),
                            "slow_stable": bool(slow_stable),
                            "medium_stable": bool(medium_stable),
                            "trigger_level": "fast",
                            "trigger_cross_up": bool(trig_up),
                            "trigger_cross_down": bool(trig_down),
                            "require_hists_rising_on_entry": bool(cfg.require_hists_rising_on_entry),
                        },
                    )
                )

            if int(max_events) > 0 and len(out) > int(max_events):
                out = out[-int(max_events) :]
            return out

        trigger_level = str(self._entry_trigger_level())
        out: list[TripleMacdRolesEvent] = []

        ts_arr = pd.to_numeric(work[str(cfg.ts_col)], errors="coerce").astype("Int64").to_numpy()

        for i in range(1, int(len(work))):
            macro_sign = int(self._macro_sign_at(work, i=int(i)))
            if int(macro_sign) == 0:
                continue

            ok_slow, reason = self._level_checks(work, i=int(i), name="slow", lv=cfg.slow, macro_sign=int(macro_sign))
            if not ok_slow:
                continue
            ok_med, reason = self._level_checks(work, i=int(i), name="medium", lv=cfg.medium, macro_sign=int(macro_sign))
            if not ok_med:
                continue
            ok_fast, reason = self._level_checks(work, i=int(i), name="fast", lv=cfg.fast, macro_sign=int(macro_sign))
            if not ok_fast:
                continue

            if bool(cfg.reject_when_all_three_respire):
                if bool(work["macd_slow_is_respiration"].iloc[i]) and bool(work["macd_medium_is_respiration"].iloc[i]) and bool(
                    work["macd_fast_is_respiration"].iloc[i]
                ):
                    continue

            trig_up = bool(work[f"macd_{trigger_level}_cross_up"].iloc[i])
            trig_down = bool(work[f"macd_{trigger_level}_cross_down"].iloc[i])

            if bool(cfg.require_trigger_in_macro_dir):
                if int(macro_sign) > 0 and (not bool(trig_up)):
                    continue
                if int(macro_sign) < 0 and (not bool(trig_down)):
                    continue
            else:
                if (not bool(trig_up)) and (not bool(trig_down)):
                    continue

            side = "LONG" if int(macro_sign) > 0 else "SHORT"
            ts_i = int(ts_arr[i]) if i < int(len(ts_arr)) and ts_arr[i] is not None else 0
            dt = ""
            if str(cfg.dt_col) in work.columns:
                try:
                    dt = str(work[str(cfg.dt_col)].iloc[i])
                except Exception:
                    dt = ""
            if not dt:
                dt = self._dt(int(ts_i))

            out.append(
                TripleMacdRolesEvent(
                    kind="entry",
                    pos=int(i),
                    ts=int(ts_i),
                    dt=str(dt),
                    side=str(side),
                    status="ACCEPT",
                    reason="",
                    meta={
                        "macro_mode": str(cfg.macro_mode),
                        "macro_sign": int(macro_sign),
                        "trigger_level": str(trigger_level),
                        "trigger_cross_up": bool(trig_up),
                        "trigger_cross_down": bool(trig_down),
                    },
                )
            )

        if int(max_events) > 0 and len(out) > int(max_events):
            out = out[-int(max_events) :]

        return out

    def answer(self, *, question: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
        kind = str(question.get("kind") or "").strip().lower()
        max_events = int(question.get("max_events") or 200)

        if kind in {"", "current", "current_metrics"}:
            m = self.current_metrics(df)
            return {"kind": "current", "metric": (None if m is None else asdict(m))}

        if kind in {"enrich", "enrich_df"}:
            work = self.enrich_df(df, in_place=False)
            return {"kind": "enrich", "df": work}

        if kind in {"entries", "find_entry_events"}:
            ev = self.find_entry_events(df, max_events=int(max_events))
            return {"kind": "entries", "max_events": int(max_events), "events": [asdict(e) for e in ev]}

        raise ValueError(f"Unsupported question.kind: {question.get('kind')}")
