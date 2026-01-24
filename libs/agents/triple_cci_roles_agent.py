from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from libs.indicators.momentum.cci_tv import cci_tv


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


def _sign_series(x: np.ndarray, *, zero_policy: str) -> np.ndarray:
    n = int(x.size)
    out = np.full(n, 0, dtype=int)
    zp = str(zero_policy or "carry_prev_sign").strip().lower()
    if zp not in {"carry_prev_sign", "neutral_zero"}:
        raise ValueError(f"Unsupported zero_policy: {zero_policy}")

    prev = 0
    for i in range(n):
        v = _safe_float(x[i])
        if v is None:
            out[i] = 0
            prev = 0
            continue
        s = _sign(float(v))
        if s == 0 and zp == "carry_prev_sign":
            s = int(prev)
        out[i] = int(s)
        prev = int(s)

    return out


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


def _extreme_sign_arr(x: np.ndarray, *, extreme_level: float, zero_policy: str) -> np.ndarray:
    n = int(x.size)
    out = np.full(n, 0, dtype=int)
    lvl = float(extreme_level)
    if (not math.isfinite(lvl)) or lvl <= 0.0:
        return out

    prev = 0
    zp = str(zero_policy or "carry_prev_sign").strip().lower()
    if zp not in {"carry_prev_sign", "neutral_zero"}:
        raise ValueError(f"Unsupported extreme_zero_policy: {zero_policy}")

    for i in range(n):
        v = _safe_float(x[i])
        if v is None:
            out[i] = 0
            prev = 0
            continue
        s = 0
        if float(v) >= float(lvl):
            s = 1
        elif float(v) <= -float(lvl):
            s = -1
        else:
            s = 0
        if s == 0 and zp == "carry_prev_sign":
            s = int(prev)
        out[i] = int(s)
        prev = int(s)

    return out


def _dwell_arr(sign_arr: np.ndarray) -> np.ndarray:
    n = int(sign_arr.size)
    out = np.zeros(n, dtype=int)
    if n <= 0:
        return out

    prev_sign = 0
    prev_dwell = 0
    for i in range(n):
        s = int(sign_arr[i])
        if s == 0:
            out[i] = 0
            prev_sign = 0
            prev_dwell = 0
            continue
        if int(prev_sign) == int(s):
            prev_dwell += 1
        else:
            prev_dwell = 1
        out[i] = int(prev_dwell)
        prev_sign = int(s)

    return out


def _cross_up_0(x0: float, x1: float) -> bool:
    return bool(float(x0) <= 0.0 and float(x1) > 0.0)


def _cross_down_0(x0: float, x1: float) -> bool:
    return bool(float(x0) >= 0.0 and float(x1) < 0.0)


@dataclass(frozen=True)
class TripleCciLevelConfig:
    enabled: bool = True

    cci_col: str = ""
    cci_period: int = 0

    extreme_level: float = 200.0
    slope_window: int = 6

    trend_zero_policy: str = "carry_prev_sign"

    reject_trend_transition: bool = True

    allow_trade_when_respiration: bool = True
    allow_trade_when_extreme_in_macro_dir: bool = True

    force_mode: Literal["abs_slope", "abs_cci"] = "abs_slope"
    min_abs_force: float = 0.0
    require_force_rising: bool = True
    force_rising_bars: int = 2

    require_align_trend_to_macro: bool = True
    require_align_slope_to_macro: bool = False


@dataclass(frozen=True)
class TripleCciRolesEvent:
    kind: str
    pos: int
    ts: int
    dt: str
    side: str
    status: str
    reason: str
    meta: dict[str, object]


@dataclass(frozen=True)
class TripleCciRolesMetrics:
    pos: int
    ts: int
    dt: str

    macro_side: str
    macro_sign: int

    slow_trend_sign: int
    slow_slope_sign: int
    slow_is_respiration: bool
    slow_extreme_sign: int
    slow_extreme_dwell: int
    slow_force: float | None
    slow_force_rising: bool

    medium_trend_sign: int
    medium_slope_sign: int
    medium_is_respiration: bool
    medium_extreme_sign: int
    medium_extreme_dwell: int
    medium_force: float | None
    medium_force_rising: bool

    fast_trend_sign: int
    fast_slope_sign: int
    fast_is_respiration: bool
    fast_extreme_sign: int
    fast_extreme_dwell: int
    fast_force: float | None
    fast_force_rising: bool

    trigger_level: str
    trigger_long: bool
    trigger_short: bool


@dataclass(frozen=True)
class TripleCciRolesAgentConfig:
    ts_col: str = "ts"
    dt_col: str = "dt"

    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"

    slow: TripleCciLevelConfig = TripleCciLevelConfig(cci_col="cci_300", cci_period=300, extreme_level=200.0, slope_window=12)
    medium: TripleCciLevelConfig = TripleCciLevelConfig(cci_col="cci_120", cci_period=120, extreme_level=200.0, slope_window=8)
    fast: TripleCciLevelConfig = TripleCciLevelConfig(cci_col="cci_30", cci_period=30, extreme_level=200.0, slope_window=6)

    macro_mode: Literal["slow_sign", "slow_sign_and_slope"] = "slow_sign"

    style: Literal["", "scalp", "swing", "position"] = ""
    entry_trigger_level: Literal["", "slow", "medium", "fast"] = ""

    trigger_mode: Literal["zero_cross", "slope_cross", "extreme_exit", "any"] = "zero_cross"
    require_trigger_in_macro_dir: bool = True

    reject_when_all_three_respire: bool = False


class TripleCciRolesAgent:
    def __init__(self, *, cfg: TripleCciRolesAgentConfig | None = None):
        self.cfg = cfg or TripleCciRolesAgentConfig()

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

    def _maybe_compute_cci(self, df: pd.DataFrame, *, cci_col: str, cci_period: int) -> pd.Series | None:
        if str(cci_col) in df.columns:
            return pd.to_numeric(df[str(cci_col)], errors="coerce").astype(float)

        p = int(cci_period)
        if p <= 0:
            return None

        cfg = self.cfg
        for c in (cfg.high_col, cfg.low_col, cfg.close_col):
            if str(c) not in df.columns:
                raise ValueError(f"Missing required column for CCI computation: {c}")

        high = pd.to_numeric(df[str(cfg.high_col)], errors="coerce").astype(float).tolist()
        low = pd.to_numeric(df[str(cfg.low_col)], errors="coerce").astype(float).tolist()
        close = pd.to_numeric(df[str(cfg.close_col)], errors="coerce").astype(float).tolist()
        return pd.Series(cci_tv(high, low, close, int(p)), index=df.index, dtype=float)

    def _macro_sign_at(self, work: pd.DataFrame, *, i: int) -> int:
        cfg = self.cfg
        mode = str(cfg.macro_mode or "slow_sign").strip().lower()
        if mode not in {"slow_sign", "slow_sign_and_slope"}:
            raise ValueError(f"Unsupported macro_mode: {cfg.macro_mode}")

        s = int(work["cci_slow_trend_sign"].iloc[i])
        if mode == "slow_sign":
            return int(s)

        sl = int(work["cci_slow_slope_sign"].iloc[i])
        if s != 0 and sl != 0 and int(s) == int(sl):
            return int(s)
        return 0

    def _level_checks(
        self,
        work: pd.DataFrame,
        *,
        i: int,
        name: str,
        lv: TripleCciLevelConfig,
        macro_sign: int,
    ) -> tuple[bool, str]:
        if not bool(lv.enabled):
            return True, ""

        trend_sign = int(work[f"cci_{name}_trend_sign"].iloc[i])
        slope_sign = int(work[f"cci_{name}_slope_sign"].iloc[i])
        is_resp = bool(work[f"cci_{name}_is_respiration"].iloc[i])
        extreme_sign = int(work[f"cci_{name}_extreme_sign"].iloc[i])

        if bool(lv.reject_trend_transition) and int(trend_sign) == 0:
            return False, f"{name}:trend_transition"

        if bool(lv.require_align_trend_to_macro) and int(macro_sign) != 0:
            if int(trend_sign) != int(macro_sign):
                return False, f"{name}:trend_not_aligned"

        if bool(lv.require_align_slope_to_macro) and int(macro_sign) != 0:
            if int(slope_sign) != int(macro_sign):
                return False, f"{name}:slope_not_aligned"

        if (not bool(lv.allow_trade_when_respiration)) and bool(is_resp):
            return False, f"{name}:respiration_forbidden"

        if (not bool(lv.allow_trade_when_extreme_in_macro_dir)) and int(macro_sign) != 0:
            if int(extreme_sign) == int(macro_sign):
                return False, f"{name}:extreme_in_macro_dir_forbidden"

        f = _safe_float(work[f"cci_{name}_force"].iloc[i])
        if f is None or not math.isfinite(float(f)):
            return False, f"{name}:force_nan"

        if float(f) < float(lv.min_abs_force):
            return False, f"{name}:force_too_low"

        if bool(lv.require_force_rising):
            if not bool(work[f"cci_{name}_force_rising"].iloc[i]):
                return False, f"{name}:force_not_rising"

        return True, ""

    def enrich_df(self, df: pd.DataFrame, *, in_place: bool = False) -> pd.DataFrame:
        cfg = self.cfg

        if str(cfg.ts_col) not in df.columns:
            raise ValueError(f"Missing required column: {cfg.ts_col}")

        ts = pd.to_numeric(df[str(cfg.ts_col)], errors="coerce").astype("Int64").to_numpy()

        out = df if bool(in_place) else df.copy()

        def _compute(prefix: str, lv: TripleCciLevelConfig) -> dict[str, np.ndarray]:
            cci_s = self._maybe_compute_cci(out, cci_col=str(lv.cci_col), cci_period=int(lv.cci_period))
            if cci_s is None:
                raise ValueError(f"Missing required CCI column and cannot compute it: {lv.cci_col}")

            cci = pd.to_numeric(cci_s, errors="coerce").astype(float).to_numpy()

            trend_sign = _sign_series(cci, zero_policy=str(lv.trend_zero_policy))

            slope = _rolling_linreg_slope(cci, window=int(max(2, int(lv.slope_window))))
            slope_sign = _sign_series(slope, zero_policy="neutral_zero")

            extreme_sign = _extreme_sign_arr(cci, extreme_level=float(lv.extreme_level), zero_policy="neutral_zero")
            extreme_dwell = _dwell_arr(extreme_sign)

            is_resp = (trend_sign != 0) & (slope_sign == (-trend_sign))

            if str(lv.force_mode) == "abs_cci":
                force = np.abs(cci).astype(float)
            else:
                force = np.abs(slope).astype(float)
            force_rising = _rolling_force_rising(force, bars=int(max(2, int(lv.force_rising_bars))))

            cross_zero_up = np.full(int(len(out)), False, dtype=bool)
            cross_zero_down = np.full(int(len(out)), False, dtype=bool)
            cross_slope_up = np.full(int(len(out)), False, dtype=bool)
            cross_slope_down = np.full(int(len(out)), False, dtype=bool)

            lvl = float(lv.extreme_level)
            exit_low_extreme = np.full(int(len(out)), False, dtype=bool)
            exit_high_extreme = np.full(int(len(out)), False, dtype=bool)

            for i in range(1, int(len(out))):
                v0 = _safe_float(cci[i - 1])
                v1 = _safe_float(cci[i])
                if v0 is not None and v1 is not None:
                    if _cross_up_0(float(v0), float(v1)):
                        cross_zero_up[i] = True
                    elif _cross_down_0(float(v0), float(v1)):
                        cross_zero_down[i] = True

                    if math.isfinite(lvl) and lvl > 0.0:
                        if float(v0) <= -float(lvl) and float(v1) > -float(lvl):
                            exit_low_extreme[i] = True
                        if float(v0) >= float(lvl) and float(v1) < float(lvl):
                            exit_high_extreme[i] = True

                s0 = _safe_float(slope[i - 1])
                s1 = _safe_float(slope[i])
                if s0 is not None and s1 is not None:
                    if _cross_up_0(float(s0), float(s1)):
                        cross_slope_up[i] = True
                    elif _cross_down_0(float(s0), float(s1)):
                        cross_slope_down[i] = True

            return {
                f"{prefix}_trend_sign": trend_sign,
                f"{prefix}_slope": slope,
                f"{prefix}_slope_sign": slope_sign,
                f"{prefix}_extreme_sign": extreme_sign,
                f"{prefix}_extreme_dwell": extreme_dwell,
                f"{prefix}_is_respiration": is_resp.astype(bool),
                f"{prefix}_force": force,
                f"{prefix}_force_rising": force_rising,
                f"{prefix}_cross_zero_up": cross_zero_up,
                f"{prefix}_cross_zero_down": cross_zero_down,
                f"{prefix}_cross_slope_up": cross_slope_up,
                f"{prefix}_cross_slope_down": cross_slope_down,
                f"{prefix}_exit_low_extreme": exit_low_extreme,
                f"{prefix}_exit_high_extreme": exit_high_extreme,
            }

        slow = _compute("cci_slow", cfg.slow)
        medium = _compute("cci_medium", cfg.medium)
        fast = _compute("cci_fast", cfg.fast)

        for d in (slow, medium, fast):
            for k, v in d.items():
                out[str(k)] = v

        if str(cfg.dt_col) not in out.columns:
            try:
                out[str(cfg.dt_col)] = pd.to_datetime(ts, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                out[str(cfg.dt_col)] = ""

        return out

    def current_metrics(self, df: pd.DataFrame) -> TripleCciRolesMetrics | None:
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

        macro_sign = int(self._macro_sign_at(work, i=int(i)))
        macro_side = "LONG" if macro_sign > 0 else "SHORT" if macro_sign < 0 else ""

        trigger_level = str(self._entry_trigger_level())

        def _force(name: str) -> float | None:
            v = _safe_float(work[f"cci_{name}_force"].iloc[i])
            return None if v is None else float(v)

        def _rising(name: str) -> bool:
            return bool(work[f"cci_{name}_force_rising"].iloc[i])

        trig_long, trig_short = self._trigger_flags_at(work, i=int(i), trigger_level=str(trigger_level))

        return TripleCciRolesMetrics(
            pos=int(i),
            ts=int(ts_i),
            dt=str(dt),
            macro_side=str(macro_side),
            macro_sign=int(macro_sign),
            slow_trend_sign=int(work["cci_slow_trend_sign"].iloc[i]),
            slow_slope_sign=int(work["cci_slow_slope_sign"].iloc[i]),
            slow_is_respiration=bool(work["cci_slow_is_respiration"].iloc[i]),
            slow_extreme_sign=int(work["cci_slow_extreme_sign"].iloc[i]),
            slow_extreme_dwell=int(work["cci_slow_extreme_dwell"].iloc[i]),
            slow_force=_force("slow"),
            slow_force_rising=bool(_rising("slow")),
            medium_trend_sign=int(work["cci_medium_trend_sign"].iloc[i]),
            medium_slope_sign=int(work["cci_medium_slope_sign"].iloc[i]),
            medium_is_respiration=bool(work["cci_medium_is_respiration"].iloc[i]),
            medium_extreme_sign=int(work["cci_medium_extreme_sign"].iloc[i]),
            medium_extreme_dwell=int(work["cci_medium_extreme_dwell"].iloc[i]),
            medium_force=_force("medium"),
            medium_force_rising=bool(_rising("medium")),
            fast_trend_sign=int(work["cci_fast_trend_sign"].iloc[i]),
            fast_slope_sign=int(work["cci_fast_slope_sign"].iloc[i]),
            fast_is_respiration=bool(work["cci_fast_is_respiration"].iloc[i]),
            fast_extreme_sign=int(work["cci_fast_extreme_sign"].iloc[i]),
            fast_extreme_dwell=int(work["cci_fast_extreme_dwell"].iloc[i]),
            fast_force=_force("fast"),
            fast_force_rising=bool(_rising("fast")),
            trigger_level=str(trigger_level),
            trigger_long=bool(trig_long),
            trigger_short=bool(trig_short),
        )

    def _trigger_flags_at(self, work: pd.DataFrame, *, i: int, trigger_level: str) -> tuple[bool, bool]:
        cfg = self.cfg
        mode = str(cfg.trigger_mode or "zero_cross").strip().lower()
        if mode not in {"zero_cross", "slope_cross", "extreme_exit", "any"}:
            raise ValueError(f"Unsupported trigger_mode: {cfg.trigger_mode}")

        pfx = f"cci_{trigger_level}"

        z_up = bool(work[f"{pfx}_cross_zero_up"].iloc[i])
        z_dn = bool(work[f"{pfx}_cross_zero_down"].iloc[i])
        s_up = bool(work[f"{pfx}_cross_slope_up"].iloc[i])
        s_dn = bool(work[f"{pfx}_cross_slope_down"].iloc[i])
        ex_low = bool(work[f"{pfx}_exit_low_extreme"].iloc[i])
        ex_high = bool(work[f"{pfx}_exit_high_extreme"].iloc[i])

        if mode == "zero_cross":
            return bool(z_up), bool(z_dn)
        if mode == "slope_cross":
            return bool(s_up), bool(s_dn)
        if mode == "extreme_exit":
            return bool(ex_low), bool(ex_high)
        return bool(z_up or s_up or ex_low), bool(z_dn or s_dn or ex_high)

    def find_entry_events(self, df: pd.DataFrame, *, max_events: int = 200) -> list[TripleCciRolesEvent]:
        cfg = self.cfg
        work = self.enrich_df(df, in_place=False)
        if len(work) <= 0:
            return []

        trigger_level = str(self._entry_trigger_level())
        out: list[TripleCciRolesEvent] = []

        ts_arr = pd.to_numeric(work[str(cfg.ts_col)], errors="coerce").astype("Int64").to_numpy()

        for i in range(1, int(len(work))):
            macro_sign = int(self._macro_sign_at(work, i=int(i)))
            if int(macro_sign) == 0:
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

            if bool(cfg.reject_when_all_three_respire):
                if bool(work["cci_slow_is_respiration"].iloc[i]) and bool(work["cci_medium_is_respiration"].iloc[i]) and bool(
                    work["cci_fast_is_respiration"].iloc[i]
                ):
                    continue

            trig_long, trig_short = self._trigger_flags_at(work, i=int(i), trigger_level=str(trigger_level))

            if bool(cfg.require_trigger_in_macro_dir):
                if int(macro_sign) > 0 and (not bool(trig_long)):
                    continue
                if int(macro_sign) < 0 and (not bool(trig_short)):
                    continue
            else:
                if (not bool(trig_long)) and (not bool(trig_short)):
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
                TripleCciRolesEvent(
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
                        "trigger_mode": str(cfg.trigger_mode),
                        "trigger_long": bool(trig_long),
                        "trigger_short": bool(trig_short),
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
