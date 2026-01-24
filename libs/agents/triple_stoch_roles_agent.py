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


def _cross_up(a0: float, a1: float, b0: float, b1: float) -> bool:
    return bool(float(a0) <= float(b0) and float(a1) > float(b1))


def _cross_down(a0: float, a1: float, b0: float, b1: float) -> bool:
    return bool(float(a0) >= float(b0) and float(a1) < float(b1))


@dataclass(frozen=True)
class TripleStochLevelConfig:
    enabled: bool = True

    k_col: str = ""
    d_col: str = ""

    k_period: int = 14
    d_period: int = 3

    regime_pivot: float = 50.0
    extreme_high: float = 80.0
    extreme_low: float = 20.0

    reject_regime_transition: bool = True

    allow_trade_when_respiration: bool = True
    allow_trade_when_extreme_in_macro_dir: bool = True

    slope_window: int = 6

    force_mode: Literal["abs_spread", "abs_k_slope"] = "abs_spread"
    min_abs_force: float = 0.0
    require_force_rising: bool = True
    force_rising_bars: int = 2

    require_align_regime_to_macro: bool = True
    require_align_momentum_to_macro: bool = False


@dataclass(frozen=True)
class TripleStochRolesEvent:
    kind: str
    pos: int
    ts: int
    dt: str
    side: str
    status: str
    reason: str
    meta: dict[str, object]


@dataclass(frozen=True)
class TripleStochRolesMetrics:
    pos: int
    ts: int
    dt: str

    macro_side: str
    macro_sign: int

    slow_regime_sign: int
    slow_momentum_sign: int
    slow_is_respiration: bool
    slow_extreme_sign: int
    slow_extreme_dwell: int
    slow_force: float | None
    slow_force_rising: bool

    medium_regime_sign: int
    medium_momentum_sign: int
    medium_is_respiration: bool
    medium_extreme_sign: int
    medium_extreme_dwell: int
    medium_force: float | None
    medium_force_rising: bool

    fast_regime_sign: int
    fast_momentum_sign: int
    fast_is_respiration: bool
    fast_extreme_sign: int
    fast_extreme_dwell: int
    fast_force: float | None
    fast_force_rising: bool

    trigger_level: str
    trigger_long: bool
    trigger_short: bool


@dataclass(frozen=True)
class TripleStochRolesAgentConfig:
    ts_col: str = "ts"
    dt_col: str = "dt"

    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"

    slow: TripleStochLevelConfig = TripleStochLevelConfig(k_col="stoch_k_slow", d_col="stoch_d_slow", k_period=56, d_period=3, slope_window=12)
    medium: TripleStochLevelConfig = TripleStochLevelConfig(k_col="stoch_k_medium", d_col="stoch_d_medium", k_period=28, d_period=3, slope_window=8)
    fast: TripleStochLevelConfig = TripleStochLevelConfig(k_col="stoch_k_fast", d_col="stoch_d_fast", k_period=14, d_period=3, slope_window=6)

    macro_mode: Literal["slow_regime", "slow_regime_and_momentum"] = "slow_regime"

    style: Literal["", "scalp", "swing", "position"] = ""
    entry_trigger_level: Literal["", "slow", "medium", "fast"] = ""

    trigger_mode: Literal["kd_cross", "regime_cross", "extreme_exit", "any"] = "kd_cross"
    require_trigger_in_macro_dir: bool = True

    reject_when_all_three_respire: bool = False


class TripleStochRolesAgent:
    def __init__(self, *, cfg: TripleStochRolesAgentConfig | None = None):
        self.cfg = cfg or TripleStochRolesAgentConfig()

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

    def _maybe_compute_stoch(self, df: pd.DataFrame, *, k_col: str, d_col: str, k_period: int, d_period: int) -> tuple[pd.Series, pd.Series]:
        if str(k_col) in df.columns and str(d_col) in df.columns:
            k_s = pd.to_numeric(df[str(k_col)], errors="coerce").astype(float)
            d_s = pd.to_numeric(df[str(d_col)], errors="coerce").astype(float)
            return k_s, d_s

        cfg = self.cfg
        for c in (cfg.high_col, cfg.low_col, cfg.close_col):
            if str(c) not in df.columns:
                raise ValueError(f"Missing required column for stoch computation: {c}")

        kp = int(k_period)
        dp = int(d_period)
        if kp < 1:
            raise ValueError("stoch_k period must be >= 1")
        if dp < 1:
            raise ValueError("stoch_d period must be >= 1")

        low_s = pd.to_numeric(df[str(cfg.low_col)], errors="coerce").astype(float)
        high_s = pd.to_numeric(df[str(cfg.high_col)], errors="coerce").astype(float)
        close_s = pd.to_numeric(df[str(cfg.close_col)], errors="coerce").astype(float)

        ll = low_s.rolling(window=kp, min_periods=kp).min()
        hh = high_s.rolling(window=kp, min_periods=kp).max()
        denom = (hh - ll).astype(float)
        numer = (close_s - ll).astype(float)

        k = 100.0 * (numer / denom.replace(0.0, np.nan))
        d = k.rolling(window=dp, min_periods=dp).mean()

        return k, d

    def _macro_sign_at(self, work: pd.DataFrame, *, i: int) -> int:
        cfg = self.cfg
        mode = str(cfg.macro_mode or "slow_regime").strip().lower()
        if mode not in {"slow_regime", "slow_regime_and_momentum"}:
            raise ValueError(f"Unsupported macro_mode: {cfg.macro_mode}")

        r = int(work["stoch_slow_regime_sign"].iloc[i])
        if mode == "slow_regime":
            return int(r)

        m = int(work["stoch_slow_momentum_sign"].iloc[i])
        if r != 0 and m != 0 and int(r) == int(m):
            return int(r)
        return 0

    def _level_checks(
        self,
        work: pd.DataFrame,
        *,
        i: int,
        name: str,
        lv: TripleStochLevelConfig,
        macro_sign: int,
    ) -> tuple[bool, str]:
        if not bool(lv.enabled):
            return True, ""

        regime_sign = int(work[f"stoch_{name}_regime_sign"].iloc[i])
        momentum_sign = int(work[f"stoch_{name}_momentum_sign"].iloc[i])
        is_resp = bool(work[f"stoch_{name}_is_respiration"].iloc[i])
        extreme_sign = int(work[f"stoch_{name}_extreme_sign"].iloc[i])

        if bool(lv.reject_regime_transition) and int(regime_sign) == 0:
            return False, f"{name}:regime_transition"

        if bool(lv.require_align_regime_to_macro) and int(macro_sign) != 0:
            if int(regime_sign) != int(macro_sign):
                return False, f"{name}:regime_not_aligned"

        if bool(lv.require_align_momentum_to_macro) and int(macro_sign) != 0:
            if int(momentum_sign) != int(macro_sign):
                return False, f"{name}:momentum_not_aligned"

        if (not bool(lv.allow_trade_when_respiration)) and bool(is_resp):
            return False, f"{name}:respiration_forbidden"

        if (not bool(lv.allow_trade_when_extreme_in_macro_dir)) and int(macro_sign) != 0:
            if int(extreme_sign) == int(macro_sign):
                return False, f"{name}:extreme_in_macro_dir_forbidden"

        f = _safe_float(work[f"stoch_{name}_force"].iloc[i])
        if f is None or not math.isfinite(float(f)):
            return False, f"{name}:force_nan"

        if float(f) < float(lv.min_abs_force):
            return False, f"{name}:force_too_low"

        if bool(lv.require_force_rising):
            if not bool(work[f"stoch_{name}_force_rising"].iloc[i]):
                return False, f"{name}:force_not_rising"

        return True, ""

    def enrich_df(self, df: pd.DataFrame, *, in_place: bool = False) -> pd.DataFrame:
        cfg = self.cfg

        if str(cfg.ts_col) not in df.columns:
            raise ValueError(f"Missing required column: {cfg.ts_col}")

        ts = pd.to_numeric(df[str(cfg.ts_col)], errors="coerce").astype("Int64").to_numpy()

        out = df if bool(in_place) else df.copy()

        def _compute(prefix: str, lv: TripleStochLevelConfig) -> dict[str, np.ndarray]:
            k_s, d_s = self._maybe_compute_stoch(
                out,
                k_col=str(lv.k_col),
                d_col=str(lv.d_col),
                k_period=int(lv.k_period),
                d_period=int(lv.d_period),
            )

            k = pd.to_numeric(k_s, errors="coerce").astype(float).to_numpy()
            d = pd.to_numeric(d_s, errors="coerce").astype(float).to_numpy()

            pivot = float(lv.regime_pivot)
            regime_sign = np.full(int(len(out)), 0, dtype=int)
            for i in range(int(len(out))):
                ki = _safe_float(k[i])
                di = _safe_float(d[i])
                if ki is None or di is None:
                    regime_sign[i] = 0
                    continue
                if float(ki) > float(pivot) and float(di) > float(pivot):
                    regime_sign[i] = 1
                elif float(ki) < float(pivot) and float(di) < float(pivot):
                    regime_sign[i] = -1
                else:
                    regime_sign[i] = 0

            spread = (k - d).astype(float)
            momentum_sign = np.full(int(len(out)), 0, dtype=int)
            for i in range(int(len(out))):
                s = _safe_float(spread[i])
                momentum_sign[i] = _sign(float(s)) if s is not None else 0

            ex_sign = np.full(int(len(out)), 0, dtype=int)
            ex_hi = float(lv.extreme_high)
            ex_lo = float(lv.extreme_low)
            for i in range(int(len(out))):
                ki = _safe_float(k[i])
                if ki is None:
                    ex_sign[i] = 0
                    continue
                if math.isfinite(ex_hi) and float(ki) >= float(ex_hi):
                    ex_sign[i] = 1
                elif math.isfinite(ex_lo) and float(ki) <= float(ex_lo):
                    ex_sign[i] = -1
                else:
                    ex_sign[i] = 0

            ex_dwell = _dwell_arr(ex_sign)

            is_resp = (regime_sign != 0) & (momentum_sign == (-regime_sign))

            k_slope = _rolling_linreg_slope(k, window=int(max(2, int(lv.slope_window))))

            if str(lv.force_mode) == "abs_k_slope":
                force = np.abs(k_slope).astype(float)
            else:
                force = np.abs(spread).astype(float)

            force_rising = _rolling_force_rising(force, bars=int(max(2, int(lv.force_rising_bars))))

            cross_up = np.full(int(len(out)), False, dtype=bool)
            cross_down = np.full(int(len(out)), False, dtype=bool)

            regime_cross_up = np.full(int(len(out)), False, dtype=bool)
            regime_cross_down = np.full(int(len(out)), False, dtype=bool)

            exit_low_extreme = np.full(int(len(out)), False, dtype=bool)
            exit_high_extreme = np.full(int(len(out)), False, dtype=bool)

            for i in range(1, int(len(out))):
                k0 = _safe_float(k[i - 1])
                k1 = _safe_float(k[i])
                d0 = _safe_float(d[i - 1])
                d1 = _safe_float(d[i])
                if k0 is not None and k1 is not None and d0 is not None and d1 is not None:
                    if _cross_up(float(k0), float(k1), float(d0), float(d1)):
                        cross_up[i] = True
                    elif _cross_down(float(k0), float(k1), float(d0), float(d1)):
                        cross_down[i] = True

                if k0 is not None and k1 is not None:
                    if float(k0) <= float(pivot) and float(k1) > float(pivot):
                        regime_cross_up[i] = True
                    elif float(k0) >= float(pivot) and float(k1) < float(pivot):
                        regime_cross_down[i] = True

                    if math.isfinite(ex_lo) and float(k0) <= float(ex_lo) and float(k1) > float(ex_lo):
                        exit_low_extreme[i] = True
                    if math.isfinite(ex_hi) and float(k0) >= float(ex_hi) and float(k1) < float(ex_hi):
                        exit_high_extreme[i] = True

            return {
                f"{prefix}_k": k,
                f"{prefix}_d": d,
                f"{prefix}_spread": spread,
                f"{prefix}_k_slope": k_slope,
                f"{prefix}_regime_sign": regime_sign,
                f"{prefix}_momentum_sign": momentum_sign,
                f"{prefix}_extreme_sign": ex_sign,
                f"{prefix}_extreme_dwell": ex_dwell,
                f"{prefix}_is_respiration": is_resp.astype(bool),
                f"{prefix}_force": force,
                f"{prefix}_force_rising": force_rising,
                f"{prefix}_kd_cross_up": cross_up,
                f"{prefix}_kd_cross_down": cross_down,
                f"{prefix}_regime_cross_up": regime_cross_up,
                f"{prefix}_regime_cross_down": regime_cross_down,
                f"{prefix}_exit_low_extreme": exit_low_extreme,
                f"{prefix}_exit_high_extreme": exit_high_extreme,
            }

        slow = _compute("stoch_slow", cfg.slow)
        medium = _compute("stoch_medium", cfg.medium)
        fast = _compute("stoch_fast", cfg.fast)

        for d in (slow, medium, fast):
            for k, v in d.items():
                out[str(k)] = v

        if str(cfg.dt_col) not in out.columns:
            try:
                out[str(cfg.dt_col)] = pd.to_datetime(ts, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")
            except Exception:
                out[str(cfg.dt_col)] = ""

        return out

    def _trigger_flags_at(self, work: pd.DataFrame, *, i: int, trigger_level: str) -> tuple[bool, bool]:
        cfg = self.cfg
        mode = str(cfg.trigger_mode or "kd_cross").strip().lower()
        if mode not in {"kd_cross", "regime_cross", "extreme_exit", "any"}:
            raise ValueError(f"Unsupported trigger_mode: {cfg.trigger_mode}")

        pfx = f"stoch_{trigger_level}"

        kd_up = bool(work[f"{pfx}_kd_cross_up"].iloc[i])
        kd_dn = bool(work[f"{pfx}_kd_cross_down"].iloc[i])
        rg_up = bool(work[f"{pfx}_regime_cross_up"].iloc[i])
        rg_dn = bool(work[f"{pfx}_regime_cross_down"].iloc[i])
        ex_low = bool(work[f"{pfx}_exit_low_extreme"].iloc[i])
        ex_high = bool(work[f"{pfx}_exit_high_extreme"].iloc[i])

        if mode == "kd_cross":
            return bool(kd_up), bool(kd_dn)
        if mode == "regime_cross":
            return bool(rg_up), bool(rg_dn)
        if mode == "extreme_exit":
            return bool(ex_low), bool(ex_high)
        return bool(kd_up or rg_up or ex_low), bool(kd_dn or rg_dn or ex_high)

    def current_metrics(self, df: pd.DataFrame) -> TripleStochRolesMetrics | None:
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
        trig_long, trig_short = self._trigger_flags_at(work, i=int(i), trigger_level=str(trigger_level))

        def _force(name: str) -> float | None:
            v = _safe_float(work[f"stoch_{name}_force"].iloc[i])
            return None if v is None else float(v)

        def _rising(name: str) -> bool:
            return bool(work[f"stoch_{name}_force_rising"].iloc[i])

        return TripleStochRolesMetrics(
            pos=int(i),
            ts=int(ts_i),
            dt=str(dt),
            macro_side=str(macro_side),
            macro_sign=int(macro_sign),
            slow_regime_sign=int(work["stoch_slow_regime_sign"].iloc[i]),
            slow_momentum_sign=int(work["stoch_slow_momentum_sign"].iloc[i]),
            slow_is_respiration=bool(work["stoch_slow_is_respiration"].iloc[i]),
            slow_extreme_sign=int(work["stoch_slow_extreme_sign"].iloc[i]),
            slow_extreme_dwell=int(work["stoch_slow_extreme_dwell"].iloc[i]),
            slow_force=_force("slow"),
            slow_force_rising=bool(_rising("slow")),
            medium_regime_sign=int(work["stoch_medium_regime_sign"].iloc[i]),
            medium_momentum_sign=int(work["stoch_medium_momentum_sign"].iloc[i]),
            medium_is_respiration=bool(work["stoch_medium_is_respiration"].iloc[i]),
            medium_extreme_sign=int(work["stoch_medium_extreme_sign"].iloc[i]),
            medium_extreme_dwell=int(work["stoch_medium_extreme_dwell"].iloc[i]),
            medium_force=_force("medium"),
            medium_force_rising=bool(_rising("medium")),
            fast_regime_sign=int(work["stoch_fast_regime_sign"].iloc[i]),
            fast_momentum_sign=int(work["stoch_fast_momentum_sign"].iloc[i]),
            fast_is_respiration=bool(work["stoch_fast_is_respiration"].iloc[i]),
            fast_extreme_sign=int(work["stoch_fast_extreme_sign"].iloc[i]),
            fast_extreme_dwell=int(work["stoch_fast_extreme_dwell"].iloc[i]),
            fast_force=_force("fast"),
            fast_force_rising=bool(_rising("fast")),
            trigger_level=str(trigger_level),
            trigger_long=bool(trig_long),
            trigger_short=bool(trig_short),
        )

    def find_entry_events(self, df: pd.DataFrame, *, max_events: int = 200) -> list[TripleStochRolesEvent]:
        cfg = self.cfg
        work = self.enrich_df(df, in_place=False)
        if len(work) <= 0:
            return []

        trigger_level = str(self._entry_trigger_level())
        out: list[TripleStochRolesEvent] = []

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
                if bool(work["stoch_slow_is_respiration"].iloc[i]) and bool(work["stoch_medium_is_respiration"].iloc[i]) and bool(
                    work["stoch_fast_is_respiration"].iloc[i]
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
                TripleStochRolesEvent(
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
