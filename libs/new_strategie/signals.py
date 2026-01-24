from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from libs.new_strategie.config import NewStrategieConfig
from libs.new_strategie.pivots import PivotPoint, is_in_any_pivot_zone


def _safe_float(v: object) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return float(x)


def _dt(ts: int) -> str:
    if int(ts) <= 0:
        return ""
    return str(pd.to_datetime(int(ts), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC"))


@dataclass(frozen=True)
class StrategySignal:
    kind: str  # premature | validated
    pos: int
    ts: int
    dt: str
    side: str  # LONG | SHORT
    meta: dict[str, object]


def _stoch_cross_side(*, k0: float, k1: float, d0: float, d1: float) -> str | None:
    # K crosses D
    if float(k0) <= float(d0) and float(k1) > float(d1):
        return "LONG"
    if float(k0) >= float(d0) and float(k1) < float(d1):
        return "SHORT"
    return None


def _stoch_d_exit_side(*, d0: float, d1: float, low: float, high: float) -> str | None:
    # Exit low extreme => LONG, exit high extreme => SHORT
    if math.isfinite(low) and float(d0) <= float(low) and float(d1) > float(low):
        return "LONG"
    if math.isfinite(high) and float(d0) >= float(high) and float(d1) < float(high):
        return "SHORT"
    return None


def _cci_return_side(*, c0: float, c1: float, level: float) -> str | None:
    # Return from +extreme => SHORT, return from -extreme => LONG
    lvl = float(abs(level))
    if float(c0) >= float(lvl) and float(c1) < float(lvl):
        return "SHORT"
    if float(c0) <= -float(lvl) and float(c1) > -float(lvl):
        return "LONG"
    return None


def _dmi_exhaustion_side(*, dx0: float, dx1: float, adx0: float, adx1: float) -> bool:
    # DX crosses under ADX
    return bool(float(dx0) > float(adx0) and float(dx1) <= float(adx1))


def find_signals(
    df: pd.DataFrame,
    *,
    pivots: list[PivotPoint],
    cfg: NewStrategieConfig,
    max_signals: int = 500,
) -> list[StrategySignal]:
    """Find signals (premature in pivot zone, validated outside pivot zone).

    Conditions are allowed to happen on different bars and in any order.
    We enforce that all required conditions must occur within a rolling window of N bars.
    """

    for c in (
        cfg.ts_col,
        cfg.close_col,
        cfg.stoch_k_col,
        cfg.stoch_d_col,
        cfg.cci_col,
        cfg.dx_col,
        cfg.adx_col,
    ):
        if str(c) not in df.columns:
            raise ValueError(f"Missing required column for signals: {c}")

    ts_s = pd.to_numeric(df[str(cfg.ts_col)], errors="coerce").astype("Int64").to_numpy()
    close_s = pd.to_numeric(df[str(cfg.close_col)], errors="coerce").astype(float).to_numpy()

    k_s = pd.to_numeric(df[str(cfg.stoch_k_col)], errors="coerce").astype(float).to_numpy()
    d_s = pd.to_numeric(df[str(cfg.stoch_d_col)], errors="coerce").astype(float).to_numpy()

    cci_s = pd.to_numeric(df[str(cfg.cci_col)], errors="coerce").astype(float).to_numpy()

    dx_s = pd.to_numeric(df[str(cfg.dx_col)], errors="coerce").astype(float).to_numpy()
    adx_s = pd.to_numeric(df[str(cfg.adx_col)], errors="coerce").astype(float).to_numpy()

    win = int(cfg.signal_condition_window_bars)
    if win < 1:
        win = 1

    last_pivot_contact: int | None = None
    last_stoch_cross: tuple[int, str] | None = None
    last_stoch_d_exit: tuple[int, str] | None = None
    last_cci_return: tuple[int, str] | None = None
    last_dmi_exhaust: int | None = None

    out: list[StrategySignal] = []

    def _within(i: int, j: int | None) -> bool:
        if j is None:
            return False
        return bool(int(i) - int(j) <= int(win))

    def _within2(i: int, ev: tuple[int, str] | None) -> bool:
        if ev is None:
            return False
        return bool(int(i) - int(ev[0]) <= int(win))

    for i in range(1, int(len(df))):
        ts_i = int(ts_s[i]) if ts_s[i] is not None else 0
        close_i = _safe_float(close_s[i])
        if close_i is None:
            continue

        # Context: pivot zone
        in_zone = is_in_any_pivot_zone(price=float(close_i), pivots=pivots)
        if bool(in_zone):
            last_pivot_contact = int(i)

        # stoch cross
        k0 = _safe_float(k_s[i - 1])
        k1 = _safe_float(k_s[i])
        d0 = _safe_float(d_s[i - 1])
        d1 = _safe_float(d_s[i])
        if k0 is not None and k1 is not None and d0 is not None and d1 is not None:
            side_sc = _stoch_cross_side(k0=float(k0), k1=float(k1), d0=float(d0), d1=float(d1))
            if side_sc is not None:
                last_stoch_cross = (int(i), str(side_sc))

            side_exit = _stoch_d_exit_side(
                d0=float(d0),
                d1=float(d1),
                low=float(cfg.stoch_extreme_low),
                high=float(cfg.stoch_extreme_high),
            )
            if side_exit is not None:
                last_stoch_d_exit = (int(i), str(side_exit))

        # cci return
        c0 = _safe_float(cci_s[i - 1])
        c1 = _safe_float(cci_s[i])
        if c0 is not None and c1 is not None:
            side_cci = _cci_return_side(c0=float(c0), c1=float(c1), level=float(cfg.cci_extreme_level))
            if side_cci is not None:
                last_cci_return = (int(i), str(side_cci))

        # dmi exhaustion (dx cross under adx)
        dx0 = _safe_float(dx_s[i - 1])
        dx1 = _safe_float(dx_s[i])
        adx0 = _safe_float(adx_s[i - 1])
        adx1 = _safe_float(adx_s[i])
        if dx0 is not None and dx1 is not None and adx0 is not None and adx1 is not None:
            if _dmi_exhaustion_side(dx0=float(dx0), dx1=float(dx1), adx0=float(adx0), adx1=float(adx1)):
                last_dmi_exhaust = int(i)

        # Signal evaluation at current bar
        dt_i = _dt(int(ts_i))

        # Premature (in pivot zone)
        if bool(in_zone):
            ok_contact = _within(i, last_pivot_contact)
            ok_sc = _within2(i, last_stoch_cross)
            ok_exit = _within2(i, last_stoch_d_exit)
            ok_cci = _within2(i, last_cci_return)
            if ok_contact and ok_sc and ok_exit and ok_cci:
                # choose side: prefer stoch exit side, else stoch cross side, else cci side
                side = str(last_stoch_d_exit[1]) if last_stoch_d_exit is not None else str(last_stoch_cross[1])
                meta = {
                    "close": float(close_i),
                    "in_pivot_zone": True,
                    "pivot_contact_pos": int(last_pivot_contact or i),
                    "stoch_cross_pos": (None if last_stoch_cross is None else int(last_stoch_cross[0])),
                    "stoch_cross_side": (None if last_stoch_cross is None else str(last_stoch_cross[1])),
                    "stoch_d_exit_pos": (None if last_stoch_d_exit is None else int(last_stoch_d_exit[0])),
                    "stoch_d_exit_side": (None if last_stoch_d_exit is None else str(last_stoch_d_exit[1])),
                    "cci_return_pos": (None if last_cci_return is None else int(last_cci_return[0])),
                    "cci_return_side": (None if last_cci_return is None else str(last_cci_return[1])),
                    "window_bars": int(win),
                    "k": _safe_float(k_s[i]),
                    "d": _safe_float(d_s[i]),
                    "cci": _safe_float(cci_s[i]),
                    "adx": _safe_float(adx_s[i]),
                    "dx": _safe_float(dx_s[i]),
                }
                out.append(
                    StrategySignal(
                        kind="premature",
                        pos=int(i),
                        ts=int(ts_i),
                        dt=str(dt_i),
                        side=str(side),
                        meta=meta,
                    )
                )
                # reset conditions once consumed
                last_pivot_contact = None
                last_stoch_cross = None
                last_stoch_d_exit = None
                last_cci_return = None

        # Validated (out of pivot zone)
        if not bool(in_zone):
            ok_exit = _within2(i, last_stoch_d_exit)
            ok_dmi = _within(i, last_dmi_exhaust)
            if ok_exit and ok_dmi and last_stoch_d_exit is not None:
                side = str(last_stoch_d_exit[1])
                meta = {
                    "close": float(close_i),
                    "in_pivot_zone": False,
                    "stoch_d_exit_pos": int(last_stoch_d_exit[0]),
                    "stoch_d_exit_side": str(last_stoch_d_exit[1]),
                    "dmi_exhaust_pos": int(last_dmi_exhaust or i),
                    "window_bars": int(win),
                    "k": _safe_float(k_s[i]),
                    "d": _safe_float(d_s[i]),
                    "cci": _safe_float(cci_s[i]),
                    "adx": _safe_float(adx_s[i]),
                    "dx": _safe_float(dx_s[i]),
                }
                out.append(
                    StrategySignal(
                        kind="validated",
                        pos=int(i),
                        ts=int(ts_i),
                        dt=str(dt_i),
                        side=str(side),
                        meta=meta,
                    )
                )
                last_stoch_d_exit = None
                last_dmi_exhaust = None

        if int(max_signals) > 0 and len(out) > int(max_signals):
            out = out[-int(max_signals) :]

    out.sort(key=lambda x: (int(x.ts), int(x.pos)))
    return out
