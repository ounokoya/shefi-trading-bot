from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from libs.agents.triple_macd_roles_agent import TripleMacdLevelConfig, TripleMacdRolesAgent, TripleMacdRolesAgentConfig
from libs.backtest_triple_macd_simple_alignment.config import AgentRoleConfig


@dataclass(frozen=True)
class BacktestTripleMacdSimpleAlignmentConfig:
    ts_col: str = "ts"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"

    fee_rate: float = 0.0015
    use_net: bool = True

    exit_mode: str = "opposite_signal"

    tp_pct: float = 0.0
    trailing_stop_pct: float = 0.0
    sl_pct: float = 0.0

    entry_on_next_bar: bool = True


def _gross_ret(*, side: str, entry: float, exit_: float) -> float:
    if entry == 0.0 or (not np.isfinite(float(entry))) or (not np.isfinite(float(exit_))):
        return float("nan")
    if str(side) == "LONG":
        return (float(exit_) / float(entry)) - 1.0
    if str(side) == "SHORT":
        return (float(entry) / float(exit_)) - 1.0
    raise ValueError(f"Unexpected side: {side}")


def _trade_dd_pct(*, side: str, entry: float, highs: np.ndarray, lows: np.ndarray) -> float:
    if entry == 0.0 or (not np.isfinite(float(entry))):
        return float("nan")

    if int(len(highs)) == 0 or int(len(lows)) == 0:
        return float("nan")

    if str(side) == "LONG":
        lo = float(np.nanmin(lows))
        if not np.isfinite(float(lo)):
            return float("nan")
        dd = (float(entry) - float(lo)) / float(entry)
        return float(max(0.0, dd))

    if str(side) == "SHORT":
        hi = float(np.nanmax(highs))
        if not np.isfinite(float(hi)):
            return float("nan")
        dd = (float(hi) - float(entry)) / float(entry)
        return float(max(0.0, dd))

    raise ValueError(f"Unexpected side: {side}")


def _fmt_dt(ts_ms: int | None) -> str:
    if ts_ms is None:
        return ""
    try:
        t = int(ts_ms)
    except Exception:
        return ""
    if t <= 0:
        return ""
    return pd.to_datetime(t, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")


def _intrabar_exit(*, side: str, high: float, low: float, tp_price: float | None, sl_price: float | None) -> tuple[float, str] | None:
    if str(side) == "LONG":
        tp_hit = tp_price is not None and np.isfinite(float(tp_price)) and float(high) >= float(tp_price)
        sl_hit = sl_price is not None and np.isfinite(float(sl_price)) and float(low) <= float(sl_price)
        if sl_hit:
            return float(sl_price), "SL_PCT"
        if tp_hit:
            return float(tp_price), "TP_PCT"
        return None

    if str(side) == "SHORT":
        tp_hit = tp_price is not None and np.isfinite(float(tp_price)) and float(low) <= float(tp_price)
        sl_hit = sl_price is not None and np.isfinite(float(sl_price)) and float(high) >= float(sl_price)
        if sl_hit:
            return float(sl_price), "SL_PCT"
        if tp_hit:
            return float(tp_price), "TP_PCT"
        return None

    raise ValueError(f"Unexpected side: {side}")


def _find_tp_or_sl_exit(
    *,
    side: str,
    entry_price: float,
    start_i: int,
    last_i: int,
    highs: np.ndarray,
    lows: np.ndarray,
    tp_pct: float,
    sl_pct: float,
) -> tuple[int, float | None, str]:
    if float(tp_pct) <= 0.0:
        raise ValueError("tp_pct must be > 0 when exit_mode=tp_pct")

    tp_price = None
    sl_price = None
    if str(side) == "LONG":
        tp_price = float(entry_price) * (1.0 + float(tp_pct))
        sl_price = float(entry_price) * (1.0 - float(sl_pct)) if float(sl_pct) > 0.0 else None
    elif str(side) == "SHORT":
        tp_price = float(entry_price) * (1.0 - float(tp_pct))
        sl_price = float(entry_price) * (1.0 + float(sl_pct)) if float(sl_pct) > 0.0 else None
    else:
        raise ValueError(f"Unexpected side: {side}")

    for i in range(int(start_i), int(last_i) + 1):
        hi = float(highs[int(i)])
        lo = float(lows[int(i)])
        if (not np.isfinite(float(hi))) or (not np.isfinite(float(lo))):
            continue
        hit = _intrabar_exit(side=str(side), high=float(hi), low=float(lo), tp_price=tp_price, sl_price=sl_price)
        if hit is None:
            continue
        px, reason = hit
        return int(i), float(px), str(reason)

    return int(last_i), None, "EOD"


def _find_trailing_stop_exit(
    *,
    side: str,
    entry_price: float,
    start_i: int,
    last_i: int,
    highs: np.ndarray,
    lows: np.ndarray,
    trail_pct: float,
    sl_pct: float,
) -> tuple[int, float | None, str]:
    if float(trail_pct) <= 0.0:
        raise ValueError("trailing_stop_pct must be > 0 when exit_mode=trailing_stop")

    sl_price = None
    if float(sl_pct) > 0.0:
        if str(side) == "LONG":
            sl_price = float(entry_price) * (1.0 - float(sl_pct))
        elif str(side) == "SHORT":
            sl_price = float(entry_price) * (1.0 + float(sl_pct))

    if str(side) == "LONG":
        best = float(entry_price)
        stop = float(entry_price) * (1.0 - float(trail_pct))
        if sl_price is not None and float(sl_price) > float(stop):
            stop = float(sl_price)

        for i in range(int(start_i), int(last_i) + 1):
            hi = float(highs[int(i)])
            lo = float(lows[int(i)])
            if np.isfinite(float(lo)) and float(lo) <= float(stop):
                return int(i), float(stop), ("SL_PCT" if sl_price is not None and float(sl_price) >= float(stop) else "TRAILING_STOP")
            if np.isfinite(float(hi)):
                best = float(max(best, float(hi)))
                stop2 = float(best) * (1.0 - float(trail_pct))
                if sl_price is not None and float(sl_price) > float(stop2):
                    stop2 = float(sl_price)
                stop = float(stop2)
        return int(last_i), None, "EOD"

    if str(side) == "SHORT":
        best = float(entry_price)
        stop = float(entry_price) * (1.0 + float(trail_pct))
        if sl_price is not None and float(sl_price) < float(stop):
            stop = float(sl_price)

        for i in range(int(start_i), int(last_i) + 1):
            hi = float(highs[int(i)])
            lo = float(lows[int(i)])
            if np.isfinite(float(hi)) and float(hi) >= float(stop):
                return int(i), float(stop), ("SL_PCT" if sl_price is not None and float(sl_price) <= float(stop) else "TRAILING_STOP")
            if np.isfinite(float(lo)):
                best = float(min(best, float(lo)))
                stop2 = float(best) * (1.0 + float(trail_pct))
                if sl_price is not None and float(sl_price) < float(stop2):
                    stop2 = float(sl_price)
                stop = float(stop2)
        return int(last_i), None, "EOD"

    raise ValueError(f"Unexpected side: {side}")


def run_backtest_triple_macd_simple_alignment(
    *,
    df: pd.DataFrame,
    agent_cfg: TripleMacdRolesAgentConfig,
    bt_cfg: BacktestTripleMacdSimpleAlignmentConfig,
    start_ts: int | None = None,
    end_ts: int | None = None,
    max_signals: int = 0,
) -> dict[str, object]:
    cfg = bt_cfg

    agent = TripleMacdRolesAgent(cfg=agent_cfg)
    work = agent.enrich_df(df, in_place=False)
    events = agent.find_entry_events(work, max_events=int(max_signals) if int(max_signals) > 0 else 0)

    def _level_ok(*, i: int, name: str, lv: TripleMacdLevelConfig, macro_sign: int) -> bool:
        if not bool(lv.enabled):
            return True

        zone_sign = int(work[f"macd_{name}_zone_sign"].iloc[int(i)])
        hist_sign = int(work[f"macd_{name}_hist_sign"].iloc[int(i)])
        is_resp = bool(work[f"macd_{name}_is_respiration"].iloc[int(i)])

        if bool(lv.reject_zone_transition) and int(zone_sign) == 0:
            return False

        if bool(lv.require_align_zone_to_macro) and int(macro_sign) != 0:
            if int(zone_sign) != int(macro_sign):
                return False

        if bool(lv.require_align_hist_to_macro) and int(macro_sign) != 0:
            if int(hist_sign) != int(macro_sign):
                return False

        if (not bool(lv.allow_trade_when_respiration)) and bool(is_resp):
            return False

        f = work[f"macd_{name}_force"].iloc[int(i)]
        try:
            ff = float(f)
        except Exception:
            return False
        if not np.isfinite(float(ff)):
            return False
        if float(ff) < float(lv.min_abs_force):
            return False
        if bool(lv.require_force_rising):
            if not bool(work[f"macd_{name}_force_rising"].iloc[int(i)]):
                return False

        return True

    if not events:
        return {
            "trades": pd.DataFrame([]),
            "equity": pd.DataFrame([]),
            "summary": {"n_trades": 0, "equity_end": 0.0, "max_dd": 0.0, "winrate": 0.0},
        }

    ts = pd.to_numeric(df[str(cfg.ts_col)], errors="coerce").astype("Int64").to_numpy()
    o = pd.to_numeric(df[str(cfg.open_col)], errors="coerce").astype(float).to_numpy()
    h = pd.to_numeric(df[str(cfg.high_col)], errors="coerce").astype(float).to_numpy()
    l = pd.to_numeric(df[str(cfg.low_col)], errors="coerce").astype(float).to_numpy()
    c = pd.to_numeric(df[str(cfg.close_col)], errors="coerce").astype(float).to_numpy()

    n = int(len(df))
    if n <= 2:
        return {
            "trades": pd.DataFrame([]),
            "equity": pd.DataFrame([]),
            "summary": {"n_trades": 0, "equity_end": 0.0, "max_dd": 0.0, "winrate": 0.0},
        }

    def _event_ts(i: int) -> int | None:
        if int(i) < 0 or int(i) >= int(n):
            return None
        v = ts[int(i)]
        if pd.isna(v):
            return None
        return int(v)

    sigs = []
    for ev in events:
        i0 = int(ev.pos)
        macro_sign = 0
        try:
            macro_sign = int((ev.meta or {}).get("macro_sign") or 0)
        except Exception:
            macro_sign = 0
        if int(macro_sign) == 0:
            try:
                macro_sign = int(work["macd_fast_hist_sign"].iloc[int(i0)])
            except Exception:
                macro_sign = 0

        if not (
            _level_ok(i=int(i0), name="slow", lv=agent_cfg.slow, macro_sign=int(macro_sign))
            and _level_ok(i=int(i0), name="medium", lv=agent_cfg.medium, macro_sign=int(macro_sign))
            and _level_ok(i=int(i0), name="fast", lv=agent_cfg.fast, macro_sign=int(macro_sign))
        ):
            continue

        t0 = _event_ts(int(i0))
        if t0 is None:
            continue
        if start_ts is not None and int(t0) < int(start_ts):
            continue
        if end_ts is not None and int(t0) > int(end_ts):
            continue
        sigs.append(ev)

    sigs.sort(key=lambda e: int(e.pos))
    if not sigs:
        return {
            "trades": pd.DataFrame([]),
            "equity": pd.DataFrame([]),
            "summary": {"n_trades": 0, "equity_end": 0.0, "max_dd": 0.0, "winrate": 0.0},
        }

    exit_mode = str(cfg.exit_mode).strip().lower()

    equity = 0.0
    equity_points: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []

    in_pos = False
    pos_side: str | None = None
    entry_i: int | None = None
    entry_ts: int | None = None
    entry_price: float | None = None
    entry_signal_ts: int | None = None
    entry_signal_i: int | None = None

    def _last_allowed_i() -> int:
        if end_ts is None:
            return int(n - 1)
        last_i = int(n - 1)
        for i in range(int(n - 1), -1, -1):
            t = _event_ts(int(i))
            if t is None:
                continue
            if int(t) <= int(end_ts):
                last_i = int(i)
                break
        return int(max(0, last_i))

    last_i_allowed = _last_allowed_i()

    def _close_position(*, exit_i: int, exit_reason: str, exit_price_override: float | None = None) -> None:
        nonlocal in_pos, pos_side, entry_i, entry_ts, entry_price, equity
        nonlocal entry_signal_ts, entry_signal_i
        nonlocal trades, equity_points

        if (not in_pos) or pos_side is None or entry_i is None or entry_ts is None or entry_price is None:
            return

        if int(exit_i) < int(entry_i):
            exit_i = int(entry_i)
        if int(exit_i) > int(last_i_allowed):
            exit_i = int(last_i_allowed)

        exit_ts2 = _event_ts(int(exit_i))
        if exit_ts2 is None:
            exit_ts2 = int(entry_ts)

        exit_price2 = float(o[int(exit_i)]) if np.isfinite(float(o[int(exit_i)])) else float(c[int(exit_i)])
        if exit_price_override is not None and np.isfinite(float(exit_price_override)) and float(exit_price_override) > 0.0:
            exit_price2 = float(exit_price_override)

        gross = _gross_ret(side=str(pos_side), entry=float(entry_price), exit_=float(exit_price2))
        net = float(gross) - (2.0 * float(cfg.fee_rate))
        ret = float(net) if bool(cfg.use_net) else float(gross)

        highs = h[int(entry_i) : int(exit_i) + 1]
        lows = l[int(entry_i) : int(exit_i) + 1]
        dd_trade = _trade_dd_pct(side=str(pos_side), entry=float(entry_price), highs=highs, lows=lows)

        dur_s = None
        if exit_ts2 is not None:
            dur_s = float(int(exit_ts2) - int(entry_ts)) / 1000.0

        trades.append(
            {
                "side": str(pos_side),
                "entry_i": int(entry_i),
                "exit_i": int(exit_i),
                "entry_ts": int(entry_ts),
                "exit_ts": int(exit_ts2),
                "entry_dt": str(_fmt_dt(int(entry_ts))),
                "exit_dt": str(_fmt_dt(int(exit_ts2))),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price2),
                "gross_ret": float(gross),
                "net_ret": float(net),
                "capture_ret": float(ret),
                "trade_dd": float(dd_trade) if np.isfinite(float(dd_trade)) else float("nan"),
                "duration_s": (None if dur_s is None or (not np.isfinite(float(dur_s))) else float(dur_s)),
                "duration_bars": int(exit_i) - int(entry_i) + 1,
                "entry_reason": "simple_alignment",
                "entry_signal_ts": (None if entry_signal_ts is None else int(entry_signal_ts)),
                "entry_signal_i": (None if entry_signal_i is None else int(entry_signal_i)),
                "exit_reason": str(exit_reason),
            }
        )

        equity += float(ret) if np.isfinite(float(ret)) else 0.0
        equity_points.append({"ts": int(exit_ts2), "equity": float(equity), "event": str(exit_reason)})

        in_pos = False
        pos_side = None
        entry_i = None
        entry_ts = None
        entry_price = None
        entry_signal_ts = None
        entry_signal_i = None

    def _open_position(*, sig_i: int, side: str, signal_i: int, signal_ts: int) -> None:
        nonlocal in_pos, pos_side, entry_i, entry_ts, entry_price
        nonlocal entry_signal_ts, entry_signal_i

        if in_pos:
            return

        if bool(cfg.entry_on_next_bar):
            sig_i = int(sig_i) + 1

        if int(sig_i) < 0 or int(sig_i) >= int(n):
            return
        if int(sig_i) > int(last_i_allowed):
            return

        ts_i = _event_ts(int(sig_i))
        if ts_i is None:
            return

        px = float(o[int(sig_i)]) if np.isfinite(float(o[int(sig_i)])) else float(c[int(sig_i)])
        if (not np.isfinite(float(px))) or float(px) <= 0.0:
            return

        in_pos = True
        pos_side = str(side)
        entry_i = int(sig_i)
        entry_ts = int(ts_i)
        entry_price = float(px)
        entry_signal_ts = int(signal_ts)
        entry_signal_i = int(signal_i)

        equity_points.append({"ts": int(ts_i), "equity": float(equity), "event": f"ENTRY_{side}"})

    next_allowed_signal_pos = 0

    for ev in sigs:
        si = int(ev.pos)
        if int(si) < int(next_allowed_signal_pos):
            continue

        t_sig = _event_ts(int(si))
        if t_sig is None:
            continue

        side = str(ev.side)
        if side not in {"LONG", "SHORT"}:
            continue

        if not in_pos:
            _open_position(sig_i=int(si), side=str(side), signal_i=int(si), signal_ts=int(t_sig))
            continue

        if pos_side is None or entry_i is None or entry_price is None:
            continue

        if exit_mode == "opposite_signal":
            if str(side) != str(pos_side):
                exit_i = int(si)
                exit_ts0 = _event_ts(int(exit_i))
                if exit_ts0 is None:
                    exit_ts0 = int(entry_ts) if entry_ts is not None else None

                if bool(cfg.entry_on_next_bar):
                    exit_i = int(exit_i) + 1
                if int(exit_i) > int(last_i_allowed):
                    exit_i = int(last_i_allowed)

                _close_position(exit_i=int(exit_i), exit_reason="OPPOSITE_SIGNAL")
                _open_position(sig_i=int(si), side=str(side), signal_i=int(si), signal_ts=int(t_sig))
                next_allowed_signal_pos = int(si)

        if exit_mode in {"tp_pct", "trailing_stop"}:
            if entry_i is None or entry_price is None:
                continue

            start_i = int(entry_i)
            last_i = int(last_i_allowed)
            if exit_mode == "tp_pct":
                ex_i, ex_px, ex_reason = _find_tp_or_sl_exit(
                    side=str(pos_side),
                    entry_price=float(entry_price),
                    start_i=int(start_i),
                    last_i=int(last_i),
                    highs=h,
                    lows=l,
                    tp_pct=float(cfg.tp_pct),
                    sl_pct=float(cfg.sl_pct),
                )
                _close_position(exit_i=int(ex_i), exit_reason=str(ex_reason), exit_price_override=ex_px)
            else:
                ex_i, ex_px, ex_reason = _find_trailing_stop_exit(
                    side=str(pos_side),
                    entry_price=float(entry_price),
                    start_i=int(start_i),
                    last_i=int(last_i),
                    highs=h,
                    lows=l,
                    trail_pct=float(cfg.trailing_stop_pct),
                    sl_pct=float(cfg.sl_pct),
                )
                _close_position(exit_i=int(ex_i), exit_reason=str(ex_reason), exit_price_override=ex_px)

            next_allowed_signal_pos = int(si)

    if in_pos and entry_i is not None:
        if exit_mode == "eod":
            _close_position(exit_i=int(last_i_allowed), exit_reason="EOD")
        elif exit_mode == "opposite_signal":
            _close_position(exit_i=int(last_i_allowed), exit_reason="EOD")
        elif exit_mode == "tp_pct":
            ex_i, ex_px, ex_reason = _find_tp_or_sl_exit(
                side=str(pos_side),
                entry_price=float(entry_price),
                start_i=int(entry_i),
                last_i=int(last_i_allowed),
                highs=h,
                lows=l,
                tp_pct=float(cfg.tp_pct),
                sl_pct=float(cfg.sl_pct),
            )
            _close_position(exit_i=int(ex_i), exit_reason=str(ex_reason), exit_price_override=ex_px)
        else:
            ex_i, ex_px, ex_reason = _find_trailing_stop_exit(
                side=str(pos_side),
                entry_price=float(entry_price),
                start_i=int(entry_i),
                last_i=int(last_i_allowed),
                highs=h,
                lows=l,
                trail_pct=float(cfg.trailing_stop_pct),
                sl_pct=float(cfg.sl_pct),
            )
            _close_position(exit_i=int(ex_i), exit_reason=str(ex_reason), exit_price_override=ex_px)

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_points)

    if len(equity_df):
        equity_df = equity_df.sort_values("ts").reset_index(drop=True)
        equity_df["peak"] = equity_df["equity"].cummax()
        equity_df["dd"] = equity_df["equity"] - equity_df["peak"]
        max_dd = float(equity_df["dd"].min())
        equity_end = float(equity_df["equity"].iloc[-1])
    else:
        max_dd = 0.0
        equity_end = 0.0

    if len(trades_df):
        wins_mask = pd.to_numeric(trades_df["capture_ret"], errors="coerce") > 0.0
        winrate = float(wins_mask.mean())
    else:
        winrate = 0.0

    summary = {
        "n_trades": int(len(trades_df)),
        "equity_end": float(equity_end),
        "max_dd": float(max_dd),
        "winrate": float(winrate),
    }

    return {
        "trades": trades_df,
        "equity": equity_df,
        "summary": summary,
    }


def _lv_from_role(*, role: AgentRoleConfig, line_col: str, signal_col: str, hist_col: str) -> TripleMacdLevelConfig:
    return TripleMacdLevelConfig(
        enabled=bool(role.enabled),
        line_col=str(line_col),
        signal_col=str(signal_col),
        hist_col=str(hist_col),
        reject_zone_transition=bool(role.reject_zone_transition),
        min_abs_force=float(role.min_abs_force),
        require_force_rising=bool(role.require_force_rising),
        force_rising_bars=int(role.force_rising_bars),
        allow_trade_when_respiration=bool(role.allow_trade_when_respiration),
        require_align_zone_to_macro=bool(role.require_align_zone_to_macro),
        require_align_hist_to_macro=bool(role.require_align_hist_to_macro),
    )


def build_agent_config(
    *,
    ts_col: str = "ts",
    dt_col: str = "dt",
    close_col: str = "close",
    hist_zero_policy: str,
    require_hists_rising_on_entry: bool,
    slow_role: AgentRoleConfig,
    medium_role: AgentRoleConfig,
    fast_role: AgentRoleConfig,
) -> TripleMacdRolesAgentConfig:
    slow_lv = _lv_from_role(
        role=slow_role,
        line_col="macd_line_slow",
        signal_col="macd_signal_slow",
        hist_col="macd_hist_slow",
    )
    medium_lv = _lv_from_role(
        role=medium_role,
        line_col="macd_line_medium",
        signal_col="macd_signal_medium",
        hist_col="macd_hist_medium",
    )
    fast_lv = _lv_from_role(
        role=fast_role,
        line_col="macd_line_fast",
        signal_col="macd_signal_fast",
        hist_col="macd_hist_fast",
    )

    return TripleMacdRolesAgentConfig(
        ts_col=str(ts_col),
        dt_col=str(dt_col),
        close_col=str(close_col),
        slow=slow_lv,
        medium=medium_lv,
        fast=fast_lv,
        hist_zero_policy=str(hist_zero_policy),
        macro_mode="slow_hist",
        entry_style="simple_alignment",
        require_hists_rising_on_entry=bool(require_hists_rising_on_entry),
        style="",
        entry_trigger_level="fast",
        require_trigger_in_macro_dir=False,
        reject_when_all_three_respire=False,
    )
