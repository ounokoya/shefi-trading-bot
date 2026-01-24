from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from libs.agents.macd_momentum_two_tf_cycle_agent import MacdMomentiumTwoTFCycleAgent, MacdMomentumTwoTFConfig


@dataclass(frozen=True)
class BacktestMacdMomentumTwoTFConfig:
    ts_col: str = "ts"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"

    fee_rate: float = 0.0015
    use_net: bool = True

    exit_mode: str = "exec_tranche_end"  # exec_tranche_end|opposite_signal|eod

    stoch_k_col: str = "stoch_k"
    stoch_d_col: str = "stoch_d"
    stoch_high: float = 80.0
    stoch_low: float = 20.0
    stoch_wait_extreme: bool = True

    tp_pct: float = 0.0
    trailing_stop_pct: float = 0.0
    sl_pct: float = 0.0


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


def run_backtest_macd_momentum_two_tf(
    *,
    df_exec: pd.DataFrame,
    df_ctx: pd.DataFrame,
    agent_cfg: MacdMomentumTwoTFConfig | None = None,
    bt_cfg: BacktestMacdMomentumTwoTFConfig | None = None,
    start_ts: int | None = None,
    end_ts: int | None = None,
    max_signals: int = 0,
) -> dict[str, object]:
    acfg = agent_cfg or MacdMomentumTwoTFConfig()
    cfg = bt_cfg or BacktestMacdMomentumTwoTFConfig()

    agent = MacdMomentiumTwoTFCycleAgent(cfg=acfg)
    metrics = agent.analyze_df(df_exec, df_ctx=df_ctx, max_signals=int(max_signals))

    signals = [m for m in metrics if str(m.status) == "ACCEPT"]
    if start_ts is not None:
        signals = [m for m in signals if int(m.signal_ts) >= int(start_ts)]
    if end_ts is not None:
        signals = [m for m in signals if int(m.signal_ts) <= int(end_ts)]

    if not signals:
        trades_df = pd.DataFrame([])
        equity_df = pd.DataFrame([])
        return {
            "trades": trades_df,
            "equity": equity_df,
            "summary": {
                "n_trades": 0,
                "equity_end": 0.0,
                "max_dd": 0.0,
                "winrate": 0.0,
            },
            "by_entry_reason": pd.DataFrame([]),
            "by_exit_reason": pd.DataFrame([]),
            "by_entry_group": pd.DataFrame([]),
        }

    ts = pd.to_numeric(df_exec[str(cfg.ts_col)], errors="coerce").astype("Int64").to_numpy()
    o = pd.to_numeric(df_exec[str(cfg.open_col)], errors="coerce").astype(float).to_numpy()
    h = pd.to_numeric(df_exec[str(cfg.high_col)], errors="coerce").astype(float).to_numpy()
    l = pd.to_numeric(df_exec[str(cfg.low_col)], errors="coerce").astype(float).to_numpy()
    c = pd.to_numeric(df_exec[str(cfg.close_col)], errors="coerce").astype(float).to_numpy()

    n = int(len(df_exec))

    equity = 0.0
    equity_points: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []

    in_pos = False
    pos_side = None
    entry_i = None
    entry_ts = None
    entry_price = None
    entry_reason = None
    entry_signal_kind = None
    entry_signal_ts = None
    planned_exit_i = None

    def _close_position(
        *,
        exit_i: int,
        exit_reason: str,
        exit_trigger_reason: str | None,
        exit_trigger_signal_kind: str | None,
        exit_trigger_ts: int | None,
        exit_price_override: float | None = None,
    ) -> None:
        nonlocal in_pos, pos_side, entry_i, entry_ts, entry_price
        nonlocal equity, trades, equity_points
        nonlocal entry_reason, entry_signal_kind, entry_signal_ts
        nonlocal planned_exit_i

        if (not in_pos) or pos_side is None or entry_i is None or entry_ts is None or entry_price is None:
            return

        if int(exit_i) < int(entry_i):
            exit_i = int(entry_i)

        if int(exit_i) >= int(n):
            exit_i = int(n - 1)

        exit_ts2 = int(ts[int(exit_i)]) if not pd.isna(ts[int(exit_i)]) else None
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
                "exit_ts": (None if exit_ts2 is None else int(exit_ts2)),
                "entry_dt": str(_fmt_dt(int(entry_ts))),
                "exit_dt": str(_fmt_dt(exit_ts2)),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price2),
                "gross_ret": float(gross),
                "net_ret": float(net),
                "capture_ret": float(ret),
                "trade_dd": float(dd_trade) if np.isfinite(float(dd_trade)) else float("nan"),
                "duration_s": (None if dur_s is None or (not np.isfinite(float(dur_s))) else float(dur_s)),
                "duration_bars": int(exit_i) - int(entry_i) + 1,
                "entry_reason": ("" if entry_reason is None else str(entry_reason)),
                "entry_signal_kind": ("" if entry_signal_kind is None else str(entry_signal_kind)),
                "entry_signal_ts": (None if entry_signal_ts is None else int(entry_signal_ts)),
                "exit_reason": str(exit_reason),
                "exit_trigger_reason": (None if exit_trigger_reason is None else str(exit_trigger_reason)),
                "exit_trigger_signal_kind": (None if exit_trigger_signal_kind is None else str(exit_trigger_signal_kind)),
                "exit_trigger_ts": (None if exit_trigger_ts is None else int(exit_trigger_ts)),
            }
        )

        equity += float(ret)
        if exit_ts2 is not None:
            equity_points.append({"ts": int(exit_ts2), "equity": float(equity), "event": str(exit_reason)})

        in_pos = False
        pos_side = None
        entry_i = None
        entry_ts = None
        entry_price = None
        entry_reason = None
        entry_signal_kind = None
        entry_signal_ts = None
        planned_exit_i = None

    def _record_trade_from_indices(
        *,
        side: str,
        entry_i2: int,
        exit_i2: int,
        entry_reason2: str,
        entry_signal_kind2: str,
        entry_signal_ts2: int,
        exit_reason2: str,
        exit_trigger_reason2: str | None,
        exit_trigger_signal_kind2: str | None,
        exit_trigger_ts2: int | None,
        exit_price_override: float | None = None,
    ) -> None:
        nonlocal equity, trades, equity_points

        if int(entry_i2) < 0 or int(entry_i2) >= int(n):
            return
        if int(exit_i2) < int(entry_i2):
            exit_i2 = int(entry_i2)
        if int(exit_i2) >= int(n):
            exit_i2 = int(n - 1)

        entry_ts3 = int(ts[int(entry_i2)]) if not pd.isna(ts[int(entry_i2)]) else None
        exit_ts3 = int(ts[int(exit_i2)]) if not pd.isna(ts[int(exit_i2)]) else None
        if entry_ts3 is None or exit_ts3 is None:
            return

        entry_price3 = float(o[int(entry_i2)]) if np.isfinite(float(o[int(entry_i2)])) else float(c[int(entry_i2)])
        exit_price3 = float(o[int(exit_i2)]) if np.isfinite(float(o[int(exit_i2)])) else float(c[int(exit_i2)])
        if exit_price_override is not None and np.isfinite(float(exit_price_override)) and float(exit_price_override) > 0.0:
            exit_price3 = float(exit_price_override)
        if (not np.isfinite(float(entry_price3))) or float(entry_price3) <= 0.0 or (not np.isfinite(float(exit_price3))):
            return

        gross = _gross_ret(side=str(side), entry=float(entry_price3), exit_=float(exit_price3))
        net = float(gross) - (2.0 * float(cfg.fee_rate))
        ret = float(net) if bool(cfg.use_net) else float(gross)

        highs = h[int(entry_i2) : int(exit_i2) + 1]
        lows = l[int(entry_i2) : int(exit_i2) + 1]
        dd_trade = _trade_dd_pct(side=str(side), entry=float(entry_price3), highs=highs, lows=lows)

        dur_s = float(int(exit_ts3) - int(entry_ts3)) / 1000.0

        trades.append(
            {
                "side": str(side),
                "entry_i": int(entry_i2),
                "exit_i": int(exit_i2),
                "entry_ts": int(entry_ts3),
                "exit_ts": int(exit_ts3),
                "entry_dt": str(_fmt_dt(int(entry_ts3))),
                "exit_dt": str(_fmt_dt(int(exit_ts3))),
                "entry_price": float(entry_price3),
                "exit_price": float(exit_price3),
                "gross_ret": float(gross),
                "net_ret": float(net),
                "capture_ret": float(ret),
                "trade_dd": float(dd_trade) if np.isfinite(float(dd_trade)) else float("nan"),
                "duration_s": (None if not np.isfinite(float(dur_s)) else float(dur_s)),
                "duration_bars": int(exit_i2) - int(entry_i2) + 1,
                "entry_reason": str(entry_reason2),
                "entry_signal_kind": str(entry_signal_kind2),
                "entry_signal_ts": int(entry_signal_ts2),
                "exit_reason": str(exit_reason2),
                "exit_trigger_reason": (None if exit_trigger_reason2 is None else str(exit_trigger_reason2)),
                "exit_trigger_signal_kind": (
                    None if exit_trigger_signal_kind2 is None else str(exit_trigger_signal_kind2)
                ),
                "exit_trigger_ts": (None if exit_trigger_ts2 is None else int(exit_trigger_ts2)),
            }
        )

        equity += float(ret)
        equity_points.append({"ts": int(exit_ts3), "equity": float(equity), "event": str(exit_reason2)})

    def _open_position(*, sig) -> None:
        nonlocal in_pos, pos_side, entry_i, entry_ts, entry_price
        nonlocal entry_reason, entry_signal_kind, entry_signal_ts
        nonlocal planned_exit_i

        if in_pos:
            return

        si = sig.exec_i
        if si is None:
            return
        if int(si) < 0 or int(si) >= int(n):
            return

        ts2 = int(ts[int(si)]) if not pd.isna(ts[int(si)]) else None
        px = float(o[int(si)])
        if ts2 is None or (not np.isfinite(float(px))) or float(px) <= 0.0:
            return

        in_pos = True
        pos_side = str(sig.side)
        entry_i = int(si)
        entry_ts = int(ts2)
        entry_price = float(px)
        entry_reason = str(sig.reason)
        entry_signal_kind = str(sig.signal_kind)
        entry_signal_ts = int(sig.signal_ts)

        ex_i = int(sig.exec_tranche_end_i) + 1
        if int(ex_i) >= int(n):
            ex_i = int(n - 1)
        if int(ex_i) < int(entry_i):
            ex_i = int(entry_i)
        planned_exit_i = int(ex_i)

    exit_mode = str(cfg.exit_mode).strip().lower()

    allowed_exit_modes = {"exec_tranche_end", "opposite_signal", "eod", "signal", "tp_pct", "trailing_stop"}
    if exit_mode not in allowed_exit_modes:
        raise ValueError(f"Unexpected exit_mode: {exit_mode}")

    last_allowed_i = int(n - 1)
    if end_ts is not None:
        end_ts2 = int(end_ts)
        j = int(n - 1)
        while j >= 0:
            if (not pd.isna(ts[int(j)])) and int(ts[int(j)]) <= int(end_ts2):
                last_allowed_i = int(j)
                break
            j -= 1
        if int(last_allowed_i) < 0:
            last_allowed_i = 0

    def _find_tp_exit(*, side: str, entry_price2: float, start_i: int) -> tuple[int, float | None, str]:
        if float(cfg.tp_pct) <= 0.0:
            raise ValueError("tp_pct must be > 0 when exit_mode=tp_pct")

        if str(side) == "LONG":
            tp_price = float(entry_price2) * (1.0 + float(cfg.tp_pct))
            for i in range(int(start_i), int(last_allowed_i) + 1):
                if np.isfinite(float(h[int(i)])) and float(h[int(i)]) >= float(tp_price):
                    return int(i), float(tp_price), "TP_PCT"
            return int(last_allowed_i), None, "EOD"

        if str(side) == "SHORT":
            tp_price = float(entry_price2) * (1.0 - float(cfg.tp_pct))
            for i in range(int(start_i), int(last_allowed_i) + 1):
                if np.isfinite(float(l[int(i)])) and float(l[int(i)]) <= float(tp_price):
                    return int(i), float(tp_price), "TP_PCT"
            return int(last_allowed_i), None, "EOD"

        raise ValueError(f"Unexpected side: {side}")

    def _find_sl_exit(*, side: str, entry_price2: float, start_i: int, end_i: int) -> tuple[int, float | None]:
        if float(cfg.sl_pct) <= 0.0:
            return int(end_i), None
        stop_pct = float(cfg.sl_pct)

        if str(side) == "LONG":
            sl_price = float(entry_price2) * (1.0 - float(stop_pct))
            for i in range(int(start_i), int(end_i) + 1):
                if np.isfinite(float(l[int(i)])) and float(l[int(i)]) <= float(sl_price):
                    return int(i), float(sl_price)
            return int(end_i), None

        if str(side) == "SHORT":
            sl_price = float(entry_price2) * (1.0 + float(stop_pct))
            for i in range(int(start_i), int(end_i) + 1):
                if np.isfinite(float(h[int(i)])) and float(h[int(i)]) >= float(sl_price):
                    return int(i), float(sl_price)
            return int(end_i), None

        raise ValueError(f"Unexpected side: {side}")

    def _find_trailing_stop_exit(*, side: str, entry_price2: float, start_i: int) -> tuple[int, float | None, str]:
        if float(cfg.trailing_stop_pct) <= 0.0:
            raise ValueError("trailing_stop_pct must be > 0 when exit_mode=trailing_stop")

        trail = float(cfg.trailing_stop_pct)
        sl_price = None
        if float(cfg.sl_pct) > 0.0:
            if str(side) == "LONG":
                sl_price = float(entry_price2) * (1.0 - float(cfg.sl_pct))
            elif str(side) == "SHORT":
                sl_price = float(entry_price2) * (1.0 + float(cfg.sl_pct))
        if str(side) == "LONG":
            stop = float(entry_price2) * (1.0 - float(trail))
            if sl_price is not None and float(sl_price) > float(stop):
                stop = float(sl_price)
            best = float(entry_price2)
            for i in range(int(start_i), int(last_allowed_i) + 1):
                if np.isfinite(float(l[int(i)])) and float(l[int(i)]) <= float(stop):
                    if sl_price is not None and float(sl_price) >= float(stop):
                        return int(i), float(stop), "SL_PCT"
                    return int(i), float(stop), "TRAILING_STOP"
                if np.isfinite(float(h[int(i)])):
                    best = float(max(float(best), float(h[int(i)])))
                    stop2 = float(best) * (1.0 - float(trail))
                    stop = float(max(float(stop), float(stop2)))
                    if sl_price is not None and float(sl_price) > float(stop):
                        stop = float(sl_price)
            return int(last_allowed_i), None, "EOD"

        if str(side) == "SHORT":
            stop = float(entry_price2) * (1.0 + float(trail))
            if sl_price is not None and float(sl_price) < float(stop):
                stop = float(sl_price)
            best = float(entry_price2)
            for i in range(int(start_i), int(last_allowed_i) + 1):
                if np.isfinite(float(h[int(i)])) and float(h[int(i)]) >= float(stop):
                    if sl_price is not None and float(sl_price) <= float(stop):
                        return int(i), float(stop), "SL_PCT"
                    return int(i), float(stop), "TRAILING_STOP"
                if np.isfinite(float(l[int(i)])):
                    best = float(min(float(best), float(l[int(i)])))
                    stop2 = float(best) * (1.0 + float(trail))
                    stop = float(min(float(stop), float(stop2)))
                    if sl_price is not None and float(sl_price) < float(stop):
                        stop = float(sl_price)
            return int(last_allowed_i), None, "EOD"

        raise ValueError(f"Unexpected side: {side}")

    def _find_stoch_signal_exit(*, side: str, entry_i2: int, k: np.ndarray, d: np.ndarray) -> tuple[int, str]:
        stoch_high = float(cfg.stoch_high)
        stoch_low = float(cfg.stoch_low)
        if (not np.isfinite(float(stoch_high))) or (not np.isfinite(float(stoch_low))):
            raise ValueError("stoch_high/stoch_low must be finite")

        def _is_extreme(i: int) -> bool:
            if int(i) < 0 or int(i) >= int(n):
                return False
            ki = k[int(i)]
            if not np.isfinite(float(ki)):
                return False
            if str(side) == "LONG":
                return float(ki) >= float(stoch_high)
            if str(side) == "SHORT":
                return float(ki) <= float(stoch_low)
            return False

        def _cross_down(i: int) -> bool:
            if int(i) <= 0 or int(i) >= int(n):
                return False
            k0 = k[int(i) - 1]
            d0 = d[int(i) - 1]
            k1 = k[int(i)]
            d1 = d[int(i)]
            if (not np.isfinite(float(k0))) or (not np.isfinite(float(d0))) or (not np.isfinite(float(k1))) or (not np.isfinite(float(d1))):
                return False
            return (float(k0) >= float(d0)) and (float(k1) < float(d1))

        def _cross_up(i: int) -> bool:
            if int(i) <= 0 or int(i) >= int(n):
                return False
            k0 = k[int(i) - 1]
            d0 = d[int(i) - 1]
            k1 = k[int(i)]
            d1 = d[int(i)]
            if (not np.isfinite(float(k0))) or (not np.isfinite(float(d0))) or (not np.isfinite(float(k1))) or (not np.isfinite(float(d1))):
                return False
            return (float(k0) <= float(d0)) and (float(k1) > float(d1))

        armed = bool(_is_extreme(int(entry_i2))) if bool(cfg.stoch_wait_extreme) else True
        for i in range(int(entry_i2) + 1, int(last_allowed_i) + 1):
            if not armed:
                if _is_extreme(int(i)):
                    armed = True
                continue

            if str(side) == "LONG" and _cross_down(int(i)):
                return int(i), "STOCH_CROSS"
            if str(side) == "SHORT" and _cross_up(int(i)):
                return int(i), "STOCH_CROSS"

        return int(last_allowed_i), "EOD"

    if exit_mode == "exec_tranche_end":
        next_available_i = 0
        for sig in sorted(signals, key=lambda x: (int(x.exec_i or -1), int(x.signal_ts), int(x.exec_tranche_id))):
            if sig.exec_i is None:
                continue
            si = int(sig.exec_i)
            if int(si) < int(next_available_i):
                continue
            if int(si) > int(last_allowed_i):
                continue
            ex_i = int(sig.exec_tranche_end_i) + 1
            if int(ex_i) > int(last_allowed_i):
                ex_i = int(last_allowed_i)
            if int(ex_i) < int(si):
                ex_i = int(si)

            entry_px0 = float(o[int(si)]) if np.isfinite(float(o[int(si)])) else float(c[int(si)])
            sl_i, sl_px = _find_sl_exit(side=str(sig.side), entry_price2=float(entry_px0), start_i=int(si), end_i=int(ex_i))
            if sl_px is not None:
                ex_i = int(sl_i)

            if not equity_points:
                entry_ts0 = int(ts[int(si)]) if not pd.isna(ts[int(si)]) else None
                if entry_ts0 is not None:
                    equity_points.append({"ts": int(entry_ts0), "equity": float(equity), "event": "ENTRY"})

            _record_trade_from_indices(
                side=str(sig.side),
                entry_i2=int(si),
                exit_i2=int(ex_i),
                entry_reason2=str(sig.reason),
                entry_signal_kind2=str(sig.signal_kind),
                entry_signal_ts2=int(sig.signal_ts),
                exit_reason2=("SL_PCT" if sl_px is not None else "EXEC_TRANCHE_END"),
                exit_trigger_reason2=None,
                exit_trigger_signal_kind2=None,
                exit_trigger_ts2=None,
                exit_price_override=sl_px,
            )

            next_available_i = int(ex_i) + 1

        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_points)
    elif exit_mode in {"tp_pct", "trailing_stop", "signal"}:
        k = np.array([])
        d = np.array([])
        if exit_mode == "signal":
            if str(cfg.stoch_k_col) not in df_exec.columns or str(cfg.stoch_d_col) not in df_exec.columns:
                raise ValueError("stoch_k_col/stoch_d_col not found in df_exec")
            k = pd.to_numeric(df_exec[str(cfg.stoch_k_col)], errors="coerce").astype(float).to_numpy()
            d = pd.to_numeric(df_exec[str(cfg.stoch_d_col)], errors="coerce").astype(float).to_numpy()

        next_available_i = 0
        for sig in sorted(signals, key=lambda x: (int(x.exec_i or -1), int(x.signal_ts), int(x.exec_tranche_id))):
            if sig.exec_i is None:
                continue
            si = int(sig.exec_i)
            if int(si) < int(next_available_i):
                continue
            if int(si) > int(last_allowed_i):
                continue

            entry_ts0 = int(ts[int(si)]) if not pd.isna(ts[int(si)]) else None
            entry_px0 = float(o[int(si)]) if np.isfinite(float(o[int(si)])) else float(c[int(si)])
            if entry_ts0 is None or (not np.isfinite(float(entry_px0))) or float(entry_px0) <= 0.0:
                continue

            if not equity_points:
                equity_points.append({"ts": int(entry_ts0), "equity": float(equity), "event": "ENTRY"})

            ex_i = int(last_allowed_i)
            exit_px = None
            exit_reason = "EOD"
            exit_trigger_ts2 = None

            if exit_mode == "tp_pct":
                ex_i, exit_px, exit_reason = _find_tp_exit(side=str(sig.side), entry_price2=float(entry_px0), start_i=int(si))
                if exit_reason == "TP_PCT":
                    exit_trigger_ts2 = int(ts[int(ex_i)]) if not pd.isna(ts[int(ex_i)]) else None
            elif exit_mode == "trailing_stop":
                ex_i, exit_px, exit_reason = _find_trailing_stop_exit(
                    side=str(sig.side), entry_price2=float(entry_px0), start_i=int(si)
                )
                if exit_reason == "TRAILING_STOP":
                    exit_trigger_ts2 = int(ts[int(ex_i)]) if not pd.isna(ts[int(ex_i)]) else None
                if exit_reason == "SL_PCT":
                    exit_trigger_ts2 = int(ts[int(ex_i)]) if not pd.isna(ts[int(ex_i)]) else None
            else:
                ex_i, exit_reason = _find_stoch_signal_exit(side=str(sig.side), entry_i2=int(si), k=k, d=d)
                if exit_reason == "STOCH_CROSS":
                    exit_trigger_ts2 = int(ts[int(ex_i)]) if not pd.isna(ts[int(ex_i)]) else None

            if exit_mode in {"tp_pct", "signal"}:
                sl_i, sl_px = _find_sl_exit(
                    side=str(sig.side), entry_price2=float(entry_px0), start_i=int(si), end_i=int(ex_i)
                )
                if sl_px is not None:
                    ex_i = int(sl_i)
                    exit_px = float(sl_px)
                    exit_reason = "SL_PCT"
                    exit_trigger_ts2 = int(ts[int(ex_i)]) if not pd.isna(ts[int(ex_i)]) else None

            if int(ex_i) < int(si):
                ex_i = int(si)
            if int(ex_i) > int(last_allowed_i):
                ex_i = int(last_allowed_i)

            _record_trade_from_indices(
                side=str(sig.side),
                entry_i2=int(si),
                exit_i2=int(ex_i),
                entry_reason2=str(sig.reason),
                entry_signal_kind2=str(sig.signal_kind),
                entry_signal_ts2=int(sig.signal_ts),
                exit_reason2=str(exit_reason),
                exit_trigger_reason2=None,
                exit_trigger_signal_kind2=None,
                exit_trigger_ts2=(None if exit_trigger_ts2 is None else int(exit_trigger_ts2)),
                exit_price_override=exit_px,
            )

            next_available_i = int(ex_i) + 1

        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_points)
    else:
        for sig in sorted(signals, key=lambda x: (int(x.signal_ts), int(x.exec_tranche_id), str(x.signal_kind))):
            if start_ts is not None and int(sig.signal_ts) < int(start_ts):
                continue
            if end_ts is not None and int(sig.signal_ts) > int(end_ts):
                continue

            if sig.exec_i is not None and int(sig.exec_i) > int(last_allowed_i):
                continue

            if not in_pos:
                _open_position(sig=sig)
                if in_pos and entry_ts is not None and (not equity_points):
                    equity_points.append({"ts": int(entry_ts), "equity": float(equity), "event": "ENTRY"})
                continue

            if pos_side is None:
                continue

            if str(sig.side) == str(pos_side):
                continue

            if float(cfg.sl_pct) > 0.0 and entry_i is not None and entry_price is not None and sig.exec_i is not None:
                sl_i, sl_px = _find_sl_exit(
                    side=str(pos_side),
                    entry_price2=float(entry_price),
                    start_i=int(entry_i),
                    end_i=int(min(int(sig.exec_i), int(last_allowed_i))),
                )
                if sl_px is not None and int(sl_i) < int(sig.exec_i):
                    _close_position(
                        exit_i=int(sl_i),
                        exit_reason="SL_PCT",
                        exit_trigger_reason=None,
                        exit_trigger_signal_kind=None,
                        exit_trigger_ts=(None if ts[int(sl_i)] is None or pd.isna(ts[int(sl_i)]) else int(ts[int(sl_i)])),
                        exit_price_override=float(sl_px),
                    )
                    _open_position(sig=sig)
                    continue

            if exit_mode == "opposite_signal":
                ex_i = int(sig.exec_i) if sig.exec_i is not None else None
                if ex_i is None:
                    continue
                if int(ex_i) > int(last_allowed_i):
                    continue
                _close_position(
                    exit_i=int(ex_i),
                    exit_reason="FLIP_SIGNAL",
                    exit_trigger_reason=str(sig.reason),
                    exit_trigger_signal_kind=str(sig.signal_kind),
                    exit_trigger_ts=int(sig.signal_ts),
                )
                _open_position(sig=sig)
                continue

        if in_pos and entry_i is not None:
            ex_i = int(last_allowed_i)
            if float(cfg.sl_pct) > 0.0 and entry_price is not None:
                sl_i, sl_px = _find_sl_exit(
                    side=str(pos_side),
                    entry_price2=float(entry_price),
                    start_i=int(entry_i),
                    end_i=int(ex_i),
                )
                if sl_px is not None:
                    ex_i = int(sl_i)
                    _close_position(
                        exit_i=int(ex_i),
                        exit_reason="SL_PCT",
                        exit_trigger_reason=None,
                        exit_trigger_signal_kind=None,
                        exit_trigger_ts=(None if ts[int(ex_i)] is None or pd.isna(ts[int(ex_i)]) else int(ts[int(ex_i)])),
                        exit_price_override=float(sl_px),
                    )
            if in_pos:
                _close_position(
                    exit_i=int(ex_i),
                    exit_reason="EOD",
                    exit_trigger_reason=None,
                    exit_trigger_signal_kind=None,
                    exit_trigger_ts=None,
                )

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

        dd_series = pd.to_numeric(trades_df.get("trade_dd"), errors="coerce")
        dd_series = dd_series.dropna()

        dur_s_series = pd.to_numeric(trades_df.get("duration_s"), errors="coerce")
        dur_s_series = dur_s_series.dropna()

        dd_max_trade = float(dd_series.max()) if len(dd_series) else 0.0
        dd_mean_trade = float(dd_series.mean()) if len(dd_series) else 0.0
        dd_min_trade = float(dd_series.min()) if len(dd_series) else 0.0

        dur_s_max = float(dur_s_series.max()) if len(dur_s_series) else 0.0
        dur_s_mean = float(dur_s_series.mean()) if len(dur_s_series) else 0.0
        dur_s_min = float(dur_s_series.min()) if len(dur_s_series) else 0.0
    else:
        winrate = 0.0
        dd_max_trade = 0.0
        dd_mean_trade = 0.0
        dd_min_trade = 0.0
        dur_s_max = 0.0
        dur_s_mean = 0.0
        dur_s_min = 0.0

    by_entry_reason = pd.DataFrame([])
    by_exit_reason = pd.DataFrame([])
    by_entry_group = pd.DataFrame([])

    if len(trades_df):
        trades_df["entry_group"] = trades_df["entry_reason"].astype(str) + "|" + trades_df["entry_signal_kind"].astype(str)

        trades_df["_cap"] = pd.to_numeric(trades_df.get("capture_ret"), errors="coerce")
        trades_df["_is_win"] = (pd.to_numeric(trades_df.get("capture_ret"), errors="coerce") > 0.0).astype(float)
        trades_df["_dd"] = pd.to_numeric(trades_df.get("trade_dd"), errors="coerce")
        trades_df["_dur"] = pd.to_numeric(trades_df.get("duration_s"), errors="coerce")

        def _group_stats_df(*, group_col: str) -> pd.DataFrame:
            g = trades_df.groupby(str(group_col), dropna=False)
            out = (
                g.agg(
                    n_trades=("_cap", "count"),
                    pnl_sum=("_cap", "sum"),
                    pnl_mean=("_cap", "mean"),
                    winrate=("_is_win", "mean"),
                    dd_trade_max=("_dd", "max"),
                    dd_trade_mean=("_dd", "mean"),
                    dd_trade_min=("_dd", "min"),
                    duration_s_max=("_dur", "max"),
                    duration_s_mean=("_dur", "mean"),
                    duration_s_min=("_dur", "min"),
                )
                .reset_index()
            )
            return out

        by_entry_reason = _group_stats_df(group_col="entry_reason")
        by_exit_reason = _group_stats_df(group_col="exit_reason")
        by_entry_group = _group_stats_df(group_col="entry_group")

        trades_df = trades_df.drop(columns=["_cap", "_is_win", "_dd", "_dur"], errors="ignore")

        by_entry_reason = by_entry_reason.sort_values(["pnl_sum", "n_trades"], ascending=[False, False]).reset_index(drop=True)
        by_exit_reason = by_exit_reason.sort_values(["pnl_sum", "n_trades"], ascending=[False, False]).reset_index(drop=True)
        by_entry_group = by_entry_group.sort_values(["pnl_sum", "n_trades"], ascending=[False, False]).reset_index(drop=True)

    summary = {
        "n_trades": int(len(trades_df)),
        "equity_end": float(equity_end),
        "max_dd": float(max_dd),
        "winrate": float(winrate),
        "dd_max_trade": float(dd_max_trade),
        "dd_mean_trade": float(dd_mean_trade),
        "dd_min_trade": float(dd_min_trade),
        "duration_s_max": float(dur_s_max),
        "duration_s_mean": float(dur_s_mean),
        "duration_s_min": float(dur_s_min),
    }

    return {
        "trades": trades_df,
        "equity": equity_df,
        "summary": summary,
        "by_entry_reason": by_entry_reason,
        "by_exit_reason": by_exit_reason,
        "by_entry_group": by_entry_group,
    }
