from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from libs.backtest.config import FullConfig
from libs.backtest.indicators import ensure_indicators_df
from libs.backtest.signals import entry_signal, exit_signal


@dataclass
class PositionState:
    side: str  # LONG|SHORT
    entry_i: int
    entry_ts: int
    entry_price: float

    peak: float
    trough: float

    atr_at_entry: float | None


@dataclass
class TradeRecord:
    side: str
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    exit_reason: str

    entry_signal_ts: int | None
    entry_signal_cand_ts: int | None
    entry_confirmed_series: str | None
    entry_trend_filter: str | None
    entry_trend_ok: bool | None
    entry_trend_vortex_side: str | None
    entry_trend_dmi_side: str | None

    exit_signal_ts: int | None

    gross_ret: float
    net_ret: float

    mfe: float
    mae: float


def _gross_ret(*, side: str, entry: float, exit_: float) -> float:
    if entry == 0.0 or (not np.isfinite(float(entry))) or (not np.isfinite(float(exit_))):
        return float("nan")
    if side == "LONG":
        return (float(exit_) / float(entry)) - 1.0
    if side == "SHORT":
        return (float(entry) / float(exit_)) - 1.0
    raise ValueError(f"Unexpected side: {side}")


def _tp_price(*, side: str, entry: float, mode: str, tp_pct: float | None) -> float | None:
    m = str(mode).lower()
    if m in {"", "none", "off", "0"}:
        return None
    if m == "fixed_pct":
        if tp_pct is None:
            return None
        if side == "LONG":
            return float(entry) * (1.0 + float(tp_pct))
        if side == "SHORT":
            return float(entry) * (1.0 - float(tp_pct))
        raise ValueError(f"Unexpected side: {side}")
    raise ValueError(f"Unexpected tp.mode: {mode}")


def _sl_price(*, side: str, pos: PositionState, mode: str, sl_pct: float | None, trail_pct: float | None, atr_mult: float | None) -> float | None:
    m = str(mode).lower()
    if m in {"", "none", "off", "0"}:
        return None

    if m == "fixed_pct":
        if sl_pct is None:
            return None
        if side == "LONG":
            return float(pos.entry_price) * (1.0 - float(sl_pct))
        if side == "SHORT":
            return float(pos.entry_price) * (1.0 + float(sl_pct))
        raise ValueError(f"Unexpected side: {side}")

    if m == "trailing_pct":
        if trail_pct is None:
            return None
        if side == "LONG":
            return float(pos.peak) * (1.0 - float(trail_pct))
        if side == "SHORT":
            return float(pos.trough) * (1.0 + float(trail_pct))
        raise ValueError(f"Unexpected side: {side}")

    if m == "atr":
        if atr_mult is None:
            return None
        if pos.atr_at_entry is None or (not np.isfinite(float(pos.atr_at_entry))):
            return None
        dist = float(atr_mult) * float(pos.atr_at_entry)
        if side == "LONG":
            return float(pos.entry_price) - dist
        if side == "SHORT":
            return float(pos.entry_price) + dist
        raise ValueError(f"Unexpected side: {side}")

    raise ValueError(f"Unexpected sl.mode: {mode}")


def _intrabar_exit(
    *,
    side: str,
    high: float,
    low: float,
    tp_price: float | None,
    sl_price: float | None,
) -> tuple[float, str] | None:
    if side == "LONG":
        tp_hit = tp_price is not None and float(high) >= float(tp_price)
        sl_hit = sl_price is not None and float(low) <= float(sl_price)
        if sl_hit:
            return float(sl_price), "SL"
        if tp_hit:
            return float(tp_price), "TP"
        return None

    if side == "SHORT":
        tp_hit = tp_price is not None and float(low) <= float(tp_price)
        sl_hit = sl_price is not None and float(high) >= float(sl_price)
        if sl_hit:
            return float(sl_price), "SL"
        if tp_hit:
            return float(tp_price), "TP"
        return None

    raise ValueError(f"Unexpected side: {side}")


def run_backtest_from_config(
    *,
    cfg: FullConfig,
    df: pd.DataFrame | None = None,
    start_ts: int | None = None,
    end_ts: int | None = None,
    ensure_indicators: bool = True,
) -> dict[str, object]:
    if df is None:
        df = pd.read_csv(Path(cfg.data.csv))
    if bool(ensure_indicators):
        df = ensure_indicators_df(df, cfg=cfg)

    ts_col = cfg.data.ts_col
    open_col = cfg.data.ohlc.open
    high_col = cfg.data.ohlc.high
    low_col = cfg.data.ohlc.low
    close_col = cfg.data.ohlc.close

    atr_len_eff = int(cfg.indicators.atr_len)
    if str(cfg.sl.mode).lower() == "atr" and cfg.sl.atr_len is not None:
        atr_len_eff = int(cfg.sl.atr_len)
    atr_col = f"atr_{int(atr_len_eff)}"

    n = int(len(df))
    if n < 3:
        raise ValueError(f"Not enough rows: {n}")

    ts_vals = pd.to_numeric(df[ts_col], errors="coerce").to_numpy()
    start_i = 0
    end_i = n - 1
    if start_ts is not None:
        start_i = int(np.searchsorted(ts_vals, int(start_ts), side="left"))
    if end_ts is not None:
        end_i = int(np.searchsorted(ts_vals, int(end_ts), side="right") - 1)
    if start_i < 0:
        start_i = 0
    if end_i > n - 1:
        end_i = n - 1
    if start_i > end_i:
        raise ValueError(f"Empty backtest range: start_ts={start_ts} end_ts={end_ts}")

    pos: PositionState | None = None
    pending_entry: dict[str, object] | None = None
    pending_exit: dict[str, object] | None = None
    last_entry_meta: dict[str, object] | None = None
    last_exit_signal_ts: int | None = None

    equity = 0.0
    equity_points: list[dict[str, object]] = [
        {"ts": int(df[ts_col].iloc[start_i]), "equity": float(equity), "event": "START"}
    ]
    trades: list[TradeRecord] = []

    win = int(cfg.backtest.window_size)
    fee_rate = float(cfg.backtest.fee_rate)
    min_net_for_signal_exit = 4.0 * float(fee_rate)

    for i in range(start_i, end_i + 1):
        bar = df.iloc[i]
        ts = int(bar[ts_col])
        o = float(bar[open_col])
        h = float(bar[high_col])
        l = float(bar[low_col])
        c = float(bar[close_col])

        # 1) execute pending EXIT at open(i)
        if pos is not None and pending_exit is not None and int(pending_exit["exec_i"]) == i:
            exit_price = float(o)
            gross = _gross_ret(side=pos.side, entry=pos.entry_price, exit_=exit_price)
            net = gross - (2.0 * fee_rate)

            peak2 = max(float(pos.peak), float(exit_price))
            trough2 = min(float(pos.trough), float(exit_price))
            if pos.side == "LONG":
                mfe = (float(peak2) / float(pos.entry_price)) - 1.0
                mae = (float(trough2) / float(pos.entry_price)) - 1.0
            else:
                mfe = (float(pos.entry_price) / float(trough2)) - 1.0
                mae = (float(pos.entry_price) / float(peak2)) - 1.0

            # Exit by signal only if net profit covers a minimum threshold.
            # The user requirement is to ignore signal exits unless net_ret >= 2x roundtrip fee.
            # With fee_rate=0.15% per side => roundtrip=0.30% => threshold=0.60%.
            is_signal_exit = str(pending_exit.get("reason", "SIGNAL")).upper() == "SIGNAL"
            if is_signal_exit and ((not np.isfinite(float(net))) or float(net) < float(min_net_for_signal_exit)):
                pending_exit = None
                last_exit_signal_ts = None
            else:
                trades.append(
                    TradeRecord(
                        side=pos.side,
                        entry_ts=int(pos.entry_ts),
                        exit_ts=int(ts),
                        entry_price=float(pos.entry_price),
                        exit_price=float(exit_price),
                        exit_reason=str(pending_exit.get("reason", "SIGNAL")),
                        entry_signal_ts=(
                            None if last_entry_meta is None else int(last_entry_meta.get("signal_ts") or 0) or None
                        ),
                        entry_signal_cand_ts=(
                            None if last_entry_meta is None else int(last_entry_meta.get("cand_ts") or 0) or None
                        ),
                        entry_confirmed_series=(
                            None
                            if last_entry_meta is None
                            else "|".join(list(last_entry_meta.get("confirmed_series") or []))
                        ),
                        entry_trend_filter=(None if last_entry_meta is None else last_entry_meta.get("trend_filter")),
                        entry_trend_ok=(None if last_entry_meta is None else last_entry_meta.get("trend_ok")),
                        entry_trend_vortex_side=(None if last_entry_meta is None else last_entry_meta.get("trend_vortex_side")),
                        entry_trend_dmi_side=(None if last_entry_meta is None else last_entry_meta.get("trend_dmi_side")),
                        exit_signal_ts=(None if last_exit_signal_ts is None else int(last_exit_signal_ts)),
                        gross_ret=float(gross),
                        net_ret=float(net),
                        mfe=float(mfe),
                        mae=float(mae),
                    )
                )

                equity += float(net)
                equity_points.append({"ts": int(ts), "equity": float(equity), "event": "EXIT"})

                pos = None
                pending_exit = None
                last_exit_signal_ts = None

        # 2) execute pending ENTRY at open(i)
        if pos is None and pending_entry is not None and int(pending_entry["exec_i"]) == i:
            side = str(pending_entry["side"]).upper()
            atr_val = None
            if i - 1 >= 0:
                atr_val = float(pd.to_numeric(df[atr_col].iloc[i - 1], errors="coerce"))
                if not np.isfinite(float(atr_val)):
                    atr_val = None

            pos = PositionState(
                side=side,
                entry_i=int(i),
                entry_ts=int(ts),
                entry_price=float(o),
                peak=float(o),
                trough=float(o),
                atr_at_entry=atr_val,
            )
            pending_entry = None
            last_exit_signal_ts = None

        # 3) intrabar TP/SL
        if pos is not None:
            prev_peak = float(pos.peak)
            prev_trough = float(pos.trough)

            tp_p = _tp_price(side=pos.side, entry=pos.entry_price, mode=cfg.tp.mode, tp_pct=cfg.tp.tp_pct)
            sl_p = _sl_price(
                side=pos.side,
                pos=pos,
                mode=cfg.sl.mode,
                sl_pct=cfg.sl.sl_pct,
                trail_pct=cfg.sl.trail_pct,
                atr_mult=cfg.sl.atr_mult,
            )

            hit = _intrabar_exit(side=pos.side, high=h, low=l, tp_price=tp_p, sl_price=sl_p)
            if hit is not None:
                exit_price, reason = hit
                gross = _gross_ret(side=pos.side, entry=pos.entry_price, exit_=exit_price)
                net = gross - (2.0 * fee_rate)

                peak2 = max(prev_peak, float(h))
                trough2 = min(prev_trough, float(l))

                if pos.side == "LONG":
                    mfe = (float(peak2) / float(pos.entry_price)) - 1.0
                    mae = (float(trough2) / float(pos.entry_price)) - 1.0
                else:
                    mfe = (float(pos.entry_price) / float(trough2)) - 1.0
                    mae = (float(pos.entry_price) / float(peak2)) - 1.0

                trades.append(
                    TradeRecord(
                        side=pos.side,
                        entry_ts=int(pos.entry_ts),
                        exit_ts=int(ts),
                        entry_price=float(pos.entry_price),
                        exit_price=float(exit_price),
                        exit_reason=str(reason),
                        entry_signal_ts=(None if last_entry_meta is None else int(last_entry_meta.get("signal_ts") or 0) or None),
                        entry_signal_cand_ts=(None if last_entry_meta is None else int(last_entry_meta.get("cand_ts") or 0) or None),
                        entry_confirmed_series=(
                            None
                            if last_entry_meta is None
                            else "|".join(list(last_entry_meta.get("confirmed_series") or []))
                        ),
                        entry_trend_filter=(None if last_entry_meta is None else last_entry_meta.get("trend_filter")),
                        entry_trend_ok=(None if last_entry_meta is None else last_entry_meta.get("trend_ok")),
                        entry_trend_vortex_side=(None if last_entry_meta is None else last_entry_meta.get("trend_vortex_side")),
                        entry_trend_dmi_side=(None if last_entry_meta is None else last_entry_meta.get("trend_dmi_side")),
                        exit_signal_ts=None,
                        gross_ret=float(gross),
                        net_ret=float(net),
                        mfe=float(mfe),
                        mae=float(mae),
                    )
                )
                equity += float(net)
                equity_points.append({"ts": int(ts), "equity": float(equity), "event": str(reason)})

                pos = None
                pending_exit = None
                last_exit_signal_ts = None
            else:
                pos.peak = max(prev_peak, float(h))
                pos.trough = min(prev_trough, float(l))

        # no next open to execute orders
        if i >= end_i:
            continue

        # 4) end-of-bar signals (close(i)) -> schedule for open(i+1)
        win_start_i = max(0, int(i) - int(win) + 1)
        window = df.iloc[win_start_i : i + 1]

        if pos is None:
            if pending_entry is None:
                dec = entry_signal(window, cfg=cfg)
                if dec.side in ("LONG", "SHORT"):
                    pending_entry = {"exec_i": int(i + 1), "side": str(dec.side), "signal_ts": int(ts)}
                    m = dict(dec.meta)
                    m["signal_ts"] = int(ts)
                    last_entry_meta = m
        else:
            if bool(cfg.exit_policy.allow_exit_signal) and pending_exit is None:
                dec = exit_signal(window, cfg=cfg, position_side=pos.side)
                if dec.side in ("LONG", "SHORT"):
                    pending_exit = {"exec_i": int(i + 1), "reason": "SIGNAL", "signal_ts": int(ts)}
                    last_exit_signal_ts = int(ts)

    # Force close at end (EOD)
    if pos is not None:
        last_bar = df.iloc[int(end_i)]
        ts = int(last_bar[ts_col])
        exit_price = float(last_bar[close_col])
        gross = _gross_ret(side=pos.side, entry=pos.entry_price, exit_=exit_price)
        net = gross - (2.0 * fee_rate)

        if pos.side == "LONG":
            mfe = (float(pos.peak) / float(pos.entry_price)) - 1.0
            mae = (float(pos.trough) / float(pos.entry_price)) - 1.0
        else:
            mfe = (float(pos.entry_price) / float(pos.trough)) - 1.0
            mae = (float(pos.entry_price) / float(pos.peak)) - 1.0

        trades.append(
            TradeRecord(
                side=pos.side,
                entry_ts=int(pos.entry_ts),
                exit_ts=int(ts),
                entry_price=float(pos.entry_price),
                exit_price=float(exit_price),
                exit_reason="EOD",
                entry_signal_ts=(None if last_entry_meta is None else int(last_entry_meta.get("signal_ts") or 0) or None),
                entry_signal_cand_ts=(None if last_entry_meta is None else int(last_entry_meta.get("cand_ts") or 0) or None),
                entry_confirmed_series=(
                    None if last_entry_meta is None else "|".join(list(last_entry_meta.get("confirmed_series") or []))
                ),
                entry_trend_filter=(None if last_entry_meta is None else last_entry_meta.get("trend_filter")),
                entry_trend_ok=(None if last_entry_meta is None else last_entry_meta.get("trend_ok")),
                entry_trend_vortex_side=(None if last_entry_meta is None else last_entry_meta.get("trend_vortex_side")),
                entry_trend_dmi_side=(None if last_entry_meta is None else last_entry_meta.get("trend_dmi_side")),
                exit_signal_ts=None,
                gross_ret=float(gross),
                net_ret=float(net),
                mfe=float(mfe),
                mae=float(mae),
            )
        )
        equity += float(net)
        equity_points.append({"ts": int(ts), "equity": float(equity), "event": "EOD"})

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    equity_df = pd.DataFrame(equity_points)

    # drawdown
    equity_df["peak"] = equity_df["equity"].cummax()
    equity_df["dd"] = equity_df["equity"] - equity_df["peak"]
    max_dd = float(equity_df["dd"].min()) if len(equity_df) else 0.0

    equity_end = float(equity_df["equity"].iloc[-1]) if len(equity_df) else 0.0
    if equity_end <= 0.0:
        ratio = 0.0
    else:
        ratio = float("inf") if max_dd == 0.0 else float(equity_end) / abs(float(max_dd))

    wins_mask = pd.to_numeric(trades_df.get("net_ret"), errors="coerce") > 0.0 if len(trades_df) else pd.Series([], dtype=bool)
    losses_mask = pd.to_numeric(trades_df.get("net_ret"), errors="coerce") < 0.0 if len(trades_df) else pd.Series([], dtype=bool)
    n_wins = int(wins_mask.sum()) if len(trades_df) else 0
    n_losses = int(losses_mask.sum()) if len(trades_df) else 0
    winrate = float(n_wins) / float(len(trades_df)) if len(trades_df) else 0.0
    avg_win = float(pd.to_numeric(trades_df.loc[wins_mask, "net_ret"], errors="coerce").mean()) if n_wins else 0.0
    avg_loss = float(pd.to_numeric(trades_df.loc[losses_mask, "net_ret"], errors="coerce").mean()) if n_losses else 0.0

    return {
        "df": df,
        "trades": trades_df,
        "equity": equity_df,
        "summary": {
            "n_trades": int(len(trades_df)),
            "equity_end": float(equity_end),
            "max_dd": float(max_dd),
            "ratio": float(ratio),
            "n_wins": int(n_wins),
            "n_losses": int(n_losses),
            "winrate": float(winrate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
        },
    }
