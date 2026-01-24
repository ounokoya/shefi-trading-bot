from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from libs.new_strategie.config import NewStrategieConfig
from libs.new_strategie.signals import StrategySignal


@dataclass(frozen=True)
class FlipTrade:
    side: str
    entry_pos: int
    exit_pos: int
    entry_ts: int
    exit_ts: int
    entry_dt: str
    exit_dt: str
    entry_price: float
    exit_price: float
    exit_reason: str  # FLIP | SL

    entry_signal_kind: str
    entry_signal_type: str
    exit_signal_kind: str | None
    exit_signal_type: str | None

    gross_ret: float


def _dt(ts: int) -> str:
    if int(ts) <= 0:
        return ""
    return str(pd.to_datetime(int(ts), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC"))


def _gross_ret(*, side: str, entry: float, exit_: float) -> float:
    if float(entry) <= 0.0 or (not np.isfinite(float(entry))) or (not np.isfinite(float(exit_))):
        return float("nan")
    if str(side) == "LONG":
        return (float(exit_) / float(entry)) - 1.0
    if str(side) == "SHORT":
        return (float(entry) / float(exit_)) - 1.0
    raise ValueError(f"Unexpected side: {side}")


def _opposite(side: str) -> str:
    return "SHORT" if str(side).upper() == "LONG" else "LONG"


def run_backtest_flip(
    df: pd.DataFrame,
    *,
    signals: list[StrategySignal],
    cfg: NewStrategieConfig,
    sl_pct: float = 0.05,
) -> dict[str, object]:
    for c in (cfg.ts_col, cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col):
        if str(c) not in df.columns:
            raise ValueError(f"Missing required OHLC column: {c}")

    n = int(len(df))
    if n < 2:
        return {"trades": pd.DataFrame([]), "equity": pd.DataFrame([]), "summary": {"n_trades": 0}}

    # Map bar position -> last signal on that bar (if multiple)
    sig_by_pos: dict[int, StrategySignal] = {}
    for s in sorted(signals, key=lambda x: (int(x.pos), int(x.ts))):
        if 0 <= int(s.pos) < n:
            sig_by_pos[int(s.pos)] = s

    ts = pd.to_numeric(df[str(cfg.ts_col)], errors="coerce").astype("Int64").to_numpy()
    o = pd.to_numeric(df[str(cfg.open_col)], errors="coerce").astype(float).to_numpy()
    h = pd.to_numeric(df[str(cfg.high_col)], errors="coerce").astype(float).to_numpy()
    l = pd.to_numeric(df[str(cfg.low_col)], errors="coerce").astype(float).to_numpy()
    c = pd.to_numeric(df[str(cfg.close_col)], errors="coerce").astype(float).to_numpy()

    in_pos = False
    pos_side: str | None = None
    entry_pos: int | None = None
    entry_price: float | None = None
    entry_signal: StrategySignal | None = None

    trades: list[FlipTrade] = []
    equity = 0.0
    equity_points: list[dict[str, object]] = []

    for i in range(0, n):
        ts_i = int(ts[i]) if ts[i] is not None and (not pd.isna(ts[i])) else 0

        # 1) SL intrabar
        if in_pos and pos_side is not None and entry_price is not None and entry_pos is not None:
            stop_price = None
            if str(pos_side) == "LONG":
                sp = float(entry_price) * (1.0 - float(sl_pct))
                if np.isfinite(float(l[i])) and float(l[i]) <= float(sp):
                    stop_price = float(sp)
            else:
                sp = float(entry_price) * (1.0 + float(sl_pct))
                if np.isfinite(float(h[i])) and float(h[i]) >= float(sp):
                    stop_price = float(sp)

            if stop_price is not None:
                exit_px = float(stop_price)
                gross = _gross_ret(side=str(pos_side), entry=float(entry_price), exit_=float(exit_px))
                trades.append(
                    FlipTrade(
                        side=str(pos_side),
                        entry_pos=int(entry_pos),
                        exit_pos=int(i),
                        entry_ts=int(ts[int(entry_pos)]) if ts[int(entry_pos)] is not None and (not pd.isna(ts[int(entry_pos)])) else 0,
                        exit_ts=int(ts_i),
                        entry_dt=_dt(int(ts[int(entry_pos)]) if ts[int(entry_pos)] is not None and (not pd.isna(ts[int(entry_pos)])) else 0),
                        exit_dt=_dt(int(ts_i)),
                        entry_price=float(entry_price),
                        exit_price=float(exit_px),
                        exit_reason="SL",
                        entry_signal_kind=("" if entry_signal is None else str(entry_signal.kind)),
                        entry_signal_type=("" if entry_signal is None else str(entry_signal.meta.get("in_pivot_zone"))),
                        exit_signal_kind=None,
                        exit_signal_type=None,
                        gross_ret=float(gross),
                    )
                )
                if np.isfinite(float(gross)):
                    equity += float(gross)
                equity_points.append({"ts": int(ts_i), "equity": float(equity), "event": "SL"})

                in_pos = False
                pos_side = None
                entry_pos = None
                entry_price = None
                entry_signal = None

        # 2) Signal processing at close
        s = sig_by_pos.get(int(i))
        if s is None:
            continue

        close_px = float(c[i]) if np.isfinite(float(c[i])) else None
        if close_px is None or close_px <= 0.0:
            continue

        if not in_pos:
            # Open in signal direction
            in_pos = True
            pos_side = str(s.side)
            entry_pos = int(i)
            entry_price = float(close_px)
            entry_signal = s
            equity_points.append({"ts": int(ts_i), "equity": float(equity), "event": f"ENTRY_{pos_side}"})
            continue

        # Flip: close current, then open opposite
        if pos_side is None or entry_pos is None or entry_price is None:
            continue

        gross = _gross_ret(side=str(pos_side), entry=float(entry_price), exit_=float(close_px))
        trades.append(
            FlipTrade(
                side=str(pos_side),
                entry_pos=int(entry_pos),
                exit_pos=int(i),
                entry_ts=int(ts[int(entry_pos)]) if ts[int(entry_pos)] is not None and (not pd.isna(ts[int(entry_pos)])) else 0,
                exit_ts=int(ts_i),
                entry_dt=_dt(int(ts[int(entry_pos)]) if ts[int(entry_pos)] is not None and (not pd.isna(ts[int(entry_pos)])) else 0),
                exit_dt=_dt(int(ts_i)),
                entry_price=float(entry_price),
                exit_price=float(close_px),
                exit_reason="FLIP",
                entry_signal_kind=("" if entry_signal is None else str(entry_signal.kind)),
                entry_signal_type=("" if entry_signal is None else str(entry_signal.meta.get("in_pivot_zone"))),
                exit_signal_kind=str(s.kind),
                exit_signal_type=str(s.meta.get("in_pivot_zone")),
                gross_ret=float(gross),
            )
        )
        if np.isfinite(float(gross)):
            equity += float(gross)
        equity_points.append({"ts": int(ts_i), "equity": float(equity), "event": "FLIP"})

        # open opposite
        pos_side = _opposite(str(pos_side))
        entry_pos = int(i)
        entry_price = float(close_px)
        entry_signal = s
        equity_points.append({"ts": int(ts_i), "equity": float(equity), "event": f"ENTRY_{pos_side}"})

    # EOD close
    if in_pos and pos_side is not None and entry_price is not None and entry_pos is not None:
        last_i = int(n - 1)
        ts_last = int(ts[last_i]) if ts[last_i] is not None and (not pd.isna(ts[last_i])) else 0
        exit_px = float(c[last_i]) if np.isfinite(float(c[last_i])) else float(o[last_i])
        gross = _gross_ret(side=str(pos_side), entry=float(entry_price), exit_=float(exit_px))
        trades.append(
            FlipTrade(
                side=str(pos_side),
                entry_pos=int(entry_pos),
                exit_pos=int(last_i),
                entry_ts=int(ts[int(entry_pos)]) if ts[int(entry_pos)] is not None and (not pd.isna(ts[int(entry_pos)])) else 0,
                exit_ts=int(ts_last),
                entry_dt=_dt(int(ts[int(entry_pos)]) if ts[int(entry_pos)] is not None and (not pd.isna(ts[int(entry_pos)])) else 0),
                exit_dt=_dt(int(ts_last)),
                entry_price=float(entry_price),
                exit_price=float(exit_px),
                exit_reason="EOD",
                entry_signal_kind=("" if entry_signal is None else str(entry_signal.kind)),
                entry_signal_type=("" if entry_signal is None else str(entry_signal.meta.get("in_pivot_zone"))),
                exit_signal_kind=None,
                exit_signal_type=None,
                gross_ret=float(gross),
            )
        )
        if np.isfinite(float(gross)):
            equity += float(gross)
        equity_points.append({"ts": int(ts_last), "equity": float(equity), "event": "EOD"})

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    eq_df = pd.DataFrame(equity_points)

    if len(eq_df):
        eq_df = eq_df.sort_values("ts").reset_index(drop=True)
        eq_df["peak"] = eq_df["equity"].cummax()
        eq_df["dd"] = eq_df["equity"] - eq_df["peak"]
        max_dd = float(eq_df["dd"].min())
        equity_end = float(eq_df["equity"].iloc[-1])
    else:
        max_dd = 0.0
        equity_end = 0.0

    n_trades = int(len(trades_df))
    winrate = 0.0
    avg_ret = 0.0
    if n_trades:
        rets = pd.to_numeric(trades_df["gross_ret"], errors="coerce")
        winrate = float((rets > 0.0).mean())
        avg_ret = float(rets.mean())

    return {
        "trades": trades_df,
        "equity": eq_df,
        "summary": {
            "n_trades": int(n_trades),
            "equity_end": float(equity_end),
            "max_dd": float(max_dd),
            "winrate": float(winrate),
            "avg_ret": float(avg_ret),
            "sl_pct": float(sl_pct),
        },
    }
