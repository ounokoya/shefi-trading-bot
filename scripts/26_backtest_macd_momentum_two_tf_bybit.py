from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import numpy as np
import pandas as pd
import requests

from libs.agents.macd_momentum_two_tf_cycle_agent import MacdMomentumTwoTFConfig
from libs.backtest_macd_momentum_two_tf.engine import BacktestMacdMomentumTwoTFConfig, run_backtest_macd_momentum_two_tf
from libs.blocks.add_macd_tv_columns_df import add_macd_tv_columns_df
from libs.indicators.momentum.cci_tv import cci_tv


@dataclass
class BybitKline:
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float


def _interval_to_bybit(interval: str) -> str:
    s = str(interval).strip()
    if not s:
        raise ValueError("interval cannot be empty")

    if s.isdigit():
        return s

    s_lower = s.lower()
    if s_lower in {"d", "1d"}:
        return "D"
    if s_lower in {"w", "1w"}:
        return "W"
    if s == "M" or s_lower in {"1mo", "mo", "1month"}:
        return "M"

    import re

    m = re.fullmatch(r"(\d+)([mhd])", s_lower)
    if not m:
        raise ValueError(f"unsupported interval format: {interval}")

    n = int(m.group(1))
    unit = m.group(2)
    if unit == "m":
        minutes = n
    elif unit == "h":
        minutes = n * 60
    elif unit == "d":
        if n != 1:
            raise ValueError(f"Bybit only supports daily interval as 1d/D, got: {interval}")
        return "D"
    else:
        raise ValueError(f"unsupported interval unit: {unit}")

    allowed_minutes = {1, 3, 5, 15, 30, 60, 120, 240, 360, 720}
    if minutes not in allowed_minutes:
        allowed_str = ", ".join(str(x) for x in sorted(allowed_minutes))
        raise ValueError(
            f"unsupported minute interval for Bybit: {minutes} (from {interval}). Allowed: {allowed_str}"
        )
    return str(minutes)


def _fetch_bybit_klines(
    *,
    symbol: str,
    interval: str,
    limit: int,
    category: str,
    base_url: str,
    timeout_s: float,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> List[BybitKline]:
    url = f"{base_url.rstrip('/')}/v5/market/kline"
    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "limit": str(int(limit)),
    }
    if start_ms is not None:
        params["start"] = str(int(start_ms))
    if end_ms is not None:
        params["end"] = str(int(end_ms))
    r = requests.get(url, params=params, timeout=timeout_s)
    r.raise_for_status()
    payload = r.json()
    if str(payload.get("retCode")) != "0":
        raise RuntimeError(f"Bybit retCode={payload.get('retCode')} retMsg={payload.get('retMsg')}")

    result = payload.get("result") or {}
    rows = result.get("list") or []
    out: List[BybitKline] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 6:
            continue
        ts = int(row[0])
        turnover = float(row[6]) if len(row) >= 7 else 0.0
        out.append(
            BybitKline(
                timestamp_ms=ts,
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                turnover=turnover,
            )
        )

    out.sort(key=lambda k: k.timestamp_ms)
    return out


def _fetch_bybit_klines_range(
    *,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    page_limit: int,
    category: str,
    base_url: str,
    timeout_s: float,
    max_pages: int = 250,
) -> list[BybitKline]:
    if int(page_limit) <= 0:
        page_limit = 200
    if int(page_limit) > 1000:
        page_limit = 1000

    start = int(start_ms)
    cur_end = int(end_ms)
    if int(cur_end) < int(start):
        return []

    seen: set[int] = set()
    out: list[BybitKline] = []
    for _ in range(int(max_pages)):
        page = _fetch_bybit_klines(
            symbol=str(symbol),
            interval=str(interval),
            limit=int(page_limit),
            category=str(category),
            base_url=str(base_url),
            timeout_s=float(timeout_s),
            start_ms=int(start),
            end_ms=int(cur_end),
        )
        if not page:
            break

        oldest = page[0].timestamp_ms
        for k in page:
            if int(k.timestamp_ms) in seen:
                continue
            seen.add(int(k.timestamp_ms))
            if int(start_ms) <= int(k.timestamp_ms) <= int(end_ms):
                out.append(k)

        if int(oldest) <= int(start_ms):
            break
        nxt_end = int(oldest) - 1
        if int(nxt_end) >= int(cur_end):
            break
        cur_end = int(nxt_end)

    out.sort(key=lambda k: int(k.timestamp_ms))
    return out


def _to_df(klines: list[BybitKline]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "ts": [k.timestamp_ms for k in klines],
            "open": [k.open for k in klines],
            "high": [k.high for k in klines],
            "low": [k.low for k in klines],
            "close": [k.close for k in klines],
            "volume": [k.volume for k in klines],
        }
    )
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return df


def _add_ccis(df: pd.DataFrame, *, cci_fast: int, cci_medium: int, cci_slow: int) -> tuple[pd.DataFrame, str, str, str]:
    out = df.copy()
    high = pd.to_numeric(out["high"], errors="coerce").astype(float).tolist()
    low = pd.to_numeric(out["low"], errors="coerce").astype(float).tolist()
    close = pd.to_numeric(out["close"], errors="coerce").astype(float).tolist()

    cci_fast_col = f"cci_{int(cci_fast)}"
    cci_medium_col = f"cci_{int(cci_medium)}"
    cci_slow_col = f"cci_{int(cci_slow)}"

    out[cci_fast_col] = cci_tv(high, low, close, int(cci_fast))
    out[cci_medium_col] = cci_tv(high, low, close, int(cci_medium))
    out[cci_slow_col] = cci_tv(high, low, close, int(cci_slow))

    return out, cci_fast_col, cci_medium_col, cci_slow_col


def _add_stoch(df: pd.DataFrame, *, k_period: int, d_period: int) -> pd.DataFrame:
    out = df.copy()
    k_period2 = int(k_period)
    d_period2 = int(d_period)
    if int(k_period2) < 1:
        raise ValueError("stoch_k must be >= 1")
    if int(d_period2) < 1:
        raise ValueError("stoch_d must be >= 1")

    low_s = pd.to_numeric(out["low"], errors="coerce").astype(float)
    high_s = pd.to_numeric(out["high"], errors="coerce").astype(float)
    close_s = pd.to_numeric(out["close"], errors="coerce").astype(float)

    ll = low_s.rolling(window=k_period2, min_periods=k_period2).min()
    hh = high_s.rolling(window=k_period2, min_periods=k_period2).max()
    denom = (hh - ll).astype(float)
    numer = (close_s - ll).astype(float)

    k = 100.0 * (numer / denom.replace(0.0, np.nan))
    out["stoch_k"] = k
    out["stoch_d"] = out["stoch_k"].rolling(window=d_period2, min_periods=d_period2).mean()
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--symbol", type=str, default="LINKUSDT")
    p.add_argument("--category", type=str, default="linear")
    p.add_argument("--base-url", type=str, default="https://api.bybit.com")
    p.add_argument("--exec-interval", type=str, default="5m")
    p.add_argument("--ctx-interval", type=str, default="15m")
    p.add_argument("--exec-limit", type=int, default=1000)
    p.add_argument("--ctx-limit", type=int, default=1000)

    p.add_argument("--start", type=str, default="2026-01-01")
    p.add_argument("--end", type=str, default="2026-01-12")
    p.add_argument("--warmup-bars", type=int, default=0)

    p.add_argument("--max-signals", type=int, default=0)

    p.add_argument("--fee-rate", type=float, default=0.0015)
    p.add_argument("--exit-mode", type=str, default="exec_tranche_end")

    p.add_argument("--tp-pct", type=float, default=0.0)
    p.add_argument("--trailing-stop-pct", type=float, default=0.0)
    p.add_argument("--sl-pct", type=float, default=0.0)

    p.add_argument("--stoch-k", type=int, default=14)
    p.add_argument("--stoch-d", type=int, default=3)
    p.add_argument("--stoch-high", type=float, default=80.0)
    p.add_argument("--stoch-low", type=float, default=20.0)
    p.add_argument("--stoch-wait-extreme", action="store_true", default=True)
    p.add_argument("--stoch-no-wait-extreme", action="store_false", dest="stoch_wait_extreme")

    p.add_argument("--exec-macd-fast", type=int, default=12)
    p.add_argument("--exec-macd-slow", type=int, default=26)
    p.add_argument("--exec-macd-signal", type=int, default=9)
    p.add_argument("--ctx-macd-fast", type=int, default=12)
    p.add_argument("--ctx-macd-slow", type=int, default=26)
    p.add_argument("--ctx-macd-signal", type=int, default=9)

    p.add_argument("--exec-cci-fast", type=int, default=30)
    p.add_argument("--exec-cci-medium", type=int, default=120)
    p.add_argument("--exec-cci-slow", type=int, default=300)
    p.add_argument("--ctx-cci-fast", type=int, default=30)
    p.add_argument("--ctx-cci-medium", type=int, default=120)
    p.add_argument("--ctx-cci-slow", type=int, default=300)

    p.add_argument("--exec-cci-extreme", type=float, default=100.0)
    p.add_argument("--ctx-cci-extreme", type=float, default=100.0)

    p.add_argument("--min-abs-force-exec", type=float, default=0.0)
    p.add_argument("--min-abs-force-ctx", type=float, default=0.0)

    p.add_argument("--take-exec-cci-extreme-if-ctx-not-extreme", action="store_true")
    p.add_argument("--take-exec-and-ctx-cci-extreme", action="store_true")
    p.add_argument("--signal-on-ctx-flip-if-exec-aligned", action="store_true")

    p.add_argument("--print-top-reasons", type=int, default=10)

    return p


def main() -> None:
    args = _build_parser().parse_args()

    start_ts = int(pd.Timestamp(args.start, tz="UTC").value // 1_000_000)
    end_ts = int((pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)).value // 1_000_000)

    exec_interval_bybit = _interval_to_bybit(str(args.exec_interval))
    ctx_interval_bybit = _interval_to_bybit(str(args.ctx_interval))

    warmup_bars = int(args.warmup_bars)
    if int(warmup_bars) <= 0:
        warmup_bars = max(
            int(args.exec_cci_slow),
            int(args.ctx_cci_slow),
            int(args.exec_macd_slow) * 3,
            int(args.ctx_macd_slow) * 3,
            50,
        )

    def _interval_to_minutes(interval: str) -> int:
        s = str(interval).strip()
        if not s:
            return 0
        if s.isdigit():
            return int(s)
        s_lower = s.lower()
        if s_lower.endswith("m") and s_lower[:-1].isdigit():
            return int(s_lower[:-1])
        if s_lower.endswith("h") and s_lower[:-1].isdigit():
            return int(s_lower[:-1]) * 60
        if s_lower in {"d", "1d"}:
            return 24 * 60
        return 0

    exec_min = _interval_to_minutes(str(args.exec_interval))
    ctx_min = _interval_to_minutes(str(args.ctx_interval))
    exec_warmup_ms = int(warmup_bars) * int(exec_min) * 60_000 if int(exec_min) > 0 else 0
    ctx_warmup_ms = int(warmup_bars) * int(ctx_min) * 60_000 if int(ctx_min) > 0 else 0
    exec_fetch_start_ms = int(max(0, int(start_ts) - int(exec_warmup_ms)))
    ctx_fetch_start_ms = int(max(0, int(start_ts) - int(ctx_warmup_ms)))

    exec_klines = _fetch_bybit_klines_range(
        symbol=str(args.symbol),
        interval=str(exec_interval_bybit),
        start_ms=int(exec_fetch_start_ms),
        end_ms=int(end_ts),
        page_limit=int(args.exec_limit),
        category=str(args.category),
        base_url=str(args.base_url),
        timeout_s=30.0,
    )
    ctx_klines = _fetch_bybit_klines_range(
        symbol=str(args.symbol),
        interval=str(ctx_interval_bybit),
        start_ms=int(ctx_fetch_start_ms),
        end_ms=int(end_ts),
        page_limit=int(args.ctx_limit),
        category=str(args.category),
        base_url=str(args.base_url),
        timeout_s=30.0,
    )

    if not exec_klines:
        raise RuntimeError("no exec klines fetched")
    if not ctx_klines:
        raise RuntimeError("no ctx klines fetched")

    df_exec = _to_df(exec_klines)
    df_ctx = _to_df(ctx_klines)

    df_exec = add_macd_tv_columns_df(
        df_exec,
        close_col="close",
        fast_period=int(args.exec_macd_fast),
        slow_period=int(args.exec_macd_slow),
        signal_period=int(args.exec_macd_signal),
    )
    df_ctx = add_macd_tv_columns_df(
        df_ctx,
        close_col="close",
        fast_period=int(args.ctx_macd_fast),
        slow_period=int(args.ctx_macd_slow),
        signal_period=int(args.ctx_macd_signal),
    )

    df_exec, cci_exec_fast_col, cci_exec_medium_col, cci_exec_slow_col = _add_ccis(
        df_exec,
        cci_fast=int(args.exec_cci_fast),
        cci_medium=int(args.exec_cci_medium),
        cci_slow=int(args.exec_cci_slow),
    )
    df_ctx, cci_ctx_fast_col, cci_ctx_medium_col, cci_ctx_slow_col = _add_ccis(
        df_ctx,
        cci_fast=int(args.ctx_cci_fast),
        cci_medium=int(args.ctx_cci_medium),
        cci_slow=int(args.ctx_cci_slow),
    )

    df_exec = _add_stoch(df_exec, k_period=int(args.stoch_k), d_period=int(args.stoch_d))

    agent_cfg = MacdMomentumTwoTFConfig(
        ts_col="ts",
        dt_col="dt",
        open_col="open",
        high_col="high",
        low_col="low",
        close_col="close",
        hist_col="macd_hist",
        cci_exec_fast_col=str(cci_exec_fast_col),
        cci_exec_medium_col=str(cci_exec_medium_col),
        cci_exec_slow_col=str(cci_exec_slow_col),
        cci_exec_fast_period=int(args.exec_cci_fast),
        cci_exec_medium_period=int(args.exec_cci_medium),
        cci_exec_slow_period=int(args.exec_cci_slow),
        cci_ctx_fast_col=str(cci_ctx_fast_col),
        cci_ctx_medium_col=str(cci_ctx_medium_col),
        cci_ctx_slow_col=str(cci_ctx_slow_col),
        cci_ctx_fast_period=int(args.ctx_cci_fast),
        cci_ctx_medium_period=int(args.ctx_cci_medium),
        cci_ctx_slow_period=int(args.ctx_cci_slow),
        min_abs_force_exec=float(args.min_abs_force_exec),
        min_abs_force_ctx=float(args.min_abs_force_ctx),
        cci_global_extreme_level_exec=float(args.exec_cci_extreme),
        cci_global_extreme_level_ctx=float(args.ctx_cci_extreme),
        take_exec_cci_extreme_if_ctx_not_extreme=bool(args.take_exec_cci_extreme_if_ctx_not_extreme),
        take_exec_and_ctx_cci_extreme=bool(args.take_exec_and_ctx_cci_extreme),
        signal_on_ctx_flip_if_exec_aligned=bool(args.signal_on_ctx_flip_if_exec_aligned),
    )

    bt_cfg = BacktestMacdMomentumTwoTFConfig(
        fee_rate=float(args.fee_rate),
        exit_mode=str(args.exit_mode),
        tp_pct=float(args.tp_pct),
        trailing_stop_pct=float(args.trailing_stop_pct),
        sl_pct=float(args.sl_pct),
        stoch_high=float(args.stoch_high),
        stoch_low=float(args.stoch_low),
        stoch_wait_extreme=bool(args.stoch_wait_extreme),
    )

    out = run_backtest_macd_momentum_two_tf(
        df_exec=df_exec,
        df_ctx=df_ctx,
        agent_cfg=agent_cfg,
        bt_cfg=bt_cfg,
        start_ts=start_ts,
        end_ts=end_ts,
        max_signals=int(args.max_signals),
    )

    print("# Backtest MacdMomentumTwoTF")
    print(f"symbol={args.symbol} exec={args.exec_interval} ctx={args.ctx_interval} start={args.start} end={args.end}")
    print(f"exit_mode={args.exit_mode} fee_rate={args.fee_rate}")
    print("# Agent config")
    print(asdict(agent_cfg))

    print("# Summary")
    for k, v in out["summary"].items():
        print(f"{k}: {v}")

    trades = out["trades"]
    if len(trades):
        print("# Trades (tail)")
        cols = [
            "side",
            "entry_dt",
            "exit_dt",
            "capture_ret",
            "trade_dd",
            "duration_s",
            "entry_reason",
            "entry_signal_kind",
            "exit_reason",
        ]
        cols = [c for c in cols if c in trades.columns]
        print(trades[cols].tail(20).to_string(index=False))

    def _print_top(df: pd.DataFrame, *, title: str) -> None:
        if df is None or (not len(df)):
            return
        n = int(args.print_top_reasons)
        print(f"# {title} (top {n} by pnl_sum)")
        print(df.head(n).to_string(index=False))
        print(f"# {title} (bottom {n} by pnl_sum)")
        print(df.tail(n).to_string(index=False))

    _print_top(out.get("by_entry_reason"), title="By entry_reason")
    _print_top(out.get("by_entry_group"), title="By entry_reason|signal_kind")
    _print_top(out.get("by_exit_reason"), title="By exit_reason")


if __name__ == "__main__":
    main()
