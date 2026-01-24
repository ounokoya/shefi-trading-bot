from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.agents.macd_momentum_two_tf_cycle_agent import MacdMomentiumTwoTFCycleAgent, MacdMomentumTwoTFConfig
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


def _parse_utc_day_start_ms(s: str) -> int:
    ts = pd.Timestamp(str(s), tz="UTC")
    return int(ts.value // 1_000_000)


def _parse_utc_day_end_ms(s: str) -> int:
    ts = pd.Timestamp(str(s), tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    return int(ts.value // 1_000_000)


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


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--symbol", default="LINKUSDT")
    ap.add_argument("--category", default="linear")
    ap.add_argument("--base-url", default="https://api.bybit.com")

    ap.add_argument("--exec-interval", default="5m")
    ap.add_argument("--ctx-interval", default="15m")
    ap.add_argument("--exec-limit", type=int, default=1000)
    ap.add_argument("--ctx-limit", type=int, default=1000)

    ap.add_argument("--start-date", default="2026-01-01")
    ap.add_argument("--end-date", default="2026-01-12")
    ap.add_argument("--warmup-bars", type=int, default=0)

    ap.add_argument("--macd-fast", type=int, default=12)
    ap.add_argument("--macd-slow", type=int, default=26)
    ap.add_argument("--macd-signal", type=int, default=9)
    ap.add_argument("--exec-macd-fast", type=int, default=None)
    ap.add_argument("--exec-macd-slow", type=int, default=None)
    ap.add_argument("--exec-macd-signal", type=int, default=None)
    ap.add_argument("--ctx-macd-fast", type=int, default=None)
    ap.add_argument("--ctx-macd-slow", type=int, default=None)
    ap.add_argument("--ctx-macd-signal", type=int, default=None)

    ap.add_argument("--cci-fast", type=int, default=30)
    ap.add_argument("--cci-medium", type=int, default=120)
    ap.add_argument("--cci-slow", type=int, default=300)
    ap.add_argument("--exec-cci-fast", type=int, default=32)
    ap.add_argument("--exec-cci-medium", type=int, default=96)
    ap.add_argument("--exec-cci-slow", type=int, default=288)
    ap.add_argument("--ctx-cci-fast", type=int, default=32)
    ap.add_argument("--ctx-cci-medium", type=int, default=96)
    ap.add_argument("--ctx-cci-slow", type=int, default=288)

    ap.add_argument("--min-abs-force-ctx", type=float, default=0.0)
    ap.add_argument("--min-abs-force-exec", type=float, default=0.0)
    ap.add_argument("--cci-level-ctx", type=float, default=100.0)
    ap.add_argument("--cci-level-exec", type=float, default=100.0)

    ap.add_argument("--take-exec-cci-extreme-if-ctx-not-extreme", action="store_true")
    ap.add_argument("--take-exec-and-ctx-cci-extreme", action="store_true")

    ap.add_argument("--signal-on-ctx-flip-if-exec-aligned", action="store_true")

    ap.add_argument("--mode", choices=["analyze", "current"], default="analyze")
    ap.add_argument("--max-signals", type=int, default=0)
    ap.add_argument("--show-rejected", action="store_true")

    args = ap.parse_args()

    exec_macd_fast = int(args.exec_macd_fast) if args.exec_macd_fast is not None else int(args.macd_fast)
    exec_macd_slow = int(args.exec_macd_slow) if args.exec_macd_slow is not None else int(args.macd_slow)
    exec_macd_signal = int(args.exec_macd_signal) if args.exec_macd_signal is not None else int(args.macd_signal)
    ctx_macd_fast = int(args.ctx_macd_fast) if args.ctx_macd_fast is not None else int(args.macd_fast)
    ctx_macd_slow = int(args.ctx_macd_slow) if args.ctx_macd_slow is not None else int(args.macd_slow)
    ctx_macd_signal = int(args.ctx_macd_signal) if args.ctx_macd_signal is not None else int(args.macd_signal)

    exec_cci_fast = int(args.exec_cci_fast) if args.exec_cci_fast is not None else int(args.cci_fast)
    exec_cci_medium = int(args.exec_cci_medium) if args.exec_cci_medium is not None else int(args.cci_medium)
    exec_cci_slow = int(args.exec_cci_slow) if args.exec_cci_slow is not None else int(args.cci_slow)
    ctx_cci_fast = int(args.ctx_cci_fast) if args.ctx_cci_fast is not None else int(args.cci_fast)
    ctx_cci_medium = int(args.ctx_cci_medium) if args.ctx_cci_medium is not None else int(args.cci_medium)
    ctx_cci_slow = int(args.ctx_cci_slow) if args.ctx_cci_slow is not None else int(args.cci_slow)

    exec_interval_bybit = _interval_to_bybit(str(args.exec_interval))
    ctx_interval_bybit = _interval_to_bybit(str(args.ctx_interval))

    start_ms = _parse_utc_day_start_ms(str(args.start_date))
    end_ms = _parse_utc_day_end_ms(str(args.end_date))

    exec_min = _interval_to_minutes(str(args.exec_interval))
    ctx_min = _interval_to_minutes(str(args.ctx_interval))
    warmup_bars = int(args.warmup_bars)
    if int(warmup_bars) <= 0:
        warmup_bars = max(int(exec_cci_slow), int(ctx_cci_slow), int(exec_macd_slow) * 3, int(ctx_macd_slow) * 3, 50)

    exec_warmup_ms = int(warmup_bars) * int(exec_min) * 60_000 if int(exec_min) > 0 else 0
    ctx_warmup_ms = int(warmup_bars) * int(ctx_min) * 60_000 if int(ctx_min) > 0 else 0
    exec_fetch_start_ms = int(max(0, int(start_ms) - int(exec_warmup_ms)))
    ctx_fetch_start_ms = int(max(0, int(start_ms) - int(ctx_warmup_ms)))

    exec_klines = _fetch_bybit_klines_range(
        symbol=str(args.symbol),
        interval=str(exec_interval_bybit),
        start_ms=int(exec_fetch_start_ms),
        end_ms=int(end_ms),
        page_limit=int(args.exec_limit),
        category=str(args.category),
        base_url=str(args.base_url),
        timeout_s=30.0,
    )
    ctx_klines = _fetch_bybit_klines_range(
        symbol=str(args.symbol),
        interval=str(ctx_interval_bybit),
        start_ms=int(ctx_fetch_start_ms),
        end_ms=int(end_ms),
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
        fast_period=int(exec_macd_fast),
        slow_period=int(exec_macd_slow),
        signal_period=int(exec_macd_signal),
    )
    df_ctx = add_macd_tv_columns_df(
        df_ctx,
        close_col="close",
        fast_period=int(ctx_macd_fast),
        slow_period=int(ctx_macd_slow),
        signal_period=int(ctx_macd_signal),
    )

    df_exec, cci_fast_col, cci_medium_col, cci_slow_col = _add_ccis(
        df_exec,
        cci_fast=int(exec_cci_fast),
        cci_medium=int(exec_cci_medium),
        cci_slow=int(exec_cci_slow),
    )
    df_ctx, cci_ctx_fast_col, cci_ctx_medium_col, cci_ctx_slow_col = _add_ccis(
        df_ctx,
        cci_fast=int(ctx_cci_fast),
        cci_medium=int(ctx_cci_medium),
        cci_slow=int(ctx_cci_slow),
    )

    cfg = MacdMomentumTwoTFConfig(
        ts_col="ts",
        dt_col="dt",
        open_col="open",
        high_col="high",
        low_col="low",
        close_col="close",
        hist_col="macd_hist",
        cci_exec_fast_col=str(cci_fast_col),
        cci_exec_medium_col=str(cci_medium_col),
        cci_exec_slow_col=str(cci_slow_col),
        cci_exec_fast_period=int(exec_cci_fast),
        cci_exec_medium_period=int(exec_cci_medium),
        cci_exec_slow_period=int(exec_cci_slow),
        cci_ctx_fast_col=str(cci_ctx_fast_col),
        cci_ctx_medium_col=str(cci_ctx_medium_col),
        cci_ctx_slow_col=str(cci_ctx_slow_col),
        cci_ctx_fast_period=int(ctx_cci_fast),
        cci_ctx_medium_period=int(ctx_cci_medium),
        cci_ctx_slow_period=int(ctx_cci_slow),
        min_abs_force_ctx=float(args.min_abs_force_ctx),
        min_abs_force_exec=float(args.min_abs_force_exec),
        cci_global_extreme_level_ctx=float(args.cci_level_ctx),
        cci_global_extreme_level_exec=float(args.cci_level_exec),
        take_exec_cci_extreme_if_ctx_not_extreme=bool(args.take_exec_cci_extreme_if_ctx_not_extreme),
        take_exec_and_ctx_cci_extreme=bool(args.take_exec_and_ctx_cci_extreme),
        signal_on_ctx_flip_if_exec_aligned=bool(args.signal_on_ctx_flip_if_exec_aligned),
    )
    agent = MacdMomentiumTwoTFCycleAgent(cfg=cfg)

    ans = agent.answer(
        question={
            "kind": str(args.mode),
            "max_signals": int(args.max_signals),
        },
        df=df_exec,
        df_ctx=df_ctx,
    )

    start_dt_str = pd.to_datetime(int(start_ms), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")
    end_dt_str = pd.to_datetime(int(end_ms), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(
        f"Fetched exec={len(df_exec)} klines({args.exec_interval}) ctx={len(df_ctx)} klines({args.ctx_interval}) "
        f"for {args.symbol} range=[{start_dt_str}..{end_dt_str}] warmup_bars={int(warmup_bars)}"
    )

    if str(args.mode) == "current":
        m = ans.get("metric")
        if not m:
            print("(none)")
            return 0
        print(
            "CURRENT"
            + f" side={m.get('side')} status={m.get('status')} reason={m.get('reason')}"
            + f" signal_kind={m.get('signal_kind')} signal_dt='{m.get('signal_dt')}'"
            + f" exec_tranche_id={m.get('exec_tranche_id')} exec_sign={m.get('exec_tranche_sign')}"
            + f" ctx_tranche_id={m.get('ctx_tranche_id')} ctx_sign={m.get('ctx_tranche_sign')}"
            + f" exec_force={m.get('exec_force_mean_abs')} ctx_force={m.get('ctx_force_mean_abs')}"
            + f" cci_ctx_last={m.get('cci_ctx_last_extreme')} cci_exec_last={m.get('cci_exec_last_extreme')}"
            + f" entry_dt='{m.get('entry_dt')}' entry={m.get('entry')}"
        )
        return 0

    metrics0 = list(ans.get("metrics") or [])
    metrics = []
    for r in metrics0:
        ts0 = r.get("signal_ts")
        try:
            ts_i = int(ts0) if ts0 is not None else None
        except Exception:
            ts_i = None
        if ts_i is None:
            continue
        if int(start_ms) <= int(ts_i) <= int(end_ms):
            metrics.append(r)
    if not metrics:
        print("(none)")
        return 0

    n_accept = 0
    n_reject = 0
    for r in metrics:
        status = str(r.get("status") or "")
        if status == "ACCEPT":
            n_accept += 1
        else:
            n_reject += 1

    print(f"Signals: ACCEPT={n_accept} REJECT={n_reject} total={len(metrics)}")

    for r in metrics:
        status = str(r.get("status") or "")
        if status != "ACCEPT" and (not bool(args.show_rejected)):
            continue

        entry = r.get("entry")
        if entry is not None:
            try:
                entry_f = float(entry)
            except Exception:
                entry_f = float("nan")
        else:
            entry_f = float("nan")

        entry_s = f"{entry_f:.6f}" if math.isfinite(float(entry_f)) else ""

        print(
            ("ACCEPT" if status == "ACCEPT" else "REJECT")
            + f" side={r.get('side')} reason={r.get('reason')}"
            + f" signal_kind={r.get('signal_kind')} signal_dt='{r.get('signal_dt')}'"
            + f" exec_tranche_id={r.get('exec_tranche_id')} exec_sign={r.get('exec_tranche_sign')}"
            + f" ctx_tranche_id={r.get('ctx_tranche_id')} ctx_sign={r.get('ctx_tranche_sign')}"
            + f" exec_force={r.get('exec_force_mean_abs')} ctx_force={r.get('ctx_force_mean_abs')}"
            + f" cci_ctx_last={r.get('cci_ctx_last_extreme')} cci_exec_last={r.get('cci_exec_last_extreme')}"
            + f" entry_dt='{r.get('entry_dt')}' entry={entry_s}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
