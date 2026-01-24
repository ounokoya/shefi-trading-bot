from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.agents.triple_macd_roles_agent import (  # noqa: E402
    TripleMacdLevelConfig,
    TripleMacdRolesAgent,
    TripleMacdRolesAgentConfig,
)
from libs.blocks.add_macd_tv_columns_df import add_macd_tv_columns_df  # noqa: E402


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
) -> List[BybitKline]:
    url = f"{base_url.rstrip('/')}/v5/market/kline"
    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "limit": str(int(limit)),
    }
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


def _print_current(metric: dict) -> None:
    print("\n=== CURRENT ===")
    print(f"dt={metric.get('dt')} pos={metric.get('pos')} macro={metric.get('macro_side')} sign={metric.get('macro_sign')}")
    for name in ("slow", "medium", "fast"):
        print(
            " | ".join(
                [
                    f"{name}",
                    f"zone={metric.get(f'{name}_zone_sign')}",
                    f"hist={metric.get(f'{name}_hist_sign')}",
                    f"resp={metric.get(f'{name}_is_respiration')}",
                    f"force={metric.get(f'{name}_force')}",
                    f"rising={metric.get(f'{name}_force_rising')}",
                ]
            )
        )
    print(
        " | ".join(
            [
                f"trigger_level={metric.get('trigger_level')}",
                f"cross_up={metric.get('trigger_cross_up')}",
                f"cross_down={metric.get('trigger_cross_down')}",
            ]
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="LINKUSDT")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--limit", type=int, default=1200)
    ap.add_argument("--category", default="linear")
    ap.add_argument("--base-url", default="https://api.bybit.com")

    ap.add_argument("--macd-fast-fast", type=int, default=4)
    ap.add_argument("--macd-fast-slow", type=int, default=12)
    ap.add_argument("--macd-fast-signal", type=int, default=4)

    ap.add_argument("--macd-medium-fast", type=int, default=12)
    ap.add_argument("--macd-medium-slow", type=int, default=36)
    ap.add_argument("--macd-medium-signal", type=int, default=9)

    ap.add_argument("--macd-slow-fast", type=int, default=24)
    ap.add_argument("--macd-slow-slow", type=int, default=72)
    ap.add_argument("--macd-slow-signal", type=int, default=18)

    ap.add_argument("--macro-mode", choices=["slow_zone", "slow_hist", "slow_zone_and_hist"], default="slow_zone")
    ap.add_argument("--entry-style", choices=["default", "simple_alignment"], default="default")
    ap.add_argument("--require-hists-rising-on-entry", action="store_true", default=True)
    ap.add_argument(
        "--no-require-hists-rising-on-entry", action="store_false", dest="require_hists_rising_on_entry"
    )
    ap.add_argument("--style", choices=["", "scalp", "swing", "position"], default="")
    ap.add_argument("--entry-trigger-level", choices=["", "slow", "medium", "fast"], default="")
    ap.add_argument("--require-trigger-in-macro-dir", action="store_true", default=True)
    ap.add_argument("--no-require-trigger-in-macro-dir", action="store_false", dest="require_trigger_in_macro_dir")

    ap.add_argument("--reject-when-all-three-respire", action="store_true", default=False)

    ap.add_argument("--slow-min-force", type=float, default=0.0)
    ap.add_argument("--medium-min-force", type=float, default=0.0)
    ap.add_argument("--fast-min-force", type=float, default=0.0)

    ap.add_argument("--force-rising-bars", type=int, default=2)

    ap.add_argument("--allow-trade-when-respiration", action="store_true", default=True)
    ap.add_argument("--no-allow-trade-when-respiration", action="store_false", dest="allow_trade_when_respiration")

    ap.add_argument("--mode", choices=["current", "entries"], default="current")
    ap.add_argument("--max-events", type=int, default=30)

    args = ap.parse_args()

    interval_bybit = _interval_to_bybit(str(args.interval))
    klines = _fetch_bybit_klines(
        symbol=str(args.symbol),
        interval=str(interval_bybit),
        limit=int(args.limit),
        category=str(args.category),
        base_url=str(args.base_url),
        timeout_s=30.0,
    )
    if not klines:
        raise RuntimeError("no klines fetched")

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

    df = add_macd_tv_columns_df(
        df,
        close_col="close",
        fast_period=int(args.macd_slow_fast),
        slow_period=int(args.macd_slow_slow),
        signal_period=int(args.macd_slow_signal),
        out_line_col="macd_line_slow",
        out_signal_col="macd_signal_slow",
        out_hist_col="macd_hist_slow",
    )
    df = add_macd_tv_columns_df(
        df,
        close_col="close",
        fast_period=int(args.macd_medium_fast),
        slow_period=int(args.macd_medium_slow),
        signal_period=int(args.macd_medium_signal),
        out_line_col="macd_line_medium",
        out_signal_col="macd_signal_medium",
        out_hist_col="macd_hist_medium",
    )
    df = add_macd_tv_columns_df(
        df,
        close_col="close",
        fast_period=int(args.macd_fast_fast),
        slow_period=int(args.macd_fast_slow),
        signal_period=int(args.macd_fast_signal),
        out_line_col="macd_line_fast",
        out_signal_col="macd_signal_fast",
        out_hist_col="macd_hist_fast",
    )

    slow_lv = TripleMacdLevelConfig(
        line_col="macd_line_slow",
        signal_col="macd_signal_slow",
        hist_col="macd_hist_slow",
        min_abs_force=float(args.slow_min_force),
        force_rising_bars=int(args.force_rising_bars),
        allow_trade_when_respiration=bool(args.allow_trade_when_respiration),
    )
    medium_lv = TripleMacdLevelConfig(
        line_col="macd_line_medium",
        signal_col="macd_signal_medium",
        hist_col="macd_hist_medium",
        min_abs_force=float(args.medium_min_force),
        force_rising_bars=int(args.force_rising_bars),
        allow_trade_when_respiration=bool(args.allow_trade_when_respiration),
    )
    fast_lv = TripleMacdLevelConfig(
        line_col="macd_line_fast",
        signal_col="macd_signal_fast",
        hist_col="macd_hist_fast",
        min_abs_force=float(args.fast_min_force),
        force_rising_bars=int(args.force_rising_bars),
        allow_trade_when_respiration=bool(args.allow_trade_when_respiration),
    )

    agent_cfg = TripleMacdRolesAgentConfig(
        ts_col="ts",
        dt_col="dt",
        close_col="close",
        slow=slow_lv,
        medium=medium_lv,
        fast=fast_lv,
        macro_mode=str(args.macro_mode),
        entry_style=str(args.entry_style),
        require_hists_rising_on_entry=bool(args.require_hists_rising_on_entry),
        style=str(args.style),
        entry_trigger_level=str(args.entry_trigger_level),
        require_trigger_in_macro_dir=bool(args.require_trigger_in_macro_dir),
        reject_when_all_three_respire=bool(args.reject_when_all_three_respire),
    )
    agent = TripleMacdRolesAgent(cfg=agent_cfg)

    print(f"Fetched {len(df)} klines for {args.symbol} {args.interval}")

    if str(args.mode) == "current":
        ans = agent.answer(question={"kind": "current"}, df=df)
        m = ans.get("metric")
        if not m:
            print("None")
            return 0
        _print_current(m)
        return 0

    ans = agent.answer(question={"kind": "entries", "max_events": int(args.max_events)}, df=df)
    events = ans.get("events") or []
    print("\n=== ENTRIES (tail) ===")
    for e in events[-int(args.max_events) :]:
        meta = e.get("meta") or {}
        entry_style = str(meta.get("entry_style") or "")
        dbg = ""
        if entry_style == "simple_alignment":
            dbg = " | ".join(
                [
                    f"fast={meta.get('fast_hist_sign_prev')}->{meta.get('fast_hist_sign')}",
                    f"medium={meta.get('medium_hist_sign_prev')}->{meta.get('medium_hist_sign')}",
                    f"slow={meta.get('slow_hist_sign_prev')}->{meta.get('slow_hist_sign')}",
                    f"medium_stable={meta.get('medium_stable')}",
                    f"slow_stable={meta.get('slow_stable')}",
                ]
            )
        print(
            " | ".join(
                [
                    f"dt={e.get('dt')}",
                    f"side={e.get('side')}",
                    f"pos={e.get('pos')}",
                    f"macro_sign={meta.get('macro_sign')}",
                    f"trigger_level={meta.get('trigger_level')}",
                    (f"entry_style={entry_style}" if entry_style else ""),
                    dbg,
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
