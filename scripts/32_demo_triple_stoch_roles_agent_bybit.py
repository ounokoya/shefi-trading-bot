from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.agents.triple_stoch_roles_agent import (  # noqa: E402
    TripleStochLevelConfig,
    TripleStochRolesAgent,
    TripleStochRolesAgentConfig,
)


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


def _add_stoch_cols(
    df: pd.DataFrame,
    *,
    k_period: int,
    d_period: int,
    out_k_col: str,
    out_d_col: str,
) -> pd.DataFrame:
    out = df.copy()
    kp = int(k_period)
    dp = int(d_period)
    if kp < 1:
        raise ValueError("stoch_k must be >= 1")
    if dp < 1:
        raise ValueError("stoch_d must be >= 1")

    low_s = pd.to_numeric(out["low"], errors="coerce").astype(float)
    high_s = pd.to_numeric(out["high"], errors="coerce").astype(float)
    close_s = pd.to_numeric(out["close"], errors="coerce").astype(float)

    ll = low_s.rolling(window=kp, min_periods=kp).min()
    hh = high_s.rolling(window=kp, min_periods=kp).max()
    denom = (hh - ll).astype(float)
    numer = (close_s - ll).astype(float)

    k = 100.0 * (numer / denom.replace(0.0, np.nan))
    d = k.rolling(window=dp, min_periods=dp).mean()

    out[str(out_k_col)] = k
    out[str(out_d_col)] = d
    return out


def _print_current(metric: dict) -> None:
    print("\n=== CURRENT ===")
    print(f"dt={metric.get('dt')} pos={metric.get('pos')} macro={metric.get('macro_side')} sign={metric.get('macro_sign')}")
    for name in ("slow", "medium", "fast"):
        print(
            " | ".join(
                [
                    f"{name}",
                    f"regime={metric.get(f'{name}_regime_sign')}",
                    f"mom={metric.get(f'{name}_momentum_sign')}",
                    f"resp={metric.get(f'{name}_is_respiration')}",
                    f"ext={metric.get(f'{name}_extreme_sign')}",
                    f"dwell={metric.get(f'{name}_extreme_dwell')}",
                    f"force={metric.get(f'{name}_force')}",
                    f"rising={metric.get(f'{name}_force_rising')}",
                ]
            )
        )
    print(
        " | ".join(
            [
                f"trigger_level={metric.get('trigger_level')}",
                f"trigger_long={metric.get('trigger_long')}",
                f"trigger_short={metric.get('trigger_short')}",
            ]
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--category", default="linear")
    ap.add_argument("--base-url", default="https://api.bybit.com")

    ap.add_argument("--k-fast", type=int, default=14)
    ap.add_argument("--k-medium", type=int, default=28)
    ap.add_argument("--k-slow", type=int, default=56)
    ap.add_argument("--d-period", type=int, default=3)

    ap.add_argument("--regime-pivot", type=float, default=50.0)
    ap.add_argument("--extreme-high", type=float, default=80.0)
    ap.add_argument("--extreme-low", type=float, default=20.0)

    ap.add_argument("--slope-window-fast", type=int, default=6)
    ap.add_argument("--slope-window-medium", type=int, default=8)
    ap.add_argument("--slope-window-slow", type=int, default=12)

    ap.add_argument("--macro-mode", choices=["slow_regime", "slow_regime_and_momentum"], default="slow_regime")
    ap.add_argument("--style", choices=["", "scalp", "swing", "position"], default="")
    ap.add_argument("--entry-trigger-level", choices=["", "slow", "medium", "fast"], default="")
    ap.add_argument("--trigger-mode", choices=["kd_cross", "regime_cross", "extreme_exit", "any"], default="kd_cross")
    ap.add_argument("--require-trigger-in-macro-dir", action="store_true", default=True)
    ap.add_argument("--no-require-trigger-in-macro-dir", action="store_false", dest="require_trigger_in_macro_dir")

    ap.add_argument("--reject-when-all-three-respire", action="store_true", default=False)

    ap.add_argument("--force-mode", choices=["abs_spread", "abs_k_slope"], default="abs_spread")
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

    df = _add_stoch_cols(df, k_period=int(args.k_slow), d_period=int(args.d_period), out_k_col="stoch_k_slow", out_d_col="stoch_d_slow")
    df = _add_stoch_cols(df, k_period=int(args.k_medium), d_period=int(args.d_period), out_k_col="stoch_k_medium", out_d_col="stoch_d_medium")
    df = _add_stoch_cols(df, k_period=int(args.k_fast), d_period=int(args.d_period), out_k_col="stoch_k_fast", out_d_col="stoch_d_fast")

    slow_lv = TripleStochLevelConfig(
        k_col="stoch_k_slow",
        d_col="stoch_d_slow",
        k_period=int(args.k_slow),
        d_period=int(args.d_period),
        regime_pivot=float(args.regime_pivot),
        extreme_high=float(args.extreme_high),
        extreme_low=float(args.extreme_low),
        slope_window=int(args.slope_window_slow),
        force_mode=str(args.force_mode),
        min_abs_force=float(args.slow_min_force),
        force_rising_bars=int(args.force_rising_bars),
        allow_trade_when_respiration=bool(args.allow_trade_when_respiration),
    )
    medium_lv = TripleStochLevelConfig(
        k_col="stoch_k_medium",
        d_col="stoch_d_medium",
        k_period=int(args.k_medium),
        d_period=int(args.d_period),
        regime_pivot=float(args.regime_pivot),
        extreme_high=float(args.extreme_high),
        extreme_low=float(args.extreme_low),
        slope_window=int(args.slope_window_medium),
        force_mode=str(args.force_mode),
        min_abs_force=float(args.medium_min_force),
        force_rising_bars=int(args.force_rising_bars),
        allow_trade_when_respiration=bool(args.allow_trade_when_respiration),
    )
    fast_lv = TripleStochLevelConfig(
        k_col="stoch_k_fast",
        d_col="stoch_d_fast",
        k_period=int(args.k_fast),
        d_period=int(args.d_period),
        regime_pivot=float(args.regime_pivot),
        extreme_high=float(args.extreme_high),
        extreme_low=float(args.extreme_low),
        slope_window=int(args.slope_window_fast),
        force_mode=str(args.force_mode),
        min_abs_force=float(args.fast_min_force),
        force_rising_bars=int(args.force_rising_bars),
        allow_trade_when_respiration=bool(args.allow_trade_when_respiration),
    )

    agent_cfg = TripleStochRolesAgentConfig(
        ts_col="ts",
        dt_col="dt",
        high_col="high",
        low_col="low",
        close_col="close",
        slow=slow_lv,
        medium=medium_lv,
        fast=fast_lv,
        macro_mode=str(args.macro_mode),
        style=str(args.style),
        entry_trigger_level=str(args.entry_trigger_level),
        trigger_mode=str(args.trigger_mode),
        require_trigger_in_macro_dir=bool(args.require_trigger_in_macro_dir),
        reject_when_all_three_respire=bool(args.reject_when_all_three_respire),
    )
    agent = TripleStochRolesAgent(cfg=agent_cfg)

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
        print(
            " | ".join(
                [
                    f"dt={e.get('dt')}",
                    f"side={e.get('side')}",
                    f"pos={e.get('pos')}",
                    f"macro_sign={meta.get('macro_sign')}",
                    f"trigger_level={meta.get('trigger_level')}",
                    f"trigger_mode={meta.get('trigger_mode')}",
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
