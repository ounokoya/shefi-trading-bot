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

from libs.agents.triple_cci_tranche_agent import TripleCciTrancheAgent, TripleCciTrancheAgentConfig
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--category", default="linear")
    ap.add_argument("--base-url", default="https://api.bybit.com")

    ap.add_argument("--macd-fast", type=int, default=12)
    ap.add_argument("--macd-slow", type=int, default=26)
    ap.add_argument("--macd-signal", type=int, default=9)

    ap.add_argument("--cci-fast", type=int, default=30)
    ap.add_argument("--cci-medium", type=int, default=120)
    ap.add_argument("--cci-slow", type=int, default=300)

    ap.add_argument("--min-tranche-len", type=int, default=5)
    ap.add_argument("--min-abs-cci-global", type=float, default=100.0)

    ap.add_argument("--max-tranches", type=int, default=12)
    ap.add_argument("--top-n", type=int, default=10)

    ap.add_argument("--mode", choices=["analyze", "current"], default="analyze")

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
        fast_period=int(args.macd_fast),
        slow_period=int(args.macd_slow),
        signal_period=int(args.macd_signal),
    )

    high = pd.to_numeric(df["high"], errors="coerce").astype(float).tolist()
    low = pd.to_numeric(df["low"], errors="coerce").astype(float).tolist()
    close = pd.to_numeric(df["close"], errors="coerce").astype(float).tolist()

    cci_fast_col = f"cci_{int(args.cci_fast)}"
    cci_medium_col = f"cci_{int(args.cci_medium)}"
    cci_slow_col = f"cci_{int(args.cci_slow)}"

    df[cci_fast_col] = cci_tv(high, low, close, int(args.cci_fast))
    df[cci_medium_col] = cci_tv(high, low, close, int(args.cci_medium))
    df[cci_slow_col] = cci_tv(high, low, close, int(args.cci_slow))

    agent_cfg = TripleCciTrancheAgentConfig(
        ts_col="ts",
        high_col="high",
        low_col="low",
        close_col="close",
        hist_col="macd_hist",
        cci_fast_col=str(cci_fast_col),
        cci_medium_col=str(cci_medium_col),
        cci_slow_col=str(cci_slow_col),
        cci_fast_period=int(args.cci_fast),
        cci_medium_period=int(args.cci_medium),
        cci_slow_period=int(args.cci_slow),
        min_tranche_len=int(args.min_tranche_len),
        min_abs_cci_global=float(args.min_abs_cci_global),
    )
    agent = TripleCciTrancheAgent(cfg=agent_cfg)

    ans = agent.answer(
        question={
            "kind": str(args.mode),
            "max_tranches": int(args.max_tranches),
        },
        df=df,
    )

    print(f"Fetched {len(df)} klines for {args.symbol} {args.interval}")

    def _print_row(r: dict) -> None:
        print(
            " | ".join(
                [
                    f"id={r.get('tranche_id')}",
                    f"type={r.get('tranche_type')}",
                    f"sign={r.get('tranche_sign')}",
                    f"len={r.get('tranche_len')}",
                    f"idx=[{r.get('tranche_start_i')}..{r.get('tranche_end_i')}]",
                    f"dt=[{r.get('tranche_start_dt')}..{r.get('tranche_end_dt')}]",
                    f"w=[{r.get('cci_weight_fast')},{r.get('cci_weight_medium')},{r.get('cci_weight_slow')}] sum={r.get('cci_weight_sum')}",
                    f"ext={r.get('cci_global_extreme')} abs={r.get('cci_global_extreme_abs')} pos={r.get('cci_global_extreme_pos')} ts={r.get('cci_global_extreme_ts')}",
                    f"cci@ext=[{r.get('cci_fast_at_extreme')},{r.get('cci_medium_at_extreme')},{r.get('cci_slow_at_extreme')}]",
                    f"last_ext={r.get('cci_global_last_extreme')} abs={r.get('cci_global_last_extreme_abs')} pos={r.get('cci_global_last_extreme_pos')} ts={r.get('cci_global_last_extreme_ts')}",
                    f"cci@last=[{r.get('cci_fast_at_last_extreme')},{r.get('cci_medium_at_last_extreme')},{r.get('cci_slow_at_last_extreme')}]",
                    f"score={float(r.get('score') or 0.0):.2f}",
                    f"interesting={bool(r.get('is_interesting'))}",
                ]
            )
        )

    if str(args.mode) == "current":
        r = ans.get("metric")
        print("\n=== Current (last tranche) ===")
        if r is None:
            print("No metric")
        else:
            _print_row(r)
        return 0

    metrics = list(ans.get("metrics") or [])
    metrics_sorted = sorted(metrics, key=lambda x: float(x.get("score") or 0.0), reverse=True)
    interesting = [m for m in metrics_sorted if bool(m.get("is_interesting"))]

    print("\n=== Ranked tranches (demo sort by score) ===")
    for r in metrics_sorted[: int(args.top_n)]:
        _print_row(r)

    print("\n=== Interesting (demo filter) ===")
    for r in interesting[: int(args.top_n)]:
        _print_row(r)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
