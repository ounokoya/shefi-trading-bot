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

from libs.agents.macd_hist_tranche_agent import HistTrancheAgentConfig, MacdHistTrancheAgent
from libs.blocks.add_macd_tv_columns_df import add_macd_tv_columns_df


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

    ap.add_argument("--mode", choices=["analyze", "current"], default="analyze")

    ap.add_argument("--max-tranches", type=int, default=12)
    ap.add_argument("--top-n", type=int, default=10)

    ap.add_argument("--min-abs-force", type=float, default=0.0)

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

    agent_cfg = HistTrancheAgentConfig(
        ts_col="ts",
        close_col="close",
        hist_col="macd_hist",
        min_abs_force=float(args.min_abs_force),
    )

    agent = MacdHistTrancheAgent(cfg=agent_cfg)

    if str(args.mode) == "current":
        ans = agent.answer(
            question={
                "kind": "current",
            },
            df=df,
        )

        r = ans.get("metric")
        if not r:
            print(f"Fetched {len(df)} klines for {args.symbol} {args.interval}")
            print("\n=== Current tranche ===")
            print("None")
            return 0

        start_dt = str(r.get("tranche_start_dt") or "")
        end_dt = str(r.get("tranche_end_dt") or "")
        start_ts = r.get("tranche_start_ts")
        end_ts = r.get("tranche_end_ts")
        start_i = r.get("tranche_start_i")
        end_i = r.get("tranche_end_i")
        force_mean_abs = r.get("force_mean_abs")
        force_peak_abs = r.get("force_peak_abs")

        print(f"Fetched {len(df)} klines for {args.symbol} {args.interval}")
        print("\n=== Current tranche ===")
        print(
            " | ".join(
                [
                    f"id={r.get('tranche_id')}",
                    f"type={r.get('tranche_type')}",
                    f"len={r.get('tranche_len')}",
                    f"idx=[{start_i}..{end_i}]",
                    f"dt=[{start_dt}..{end_dt}]",
                    f"ts=[{start_ts}..{end_ts}]",
                    f"force_mean_abs={force_mean_abs}",
                    f"force_peak_abs={force_peak_abs}",
                    f"score={float(r.get('score') or 0.0):.4f}",
                ]
            )
        )
        return 0

    ans = agent.answer(
        question={
            "max_tranches": int(args.max_tranches),
            "kind": "analyze",
        },
        df=df,
    )

    metrics = list(ans.get("metrics") or [])
    metrics.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    ranked = metrics[: int(args.top_n)]
    interesting = [m for m in metrics if float(m.get("score") or 0.0) >= float(args.min_abs_force)][: int(args.top_n)]

    print(f"Fetched {len(df)} klines for {args.symbol} {args.interval}")
    print("\n=== Ranked tranches ===")
    for r in ranked:
        start_dt = str(r.get("tranche_start_dt") or "")
        end_dt = str(r.get("tranche_end_dt") or "")
        start_ts = r.get("tranche_start_ts")
        end_ts = r.get("tranche_end_ts")
        start_i = r.get("tranche_start_i")
        end_i = r.get("tranche_end_i")
        force_mean_abs = r.get("force_mean_abs")
        force_peak_abs = r.get("force_peak_abs")
        print(
            " | ".join(
                [
                    f"id={r.get('tranche_id')}",
                    f"type={r.get('tranche_type')}",
                    f"len={r.get('tranche_len')}",
                    f"idx=[{start_i}..{end_i}]",
                    f"dt=[{start_dt}..{end_dt}]",
                    f"ts=[{start_ts}..{end_ts}]",
                    f"force_mean_abs={force_mean_abs}",
                    f"force_peak_abs={force_peak_abs}",
                    f"score={float(r.get('score') or 0.0):.4f}",
                ]
            )
        )

    print("\n=== Interesting (filtered) ===")
    for r in interesting:
        start_dt = str(r.get("tranche_start_dt") or "")
        end_dt = str(r.get("tranche_end_dt") or "")
        start_ts = r.get("tranche_start_ts")
        end_ts = r.get("tranche_end_ts")
        start_i = r.get("tranche_start_i")
        end_i = r.get("tranche_end_i")

        force_mean_abs = r.get("force_mean_abs")
        force_peak_abs = r.get("force_peak_abs")
        print(
            " | ".join(
                [
                    f"id={r.get('tranche_id')}",
                    f"type={r.get('tranche_type')}",
                    f"len={r.get('tranche_len')}",
                    f"idx=[{start_i}..{end_i}]",
                    f"dt=[{start_dt}..{end_dt}]",
                    f"ts=[{start_ts}..{end_ts}]",
                    f"force_mean_abs={force_mean_abs}",
                    f"force_peak_abs={force_peak_abs}",
                    f"score={float(r.get('score') or 0.0):.4f}",
                ]
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
