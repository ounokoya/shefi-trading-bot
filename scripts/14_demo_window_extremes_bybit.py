from __future__ import annotations

import argparse
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.blocks.add_macd_tv_columns_df import add_macd_tv_columns_df
from libs.extremes.window_extremes import extract_window_close_extremes
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
    ap.add_argument("--out", default="")

    ap.add_argument("--macd-fast", type=int, default=12)
    ap.add_argument("--macd-slow", type=int, default=26)
    ap.add_argument("--macd-signal", type=int, default=9)

    ap.add_argument("--cci-fast", type=int, default=30)
    ap.add_argument("--cci-medium", type=int, default=120)
    ap.add_argument("--cci-slow", type=int, default=300)

    ap.add_argument("--cci-thr-fast", type=float, default=100.0)
    ap.add_argument("--cci-thr-medium", type=float, default=100.0)
    ap.add_argument("--cci-thr-slow", type=float, default=100.0)

    ap.add_argument("--zone-radius-pct", type=float, default=0.01)

    args = ap.parse_args()

    interval_bybit = _interval_to_bybit(str(args.interval))
    klines = _fetch_bybit_klines(
        symbol=str(args.symbol),
        interval=interval_bybit,
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

    extremes_df = extract_window_close_extremes(
        df,
        ts_col="ts",
        close_col="close",
        hist_col="macd_hist",
        cci_fast_col=cci_fast_col,
        cci_medium_col=cci_medium_col,
        cci_slow_col=cci_slow_col,
        cci_fast_threshold=float(args.cci_thr_fast),
        cci_medium_threshold=float(args.cci_thr_medium),
        cci_slow_threshold=float(args.cci_thr_slow),
        zone_radius_pct=float(args.zone_radius_pct),
    )

    out_path: Path
    if args.out:
        out_path = Path(str(args.out))
    else:
        ts_now = int(time.time())
        out_path = (
            PROJECT_ROOT
            / "data/processed/extremes"
            / f"bybit_window_extremes_{str(args.symbol)}_{str(args.interval)}_{int(args.limit)}_{ts_now}.csv"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    extremes_df.to_csv(out_path, index=False)

    print(f"Fetched {len(df)} klines for {args.symbol} {args.interval}")
    print(f"Detected {len(extremes_df)} confirmed close extremes")
    print(f"Wrote: {out_path}")

    for r in extremes_df.tail(200).to_dict("records"):
        print(
            f"extreme | {r.get('dt')} kind={r.get('kind')} close={r.get('close')} bars_ago={r.get('bars_ago')} "
            f"pct_from_now={r.get('pct_from_now'):.5f} cci_cat={r.get('cci_category')} cci={r.get('cci_value')} "
            f"zone_same={r.get('zone_same_type_count')} zone_other={r.get('zone_other_type_count')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
