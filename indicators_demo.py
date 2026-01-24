from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd
import requests

from libs.indicators.momentum.cci_tv import cci_tv
from libs.indicators.momentum.dmi_tv import dmi_tv
from libs.indicators.momentum.macd_tv import macd_tv
from libs.indicators.momentum.vortex_tv import vortex_tv
from libs.indicators.moving_averages.vwma_tv import vwma_tv
from libs.indicators.volatility.atr_tv import atr_tv
from libs.indicators.volume.volume_oscillator_tv import percentage_volume_oscillator_tv
from libs.indicators.volume.mfi_tv import mfi_tv


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
    ap.add_argument("--interval", default="4h")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--category", default="linear")
    ap.add_argument("--base-url", default="https://api.bybit.com")
    ap.add_argument("--out", default="")
    ap.add_argument("--volume-field", choices=["base", "quote"], default="base")

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
            "timestamp_ms": [k.timestamp_ms for k in klines],
        }
    )

    df["time_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%SZ")

    close = [k.close for k in klines]
    high = [k.high for k in klines]
    low = [k.low for k in klines]
    volume_base = [k.volume for k in klines]
    volume_quote = [k.turnover for k in klines]
    volume = volume_base if args.volume_field == "base" else volume_quote

    df["pvo_30_300"] = percentage_volume_oscillator_tv(volume, 30, 300)

    _, plus_di_300, minus_di_300 = dmi_tv(high, low, close, 300, adx_smoothing=6)
    df["plus_di_300"] = plus_di_300
    df["minus_di_300"] = minus_di_300

    vortex_plus_300, vortex_minus_300 = vortex_tv(high, low, close, 300)
    df["vortex_plus_300"] = vortex_plus_300
    df["vortex_minus_300"] = vortex_minus_300

    df["cci_300"] = cci_tv(high, low, close, 300)
    df["mfi_14"] = mfi_tv(high, low, close, volume, 14)
    df["atr_14"] = atr_tv(high, low, close, 14)

    macd_line, macd_signal, macd_hist = macd_tv(close, 12, 26, 9)
    df["macd_line"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    df["vwma_6"] = vwma_tv(close, volume, 6)

    df = df[
        [
            "timestamp_ms",
            "time_utc",
            "pvo_30_300",
            "plus_di_300",
            "minus_di_300",
            "vortex_plus_300",
            "vortex_minus_300",
            "cci_300",
            "mfi_14",
            "atr_14",
            "macd_line",
            "macd_signal",
            "macd_hist",
            "vwma_6",
        ]
    ]

    out_path: Path
    if args.out:
        out_path = Path(str(args.out))
    else:
        ts = int(time.time())
        out_path = Path(__file__).resolve().parent / f"bybit_{args.symbol}_{args.interval}_{args.limit}_indicators_demo_{ts}.csv"

    df.to_csv(out_path, index=False)
    print(json.dumps({"csv": str(out_path), "volume_field": str(args.volume_field)}, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
