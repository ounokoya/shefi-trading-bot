from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.indicators.volume.pvt_tv import pvt_tv  # noqa: E402


def _auto_csv_path(*, symbol: str, interval: str) -> str:
    cache_dir = PROJECT_ROOT / "data" / "cache" / "klines"
    if not cache_dir.exists():
        return ""

    prefix = f"bybit_klines_{str(symbol).upper()}_{str(interval)}_"
    candidates = [p for p in cache_dir.glob(f"{prefix}*.csv") if p.is_file()]
    if not candidates:
        return ""

    best = max(candidates, key=lambda p: p.stat().st_size)
    return str(best)


def _load_klines_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(str(csv_path))
    if "ts" not in df.columns:
        raise ValueError("CSV must contain 'ts' column")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["ts"]).copy()
    df["ts"] = df["ts"].astype(int)

    needed = ["open", "high", "low", "close", "volume", "turnover"]
    for c in needed:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    if "dt" not in df.columns:
        df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    df = df.sort_values("ts").reset_index(drop=True)
    return df


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
        raise ValueError(f"unsupported minute interval for Bybit: {minutes} (from {interval}). Allowed: {allowed_str}")
    return str(minutes)


def _fetch_bybit_klines_last_n(
    *,
    symbol: str,
    interval: str,
    limit: int,
    category: str,
    base_url: str,
    timeout_s: float,
) -> pd.DataFrame:
    import requests

    url = f"{base_url.rstrip('/')}/v5/market/kline"

    page_limit = int(limit)
    if page_limit <= 0:
        page_limit = 200
    if page_limit > 1000:
        page_limit = 1000

    params: dict[str, Any] = {
        "category": str(category),
        "symbol": str(symbol),
        "interval": str(interval),
        "limit": str(int(page_limit)),
    }

    r = requests.get(url, params=params, timeout=float(timeout_s))
    r.raise_for_status()
    payload = r.json()
    if str(payload.get("retCode")) != "0":
        raise RuntimeError(f"Bybit retCode={payload.get('retCode')} retMsg={payload.get('retMsg')}")

    result = payload.get("result") or {}
    rows = result.get("list") or []
    out_rows: list[dict[str, object]] = []

    for row in rows:
        if not isinstance(row, list) or len(row) < 6:
            continue
        try:
            ts = int(row[0])
        except Exception:
            continue

        turnover = float("nan")
        if len(row) >= 7:
            try:
                turnover = float(row[6])
            except Exception:
                turnover = float("nan")
        out_rows.append(
            {
                "ts": ts,
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "turnover": turnover,
            }
        )

    df = pd.DataFrame(out_rows)
    if len(df):
        df = df.sort_values("ts").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="LINKUSDT")
    ap.add_argument("--category", default="linear")
    ap.add_argument("--base-url", default="https://api.bybit.com")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--volume-source", choices=["volume", "turnover"], default="volume")
    ap.add_argument("--csv-path", default="")
    ap.add_argument(
        "--auto-csv",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    ap.add_argument("--tail", type=int, default=10)
    ap.add_argument(
        "--print-bars",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    args = ap.parse_args()

    interval_bybit = _interval_to_bybit(str(args.interval))

    csv_path = str(args.csv_path).strip()
    if not csv_path and bool(args.auto_csv):
        csv_path = _auto_csv_path(symbol=str(args.symbol), interval=str(args.interval))

    if csv_path:
        df = _load_klines_csv(csv_path)
    else:
        df = _fetch_bybit_klines_last_n(
            symbol=str(args.symbol),
            interval=interval_bybit,
            limit=int(args.limit),
            category=str(args.category),
            base_url=str(args.base_url),
            timeout_s=30.0,
        )
    if not len(df):
        raise RuntimeError("no_klines")

    vol_col = str(args.volume_source)
    pvt = pvt_tv(
        close=pd.to_numeric(df["close"], errors="coerce").astype(float).tolist(),
        volume=pd.to_numeric(df[vol_col], errors="coerce").astype(float).tolist(),
    )
    df["pvt"] = pvt

    if bool(args.print_bars):
        tail = int(args.tail)
        if tail <= 0:
            tail = 50
        cols = ["dt", "close", vol_col, "pvt"]
        print(df[cols].tail(tail).to_string(index=False))

    last = df.iloc[-1]
    print(f"LAST | {args.symbol} | tf={args.interval} | pvt={last.get('pvt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
