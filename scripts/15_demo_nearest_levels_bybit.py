from __future__ import annotations

import argparse
import math
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
from libs.pivots.pivot_registry import PivotRegistry


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


def _interval_to_minutes(interval: str) -> int:
    s = str(interval).strip().lower()
    if not s:
        raise ValueError("interval cannot be empty")
    if s.isdigit():
        return int(s)
    if s in {"d", "1d"}:
        return 24 * 60
    if s in {"w", "1w"}:
        return 7 * 24 * 60
    import re

    m = re.fullmatch(r"(\d+)([mhd])", s)
    if not m:
        raise ValueError(f"unsupported interval format: {interval}")
    n = int(m.group(1))
    u = m.group(2)
    if u == "m":
        return n
    if u == "h":
        return n * 60
    if u == "d":
        return n * 24 * 60
    raise ValueError(f"unsupported interval format: {interval}")


def _recommended_history_days(interval: str) -> int | None:
    m = _interval_to_minutes(interval)
    if m == 5:
        return 7
    if m == 15:
        return 14
    if m in {30, 60}:
        return 30
    if m == 240:
        return 90
    return None


def _fetch_bybit_klines(
    *,
    symbol: str,
    interval: str,
    limit: int,
    category: str,
    base_url: str,
    timeout_s: float,
    start_ts_ms: int | None = None,
    end_ts_ms: int | None = None,
) -> List[BybitKline]:
    url = f"{base_url.rstrip('/')}/v5/market/kline"
    params = {
        "category": category,
        "symbol": symbol,
        "interval": interval,
        "limit": str(int(limit)),
    }
    if start_ts_ms is not None:
        params["start"] = str(int(start_ts_ms))
    if end_ts_ms is not None:
        params["end"] = str(int(end_ts_ms))
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


def _fetch_bybit_klines_history_days(
    *,
    symbol: str,
    interval: str,
    category: str,
    base_url: str,
    timeout_s: float,
    history_days: int,
    page_limit: int,
) -> List[BybitKline]:
    if int(history_days) <= 0:
        raise ValueError("history_days must be > 0")

    end_ms = int(time.time() * 1000)
    start_ms = int(end_ms - int(history_days) * 24 * 60 * 60 * 1000)
    cur_end = int(end_ms)

    out: List[BybitKline] = []
    seen: set[int] = set()

    lim = int(page_limit)
    if lim <= 0:
        lim = 200
    if lim > 1000:
        lim = 1000

    for _ in range(50):
        batch = _fetch_bybit_klines(
            symbol=str(symbol),
            interval=str(interval),
            limit=int(lim),
            category=str(category),
            base_url=str(base_url),
            timeout_s=float(timeout_s),
            start_ts_ms=int(start_ms),
            end_ts_ms=int(cur_end),
        )
        if not batch:
            break

        oldest = batch[0].timestamp_ms
        for k in batch:
            if k.timestamp_ms not in seen:
                seen.add(k.timestamp_ms)
                out.append(k)

        if int(oldest) <= int(start_ms):
            break
        nxt_end = int(oldest) - 1
        if nxt_end >= int(cur_end):
            break
        cur_end = nxt_end

    out.sort(key=lambda k: k.timestamp_ms)
    return out


def _pick_level(
    pivots: PivotRegistry,
    *,
    current_ts: int,
    current_price: float,
    category: str,
    role: str,
    kind_filter: str,
    threshold_pct: float,
    threshold_mode: str,
) -> dict[str, object] | None:
    return pivots.pick_nearest_level(
        current_ts=int(current_ts),
        current_price=float(current_price),
        category=str(category),
        role=str(role),
        kind_filter=str(kind_filter),
        threshold_pct=float(threshold_pct),
        threshold_mode=str(threshold_mode),
    )


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
    ap.add_argument("--cci-thr-medium", type=float, default=90.0)
    ap.add_argument("--cci-thr-slow", type=float, default=80.0)

    ap.add_argument("--zone-radius-pct", type=float, default=0.01)

    ap.add_argument("--cci-category", default="all", choices=["all", "slow", "medium", "fast"])

    ap.add_argument("--resistance-kind", default="high", choices=["high", "both"])
    ap.add_argument("--support-kind", default="low", choices=["low", "both"])

    ap.add_argument("--extremes-max-bars-ago", type=int, default=-1)

    ap.add_argument("--level-threshold-pct", type=float, default=0.0)
    ap.add_argument("--level-threshold-mode", default="exclusive", choices=["exclusive", "inclusive"])

    ap.add_argument("--pivot-registry", default="")
    ap.add_argument("--pivot-eps-pct", type=float, default=-1.0)

    ap.add_argument("--history-days", type=int, default=-1)

    args = ap.parse_args()

    interval_bybit = _interval_to_bybit(str(args.interval))
    history_days_eff = int(args.history_days)
    if history_days_eff < 0:
        rec = _recommended_history_days(str(args.interval))
        history_days_eff = int(rec) if rec is not None else 0

    if history_days_eff > 0:
        klines = _fetch_bybit_klines_history_days(
            symbol=str(args.symbol),
            interval=interval_bybit,
            category=str(args.category),
            base_url=str(args.base_url),
            timeout_s=30.0,
            history_days=int(history_days_eff),
            page_limit=int(args.limit),
        )
    else:
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

    max_bars_ago: int | None = None
    if int(args.extremes_max_bars_ago) >= 0:
        max_bars_ago = int(args.extremes_max_bars_ago)

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
        max_bars_ago=max_bars_ago,
    )

    now_ts = int(df["ts"].iloc[-1])
    now_dt = str(df["dt"].iloc[-1])
    now_close = float(df["close"].iloc[-1])

    eps = float(args.zone_radius_pct)
    if float(args.pivot_eps_pct) >= 0:
        eps = float(args.pivot_eps_pct)

    piv_path = Path(str(args.pivot_registry)) if str(args.pivot_registry) else None
    if piv_path is not None and piv_path.exists():
        pivots = PivotRegistry.from_json(piv_path)
        pivots.update_from_extremes_df(extremes_df)
        pivots.to_json(piv_path)
    else:
        pivots = PivotRegistry.from_extremes_df(extremes_df, symbol=str(args.symbol), tf=str(args.interval), eps=eps)
        if piv_path is not None:
            pivots.to_json(piv_path)

    rows: list[dict[str, object]] = []
    cats: tuple[str, ...]
    if str(args.cci_category).lower() == "all":
        cats = ("slow", "medium", "fast")
    else:
        cats = (str(args.cci_category).lower(),)

    for cat in cats:
        r_res = _pick_level(
            pivots,
            current_ts=now_ts,
            current_price=now_close,
            category=cat,
            role="resistance",
            kind_filter=str(args.resistance_kind),
            threshold_pct=float(args.level_threshold_pct),
            threshold_mode=str(args.level_threshold_mode),
        )
        if r_res is not None:
            r_res["cci_category"] = str(cat)
            r_res["level_role"] = "resistance"
            rows.append(r_res)

        r_sup = _pick_level(
            pivots,
            current_ts=now_ts,
            current_price=now_close,
            category=cat,
            role="support",
            kind_filter=str(args.support_kind),
            threshold_pct=float(args.level_threshold_pct),
            threshold_mode=str(args.level_threshold_mode),
        )
        if r_sup is not None:
            r_sup["cci_category"] = str(cat)
            r_sup["level_role"] = "support"
            rows.append(r_sup)

    out_df = pd.DataFrame(rows)
    if len(out_df) > 0:
        out_df["current_price"] = pd.to_numeric(out_df.get("current_price"), errors="coerce")
        out_df["level_price"] = pd.to_numeric(out_df.get("level"), errors="coerce")
        out_df["distance_price"] = pd.to_numeric(out_df.get("d_price"), errors="coerce")
        out_df["distance_pct_abs"] = pd.to_numeric(out_df.get("d_abs"), errors="coerce")
        out_df = out_df.sort_values(["cci_category", "level_role"], ascending=[True, True]).reset_index(drop=True)

    out_path: Path
    if args.out:
        out_path = Path(str(args.out))
    else:
        ts_now = int(time.time())
        out_path = (
            PROJECT_ROOT
            / "data/processed/extremes"
            / f"bybit_nearest_levels_{str(args.symbol)}_{str(args.interval)}_{int(args.limit)}_{ts_now}.csv"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Fetched {len(df)} klines for {args.symbol} {args.interval}")
    print(f"Now: ts={now_ts} dt={now_dt} close={now_close}")
    print(f"History days: {history_days_eff}")
    print(f"CCI categories: {','.join(cats)}")
    print(f"Extremes max_bars_ago: {max_bars_ago}")
    print(f"Level threshold: pct={float(args.level_threshold_pct)} mode={str(args.level_threshold_mode)}")
    print(f"Pivot registry: events={len(pivots.events)} zones={len(pivots.zones)} eps={eps}")
    print(f"All confirmed categorized extremes in window: {len(extremes_df)}")
    print(f"Nearest levels rows: {len(out_df)}")
    print(f"Wrote: {out_path}")

    for r in out_df.to_dict("records"):
        cur = r.get("current_price")
        lvl = r.get("level_price")
        dist_pct = r.get("d_pct")
        dist_abs = r.get("distance_pct_abs")
        dist_price = r.get("distance_price")
        dist_pct_s = ""
        if dist_pct is not None and math.isfinite(float(dist_pct)):
            dist_pct_s = f"{float(dist_pct):.5f}"
        print(
            f"level | cat={r.get('cci_category')} primary={r.get('cat')} role={r.get('level_role')} kind={r.get('kind')} "
            f"dt={r.get('dt')} level={lvl} current={cur} d_price={dist_price} d_pct={dist_pct_s} "
            f"d_abs={dist_abs} bars_ago={r.get('bars_ago')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
