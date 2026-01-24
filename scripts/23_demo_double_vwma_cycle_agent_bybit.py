from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.agents.double_vwma_cycle_agent import DoubleVwmaCycleAgent, DoubleVwmaCycleAgentConfig
from libs.indicators.moving_averages.vwma_tv import vwma_tv


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
    ap.add_argument("--limit", type=int, default=900)
    ap.add_argument("--category", default="linear")
    ap.add_argument("--base-url", default="https://api.bybit.com")

    ap.add_argument("--vwma-fast-len", type=int, default=4)
    ap.add_argument("--vwma-slow-len", type=int, default=12)

    ap.add_argument("--zone-fast-radius-pct", type=float, default=0.001)
    ap.add_argument("--zone-slow-radius-pct", type=float, default=0.001)
    ap.add_argument("--zone-large-mult", type=float, default=2.0)

    ap.add_argument("--break-confirm-bars", type=int, default=3)

    ap.add_argument("--spread-ref-pct", type=float, default=0.002)

    ap.add_argument("--min-cycle-len", type=int, default=20)
    ap.add_argument("--min-score", type=float, default=0.05)

    ap.add_argument("--max-cycles", type=int, default=8)
    ap.add_argument("--top-n", type=int, default=6)

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

    now_utc_dt = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    first_dt = str(df["dt"].iloc[0]) if len(df) else ""
    last_dt = str(df["dt"].iloc[-1]) if len(df) else ""
    last_ts = int(pd.to_numeric(df["ts"].iloc[-1], errors="coerce")) if len(df) else 0
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    lag_min = (float(now_ms) - float(last_ts)) / 60000.0 if last_ts else float("nan")
    print(f"Now: {now_utc_dt}")
    print(f"Data range: {first_dt} -> {last_dt} | last_ts={last_ts} | lag_min={lag_min:.2f}")

    close = pd.to_numeric(df["close"], errors="coerce").astype(float).tolist()
    volume = pd.to_numeric(df["volume"], errors="coerce").astype(float).tolist()

    fast_len = int(args.vwma_fast_len)
    slow_len = int(args.vwma_slow_len)

    vwma_fast_col = f"vwma_{fast_len}"
    vwma_slow_col = f"vwma_{slow_len}"

    df[vwma_fast_col] = vwma_tv(close, volume, int(fast_len))
    df[vwma_slow_col] = vwma_tv(close, volume, int(slow_len))

    agent_cfg = DoubleVwmaCycleAgentConfig(
        ts_col="ts",
        open_col="open",
        high_col="high",
        low_col="low",
        close_col="close",
        vwma_fast_col=str(vwma_fast_col),
        vwma_slow_col=str(vwma_slow_col),
        zone_fast_radius_pct=float(args.zone_fast_radius_pct),
        zone_slow_radius_pct=float(args.zone_slow_radius_pct),
        zone_large_mult=float(args.zone_large_mult),
        break_confirm_bars=int(args.break_confirm_bars),
        spread_ref_pct=float(args.spread_ref_pct),
        min_cycle_len=int(args.min_cycle_len),
        min_score=float(args.min_score),
    )

    agent = DoubleVwmaCycleAgent(cfg=agent_cfg)
    ans = agent.answer(
        question={
            "kind": str(args.mode),
            "max_cycles": int(args.max_cycles),
        },
        df=df,
    )

    if str(args.mode) == "current":
        print(f"Fetched {len(df)} klines for {args.symbol} {args.interval}")

        def _print_cycle(cyc: object, *, prefix: str = "", show_events: bool = True) -> None:
            if not isinstance(cyc, dict):
                cyc = asdict(cyc)
            pfx = str(prefix).strip() or "current"
            print(
                " | ".join(
                    [
                        pfx,
                        f"cycle={cyc.get('cycle_id')}",
                        f"trend={cyc.get('trend_side')}",
                        f"idx=[{cyc.get('start_i')}..{cyc.get('end_i')}]",
                        f"dt=[{cyc.get('start_dt')}..{cyc.get('end_dt')}]",
                        f"spread_max%={cyc.get('spread_abs_max_pct')} end%={cyc.get('spread_abs_end_pct')} slope%={cyc.get('spread_abs_slope_mean_pct')}",
                        f"slopes%: fast={cyc.get('vwma_fast_slope_mean_pct')} slow={cyc.get('vwma_slow_slope_mean_pct')} harmony={cyc.get('vwma_slope_harmony_ratio')}",
                        f"pullbacks: weak={cyc.get('pullback_weak_count')} med={cyc.get('pullback_medium_count')} strong={cyc.get('pullback_strong_count')}",
                        f"last_pb={cyc.get('last_pullback_kind')} rec={cyc.get('last_pullback_recency')} break_slow_confirmed={cyc.get('break_slow_confirmed')}",
                        f"score={float(cyc.get('score') or 0.0):.4f}",
                        f"interesting={bool(cyc.get('is_interesting'))}",
                    ]
                )
            )
            if bool(show_events):
                for e in cyc.get("events") or []:
                    print(
                        "  "
                        + " | ".join(
                            [
                                f"{e.get('pos')}",
                                f"{e.get('dt')}",
                                f"{e.get('kind')}",
                                f"meta={e.get('meta')}",
                            ]
                        )
                    )

        print("\n=== Current (last cycle) ===")
        m = ans.get("metric")
        if m is None:
            print("(none)")
        else:
            _print_cycle(m, prefix="order=current")
        return 0

    metrics = list(ans.get("metrics") or [])
    cycles_chrono_dicts = list(metrics)
    cycles_all = agent.analyze_df(df, max_cycles=0)

    print(f"Fetched {len(df)} klines for {args.symbol} {args.interval}")

    first_k_ts = int(pd.to_numeric(df["ts"].iloc[0], errors="coerce")) if len(df) else 0
    last_k_ts = int(pd.to_numeric(df["ts"].iloc[-1], errors="coerce")) if len(df) else 0
    first_k_dt = str(pd.to_datetime(first_k_ts, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")) if first_k_ts else ""
    last_k_dt = str(pd.to_datetime(last_k_ts, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")) if last_k_ts else ""

    cycles_ok = True
    for a, b in zip(cycles_chrono_dicts, cycles_chrono_dicts[1:]):
        if int(a.get("start_ts") or 0) > int(b.get("start_ts") or 0):
            cycles_ok = False
            break

    cycles_all_ok = True
    for a, b in zip(cycles_all, cycles_all[1:]):
        if int(a.start_ts) > int(b.start_ts):
            cycles_all_ok = False
            break

    fc = cycles_chrono_dicts[0] if len(cycles_chrono_dicts) else None
    lc = cycles_chrono_dicts[-1] if len(cycles_chrono_dicts) else None
    fc_all = cycles_all[0] if len(cycles_all) else None
    lc_all = cycles_all[-1] if len(cycles_all) else None

    # metrics in analyze mode are already dicts
    chrono_pos_by_id = {int(c.get("cycle_id") or -1): int(i + 1) for i, c in enumerate(cycles_chrono_dicts)}

    def _print_cycle(cyc: object, *, prefix: str = "", show_events: bool = True) -> None:
        if not isinstance(cyc, dict):
            cyc = asdict(cyc)
        cid = int(cyc.get("cycle_id") or -1)
        chrono_pos = chrono_pos_by_id.get(int(cid))
        pfx = str(prefix).strip()
        if not pfx:
            if chrono_pos is None:
                pfx = f"cycle={cid}"
            else:
                pfx = f"chrono_pos={int(chrono_pos)}/{int(len(cycles_chrono_dicts))}"
        print(
            " | ".join(
                [
                    pfx,
                    f"cycle={cyc.get('cycle_id')}",
                    f"trend={cyc.get('trend_side')}",
                    f"idx=[{cyc.get('start_i')}..{cyc.get('end_i')}]",
                    f"dt=[{cyc.get('start_dt')}..{cyc.get('end_dt')}]",
                    f"spread_max%={cyc.get('spread_abs_max_pct')} end%={cyc.get('spread_abs_end_pct')} slope%={cyc.get('spread_abs_slope_mean_pct')}",
                    f"slopes%: fast={cyc.get('vwma_fast_slope_mean_pct')} slow={cyc.get('vwma_slow_slope_mean_pct')} harmony={cyc.get('vwma_slope_harmony_ratio')}",
                    f"pullbacks: weak={cyc.get('pullback_weak_count')} med={cyc.get('pullback_medium_count')} strong={cyc.get('pullback_strong_count')}",
                    f"last_pb={cyc.get('last_pullback_kind')} rec={cyc.get('last_pullback_recency')} break_slow_confirmed={cyc.get('break_slow_confirmed')}",
                    f"score={float(cyc.get('score') or 0.0):.4f}",
                ]
            )
        )
        if bool(show_events):
            for e in cyc.get("events") or []:
                print(
                    "  "
                    + " | ".join(
                        [
                            f"{e.get('pos')}",
                            f"{e.get('dt')}",
                            f"{e.get('kind')}",
                            f"meta={e.get('meta')}",
                        ]
                    )
                )

    print("\n=== Cycles (chronological: oldest -> newest) ===")
    for i, c in enumerate(cycles_chrono_dicts, start=1):
        _print_cycle(c, prefix=f"order=chrono chrono_pos={i}/{len(cycles_chrono_dicts)}")

    last_completed = None
    current_cycle = None
    if len(cycles_chrono_dicts) >= 2:
        last_completed = cycles_chrono_dicts[-2]
        current_cycle = cycles_chrono_dicts[-1]
    elif len(cycles_chrono_dicts) == 1:
        current_cycle = cycles_chrono_dicts[-1]

    print("\n=== Last completed cycle ===")
    if last_completed is None:
        print("(none)")
    else:
        _print_cycle(last_completed, prefix="order=chrono last_completed")

    print("\n=== Current (ongoing) cycle ===")
    if current_cycle is None:
        print("(none)")
    else:
        _print_cycle(current_cycle, prefix="order=chrono current")

    print("\n=== Ranked cycles (by score; not chronological) ===")
    metrics_sorted = sorted(cycles_chrono_dicts, key=lambda x: float(x.get("score") or 0.0), reverse=True)
    for rank, c in enumerate(metrics_sorted[: int(args.top_n)], start=1):
        _print_cycle(c, prefix=f"order=ranked(score) rank={rank}")

    print("\n=== Interesting (filtered; by score; not chronological) ===")
    interesting_sorted = [c for c in metrics_sorted if bool(c.get("is_interesting"))]
    for rank, c in enumerate(interesting_sorted[: int(args.top_n)], start=1):
        _print_cycle(c, prefix=f"order=interesting(score) rank={rank}")

    print("\n=== Most recent cycle (chronological; even if not interesting) ===")
    if len(cycles_chrono_dicts):
        _print_cycle(cycles_chrono_dicts[-1], prefix="order=chrono last", show_events=False)
    else:
        print("(none)")

    print("\n=== Summary (ranges) ===")
    print(f"Klines range: {first_k_dt} -> {last_k_dt}")
    print(f"Cycles chronological sorted (selected): {bool(cycles_ok)}")
    print(f"Cycles chronological sorted (all): {bool(cycles_all_ok)}")
    if fc is None or lc is None:
        print(f"Cycles range (selected last max_cycles={int(args.max_cycles)}): (none)")
    else:
        print(
            f"Cycles range (selected last max_cycles={int(args.max_cycles)}): {fc.start_dt} -> {lc.end_dt} | n_cycles={len(cycles_chrono)}"
        )
    if fc_all is None or lc_all is None:
        print("Cycles range (all cycles): (none)")
    else:
        print(f"Cycles range (all cycles): {fc_all.start_dt} -> {lc_all.end_dt} | n_cycles={len(cycles_all)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
