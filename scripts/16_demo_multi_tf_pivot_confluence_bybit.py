from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.blocks.add_macd_tv_columns_df import add_macd_tv_columns_df
from libs.extremes.window_extremes import extract_window_close_extremes
from libs.indicators.momentum.cci_tv import cci_tv
from libs.pivots.pivot_registry import PivotRegistry
from libs.pivots.mtf_confluence import (
    build_triple_from_pairs,
    format_dt_ms_utc,
    match_zones,
    zone_representative_event,
)
from libs.pivots.grid_confluence import build_grid_confluence, extract_execution_pivot_price_weight_table


@dataclass
class BybitKline:
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    turnover: float


def _fetch_latest_price(
    *,
    symbol: str,
    interval: str,
    category: str,
    base_url: str,
    timeout_s: float,
) -> tuple[int, float]:
    bybit_interval = _interval_to_bybit(str(interval))
    klines = _fetch_bybit_klines(
        symbol=str(symbol),
        interval=str(bybit_interval),
        limit=1,
        category=str(category),
        base_url=str(base_url),
        timeout_s=float(timeout_s),
    )
    if not klines:
        raise RuntimeError("cannot fetch latest kline for current price")
    k = klines[-1]
    return int(k.timestamp_ms), float(k.close)


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


def _bootstrap_history_days(interval: str) -> int:
    m = _interval_to_minutes(interval)
    if m == 5:
        return 7
    if m == 15:
        return 14
    if m in {30, 60}:
        return 30
    if m == 240:
        return 90
    if m == 480:
        return 180
    if m == 1440:
        return 540
    raise ValueError(f"unsupported interval for bootstrap mapping: {interval}")


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
    params: dict[str, str] = {
        "category": str(category),
        "symbol": str(symbol),
        "interval": str(interval),
        "limit": str(int(limit)),
    }
    if start_ts_ms is not None:
        params["start"] = str(int(start_ts_ms))
    if end_ts_ms is not None:
        params["end"] = str(int(end_ts_ms))

    r = requests.get(url, params=params, timeout=float(timeout_s))
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

    for _ in range(200):
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


def _klines_to_df(klines: List[BybitKline]) -> pd.DataFrame:
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


def _resample_ohlcv(df: pd.DataFrame, *, target_minutes: int) -> pd.DataFrame:
    if len(df) == 0:
        return df

    work = df.copy()
    idx = pd.to_datetime(work["ts"], unit="ms", utc=True)
    work = work.set_index(idx)
    rule = f"{int(target_minutes)}T"

    agg = work.resample(rule, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )

    agg = agg.dropna(subset=["open", "high", "low", "close"]).reset_index().rename(columns={"index": "dt"})
    agg["ts"] = (agg["dt"].astype("datetime64[ns]").astype("int64") // 1_000_000).astype("int64")
    agg["dt"] = pd.to_datetime(agg["ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    agg = agg[["ts", "dt", "open", "high", "low", "close", "volume"]]
    return agg


def _compute_extremes_df(
    df: pd.DataFrame,
    *,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    cci_fast: int,
    cci_medium: int,
    cci_slow: int,
    cci_thr_fast: float,
    cci_thr_medium: float,
    cci_thr_slow: float,
    zone_radius_pct: float,
) -> pd.DataFrame:
    w = df.copy()
    w = add_macd_tv_columns_df(
        w,
        close_col="close",
        fast_period=int(macd_fast),
        slow_period=int(macd_slow),
        signal_period=int(macd_signal),
    )

    high = pd.to_numeric(w["high"], errors="coerce").astype(float).tolist()
    low = pd.to_numeric(w["low"], errors="coerce").astype(float).tolist()
    close = pd.to_numeric(w["close"], errors="coerce").astype(float).tolist()

    cci_fast_col = f"cci_{int(cci_fast)}"
    cci_medium_col = f"cci_{int(cci_medium)}"
    cci_slow_col = f"cci_{int(cci_slow)}"

    w[cci_fast_col] = cci_tv(high, low, close, int(cci_fast))
    w[cci_medium_col] = cci_tv(high, low, close, int(cci_medium))
    w[cci_slow_col] = cci_tv(high, low, close, int(cci_slow))

    extremes_df = extract_window_close_extremes(
        w,
        ts_col="ts",
        close_col="close",
        hist_col="macd_hist",
        cci_fast_col=cci_fast_col,
        cci_medium_col=cci_medium_col,
        cci_slow_col=cci_slow_col,
        cci_fast_threshold=float(cci_thr_fast),
        cci_medium_threshold=float(cci_thr_medium),
        cci_slow_threshold=float(cci_thr_slow),
        zone_radius_pct=float(zone_radius_pct),
        max_bars_ago=None,
    )
    return extremes_df


def _ensure_registry(
    *,
    symbol: str,
    interval: str,
    eps_pct: float,
    pivot_path: Path,
    base_url: str,
    category: str,
    timeout_s: float,
    page_limit: int,
    history_days: int,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    cci_fast: int,
    cci_medium: int,
    cci_slow: int,
    cci_thr_fast: float,
    cci_thr_medium: float,
    cci_thr_slow: float,
    fetch: bool,
    write: bool,
    skip_write_unchanged: bool,
) -> PivotRegistry:
    if pivot_path.exists():
        reg = PivotRegistry.from_json(pivot_path)
    else:
        reg = PivotRegistry.empty(symbol=str(symbol), tf=str(interval), eps=float(eps_pct))

    if not bool(fetch):
        if not pivot_path.exists():
            raise RuntimeError(f"--no-fetch requires existing registry file: {pivot_path}")
        return reg

    before_sig = (len(reg.events), len(reg.zones), int(reg.meta.get("last_ts") or 0))

    interval_bybit = _interval_to_bybit(interval)
    klines = _fetch_bybit_klines_history_days(
        symbol=str(symbol),
        interval=str(interval_bybit),
        category=str(category),
        base_url=str(base_url),
        timeout_s=float(timeout_s),
        history_days=int(history_days),
        page_limit=int(page_limit),
    )
    df = _klines_to_df(klines)

    extremes_df = _compute_extremes_df(
        df,
        macd_fast=int(macd_fast),
        macd_slow=int(macd_slow),
        macd_signal=int(macd_signal),
        cci_fast=int(cci_fast),
        cci_medium=int(cci_medium),
        cci_slow=int(cci_slow),
        cci_thr_fast=float(cci_thr_fast),
        cci_thr_medium=float(cci_thr_medium),
        cci_thr_slow=float(cci_thr_slow),
        zone_radius_pct=float(eps_pct),
    )

    reg.update_from_extremes_df(extremes_df)

    if bool(write):
        after_sig = (len(reg.events), len(reg.zones), int(reg.meta.get("last_ts") or 0))
        if bool(skip_write_unchanged) and before_sig == after_sig:
            return reg
        reg.to_json(pivot_path)
    return reg


def _ensure_registry_resampled(
    *,
    symbol: str,
    source_interval: str,
    target_interval: str,
    eps_pct: float,
    pivot_path: Path,
    base_url: str,
    category: str,
    timeout_s: float,
    page_limit: int,
    history_days: int,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    cci_fast: int,
    cci_medium: int,
    cci_slow: int,
    cci_thr_fast: float,
    cci_thr_medium: float,
    cci_thr_slow: float,
    fetch: bool,
    write: bool,
    skip_write_unchanged: bool,
) -> PivotRegistry:
    if pivot_path.exists():
        reg = PivotRegistry.from_json(pivot_path)
    else:
        reg = PivotRegistry.empty(symbol=str(symbol), tf=str(target_interval), eps=float(eps_pct))

    if not bool(fetch):
        if not pivot_path.exists():
            raise RuntimeError(f"--no-fetch requires existing registry file: {pivot_path}")
        return reg

    before_sig = (len(reg.events), len(reg.zones), int(reg.meta.get("last_ts") or 0))

    src_bybit = _interval_to_bybit(source_interval)
    klines = _fetch_bybit_klines_history_days(
        symbol=str(symbol),
        interval=str(src_bybit),
        category=str(category),
        base_url=str(base_url),
        timeout_s=float(timeout_s),
        history_days=int(history_days),
        page_limit=int(page_limit),
    )
    df_src = _klines_to_df(klines)
    df = _resample_ohlcv(df_src, target_minutes=_interval_to_minutes(target_interval))

    extremes_df = _compute_extremes_df(
        df,
        macd_fast=int(macd_fast),
        macd_slow=int(macd_slow),
        macd_signal=int(macd_signal),
        cci_fast=int(cci_fast),
        cci_medium=int(cci_medium),
        cci_slow=int(cci_slow),
        cci_thr_fast=float(cci_thr_fast),
        cci_thr_medium=float(cci_thr_medium),
        cci_thr_slow=float(cci_thr_slow),
        zone_radius_pct=float(eps_pct),
    )

    reg.update_from_extremes_df(extremes_df)
    reg.to_json(pivot_path)
    return reg


def _print_audit_top2_5m_aligned(
    title: str,
    *,
    reg_5m: PivotRegistry,
    reg_other: PivotRegistry,
    pairs_5m_other: list[dict[str, Any]],
    other_tf_label: str,
    now_ts_ms: int,
    bar_ms_5m: int,
    bar_ms_other: int,
) -> None:
    rows: list[tuple[int, int, int, dict[str, Any], dict[str, Any], dict[str, Any]]] = []

    for p in pairs_5m_other:
        ev_5m = zone_representative_event(reg_5m, str(p["zone_a"]), prefer="last")
        ev_ot = zone_representative_event(reg_other, str(p["zone_b"]), prefer="last")
        if not ev_5m or not ev_ot:
            continue
        dt_ms_5m = ev_5m.get("dt_ms")
        if not isinstance(dt_ms_5m, int):
            continue

        dt_ms_ot = ev_ot.get("dt_ms")
        dt_ms_ot_int = int(dt_ms_ot) if isinstance(dt_ms_ot, int) else None

        bars_ago_5m = 0
        if int(bar_ms_5m) > 0:
            bars_ago_5m = int(max(0, (int(now_ts_ms) - int(dt_ms_5m)) // int(bar_ms_5m)))

        bars_ago_ot = 0
        if dt_ms_ot_int is not None and int(bar_ms_other) > 0:
            bars_ago_ot = int(max(0, (int(now_ts_ms) - int(dt_ms_ot_int)) // int(bar_ms_other)))

        rows.append((bars_ago_5m, int(dt_ms_5m), bars_ago_ot, p, ev_5m, ev_ot))

    rows.sort(key=lambda t: (t[0], -t[1], float(t[3].get("d_abs") or 0.0)))
    top_rows = rows[:2]
    top_rows.sort(key=lambda t: t[1])

    print(title)
    if not top_rows:
        print("  (none)")
        return

    for bars_ago_5m, dt_ms_5m, bars_ago_ot, p, ev_5m, ev_ot in top_rows:
        dt_5m = format_dt_ms_utc(int(dt_ms_5m))
        dt_ot = format_dt_ms_utc(int(ev_ot.get("dt_ms")) if isinstance(ev_ot.get("dt_ms"), int) else None)
        print(
            "  "
            f"{dt_5m}  bars_ago_5m={int(bars_ago_5m)}  "
            f"5m[{ev_5m.get('kind')} {ev_5m.get('role')} lvl={ev_5m.get('level')} eid={ev_5m.get('event_id')} z={p.get('zone_a')}]"
            f"  <->  {other_tf_label}[{dt_ot} bars_ago={int(bars_ago_ot)} lvl={ev_ot.get('level')} eid={ev_ot.get('event_id')} z={p.get('zone_b')}]"
            f"  d={float(p.get('d_abs') or 0.0):.6f}"
        )


def _print_pairs(title: str, pairs: list[dict[str, Any]], *, max_rows: int = 20) -> None:
    print(title)
    if not pairs:
        print("  (none)")
        return
    for r in pairs[: int(max_rows)]:
        print(
            f"  role={r['role']} a={r['zone_a']} b={r['zone_b']} ref_a={r['ref_a']} ref_b={r['ref_b']} "
            f"d={r['d_abs']:.6f} touches_a={r['touches_a']} touches_b={r['touches_b']}"
        )


def _write_json(path: Path, obj: dict[str, Any], *, write: bool, skip_unchanged: bool) -> None:
    if not bool(write):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    s = json.dumps(obj, indent=2, sort_keys=True)
    if bool(skip_unchanged) and path.exists():
        prev = path.read_text()
        if prev == s:
            return
    path.write_text(s)


def _print_grid_zones(payload: dict[str, Any], *, max_macro: int = 6, max_context: int = 4, max_exec: int = 4) -> None:
    zones = payload.get("zones") or []
    print(f"Zones macro (nested): {len(zones)}")
    for mz in zones[: int(max_macro)]:
        role = str(mz.get("role"))
        tf = str(mz.get("tf"))
        center = float(mz.get("center_level") or 0.0)
        b = mz.get("bounds") or {}
        lo = float(b.get("lower") or 0.0)
        hi = float(b.get("upper") or 0.0)
        sub = mz.get("subzones") or []
        print(f"Macro role={role} tf={tf} center={center:.6f} bounds=[{lo:.6f}..{hi:.6f}] sub={len(sub)}")
        for cz in sub[: int(max_context)]:
            ctf = str(cz.get("tf"))
            ccenter = float(cz.get("center_level") or 0.0)
            cb = cz.get("bounds") or {}
            clo = float(cb.get("lower") or 0.0)
            chi = float(cb.get("upper") or 0.0)
            exs = cz.get("subzones") or []
            print(f"  Ctx tf={ctf} center={ccenter:.6f} bounds=[{clo:.6f}..{chi:.6f}] exec={len(exs)}")
            for ez in exs[: int(max_exec)]:
                etf = str(ez.get("tf"))
                ecenter = float(ez.get("center_level") or 0.0)
                eb = ez.get("bounds") or {}
                elo = float(eb.get("lower") or 0.0)
                ehi = float(eb.get("upper") or 0.0)
                picks = ez.get("picks") or []
                print(f"    Exec tf={etf} center={ecenter:.6f} bounds=[{elo:.6f}..{ehi:.6f}] picks={len(picks)}")

                for pick in picks[:2]:
                    members = pick.get("members") or {}
                    p5 = members.get("5m") or {}
                    p1 = members.get("1h") or {}
                    p4 = members.get("4h") or {}
                    score = pick.get("score") or {}
                    print(
                        f"      pick global={int(score.get('importance_global') or 0)} local={int(score.get('importance_local') or 0)} "
                        f"5m(bars_ago={p5.get('bars_ago')} lvl={p5.get('level')}) "
                        f"1h(bars_ago={p1.get('bars_ago')} lvl={p1.get('level')}) "
                        f"4h(bars_ago={p4.get('bars_ago')} lvl={p4.get('level')})"
                    )


def _print_pivot_price_weight_table(title: str, table: list[dict[str, Any]], *, max_rows: int = 30) -> None:
    print(title)
    if not table:
        print("  (empty)")
        return
    for r in table[: int(max_rows)]:
        role = str(r.get("role") or "")
        price = float(r.get("price") or 0.0)
        weight = int(r.get("weight") or 0)
        zid = str(r.get("zone_id") or "")
        eid = str(r.get("event_id") or "")
        ids = ""
        if zid or eid:
            ids = f" ids=(z={zid} eid={eid})"
        print(f"  {role} price={price:.6f} weight={weight}{ids}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--triad", default="5m_1h_4h", choices=["5m_1h_4h", "1h_8h_1d"])
    ap.add_argument("--category", default="linear")
    ap.add_argument("--base-url", default="https://api.bybit.com")
    ap.add_argument("--timeout-s", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=1000)

    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--skip-write", action="store_true")
    ap.add_argument("--skip-write-unchanged", action="store_true")
    ap.add_argument("--current-price", type=float, default=None)
    ap.add_argument("--now-ts-ms", type=int, default=None)

    ap.add_argument("--eps-pct", type=float, default=0.01)
    ap.add_argument("--mtf-eps-pct", type=float, default=0.01)

    ap.add_argument("--macd-fast", type=int, default=12)
    ap.add_argument("--macd-slow", type=int, default=26)
    ap.add_argument("--macd-signal", type=int, default=9)

    ap.add_argument("--cci-fast", type=int, default=30)
    ap.add_argument("--cci-medium", type=int, default=120)
    ap.add_argument("--cci-slow", type=int, default=300)

    ap.add_argument("--cci-thr-fast", type=float, default=100.0)
    ap.add_argument("--cci-thr-medium", type=float, default=90.0)
    ap.add_argument("--cci-thr-slow", type=float, default=80.0)

    ap.add_argument("--pivot-dir", default=str(PROJECT_ROOT / "data/processed/pivots"))
    ap.add_argument("--mtf-out", default="")

    ap.add_argument("--grid-mode", default="grid", choices=["grid", "zones"])
    ap.add_argument("--grid-pct", type=float, default=0.05)
    ap.add_argument("--grid-out", default="")

    ap.add_argument("--zones-macro-radius-pct", type=float, default=None)
    ap.add_argument("--zones-macro-padding-pct", type=float, default=None)
    ap.add_argument("--zones-context-radius-pct", type=float, default=None)
    ap.add_argument("--zones-context-padding-pct", type=float, default=None)
    ap.add_argument("--zones-exec-radius-pct", type=float, default=None)
    ap.add_argument("--zones-exec-padding-pct", type=float, default=None)

    args = ap.parse_args()

    triad = str(args.triad)
    if triad == "5m_1h_4h":
        tfs = ("5m", "1h", "4h")
        log_pair = ("5m", "1h")
    else:
        tfs = ("1h", "8h", "1d")
        log_pair = ("1h", "8h")

    pivot_dir = Path(str(args.pivot_dir))
    pivot_dir.mkdir(parents=True, exist_ok=True)

    eps_pct = float(args.eps_pct)
    eps_mtf = float(args.mtf_eps_pct)

    days_by_tf = {tf: _bootstrap_history_days(tf) for tf in tfs}

    regs: dict[str, PivotRegistry] = {}

    if triad == "1h_8h_1d":
        tf_1h = "1h"
        tf_8h = "8h"
        tf_1d = "1d"

        d_8h = int(days_by_tf[tf_8h])

        reg_8h = _ensure_registry_resampled(
            symbol=str(args.symbol),
            source_interval=tf_1h,
            target_interval=tf_8h,
            eps_pct=float(eps_pct),
            pivot_path=pivot_dir / f"{str(args.symbol)}_{tf_8h}.json",
            base_url=str(args.base_url),
            category=str(args.category),
            timeout_s=float(args.timeout_s),
            page_limit=int(args.limit),
            history_days=int(d_8h),
            macd_fast=int(args.macd_fast),
            macd_slow=int(args.macd_slow),
            macd_signal=int(args.macd_signal),
            cci_fast=int(args.cci_fast),
            cci_medium=int(args.cci_medium),
            cci_slow=int(args.cci_slow),
            cci_thr_fast=float(args.cci_thr_fast),
            cci_thr_medium=float(args.cci_thr_medium),
            cci_thr_slow=float(args.cci_thr_slow),
            fetch=not bool(args.no_fetch),
            write=not bool(args.skip_write),
            skip_write_unchanged=bool(args.skip_write_unchanged),
        )
        regs[tf_8h] = reg_8h

        reg_1h = _ensure_registry(
            symbol=str(args.symbol),
            interval=tf_1h,
            eps_pct=float(eps_pct),
            pivot_path=pivot_dir / f"{str(args.symbol)}_{tf_1h}.json",
            base_url=str(args.base_url),
            category=str(args.category),
            timeout_s=float(args.timeout_s),
            page_limit=int(args.limit),
            history_days=int(days_by_tf[tf_1h]),
            macd_fast=int(args.macd_fast),
            macd_slow=int(args.macd_slow),
            macd_signal=int(args.macd_signal),
            cci_fast=int(args.cci_fast),
            cci_medium=int(args.cci_medium),
            cci_slow=int(args.cci_slow),
            cci_thr_fast=float(args.cci_thr_fast),
            cci_thr_medium=float(args.cci_thr_medium),
            cci_thr_slow=float(args.cci_thr_slow),
        )
        regs[tf_1h] = reg_1h

        reg_1d = _ensure_registry(
            symbol=str(args.symbol),
            interval=tf_1d,
            eps_pct=float(eps_pct),
            pivot_path=pivot_dir / f"{str(args.symbol)}_{tf_1d}.json",
            base_url=str(args.base_url),
            category=str(args.category),
            timeout_s=float(args.timeout_s),
            page_limit=int(args.limit),
            history_days=int(days_by_tf[tf_1d]),
            macd_fast=int(args.macd_fast),
            macd_slow=int(args.macd_slow),
            macd_signal=int(args.macd_signal),
            cci_fast=int(args.cci_fast),
            cci_medium=int(args.cci_medium),
            cci_slow=int(args.cci_slow),
            cci_thr_fast=float(args.cci_thr_fast),
            cci_thr_medium=float(args.cci_thr_medium),
            cci_thr_slow=float(args.cci_thr_slow),
        )
        regs[tf_1d] = reg_1d
    else:
        for tf in tfs:
            regs[tf] = _ensure_registry(
                symbol=str(args.symbol),
                interval=str(tf),
                eps_pct=float(eps_pct),
                pivot_path=pivot_dir / f"{str(args.symbol)}_{str(tf)}.json",
                base_url=str(args.base_url),
                category=str(args.category),
                timeout_s=float(args.timeout_s),
                page_limit=int(args.limit),
                history_days=int(days_by_tf[str(tf)]),
                macd_fast=int(args.macd_fast),
                macd_slow=int(args.macd_slow),
                macd_signal=int(args.macd_signal),
                cci_fast=int(args.cci_fast),
                cci_medium=int(args.cci_medium),
                cci_slow=int(args.cci_slow),
                cci_thr_fast=float(args.cci_thr_fast),
                cci_thr_medium=float(args.cci_thr_medium),
                cci_thr_slow=float(args.cci_thr_slow),
                fetch=not bool(args.no_fetch),
                write=not bool(args.skip_write),
                skip_write_unchanged=bool(args.skip_write_unchanged),
            )

    low_tf, mid_tf, high_tf = tfs
    reg_low = regs[str(low_tf)]
    reg_mid = regs[str(mid_tf)]
    reg_high = regs[str(high_tf)]

    if args.current_price is not None and args.now_ts_ms is not None:
        now_ts_ms = int(args.now_ts_ms)
        current_price = float(args.current_price)
    else:
        now_ts_ms, current_price = _fetch_latest_price(
            symbol=str(args.symbol),
            interval=str(low_tf),
            category=str(args.category),
            base_url=str(args.base_url),
            timeout_s=float(args.timeout_s),
        )

    pairs_low_mid = match_zones(reg_low, reg_mid, eps_mtf=float(eps_mtf), current_price=float(current_price))
    pairs_mid_high = match_zones(reg_mid, reg_high, eps_mtf=float(eps_mtf), current_price=float(current_price))
    pairs_low_high: list[dict[str, Any]] = []
    if str(triad) == "5m_1h_4h":
        pairs_low_high = match_zones(reg_low, reg_high, eps_mtf=float(eps_mtf), current_price=float(current_price))
    triples = build_triple_from_pairs(
        pairs_low_mid=pairs_low_mid,
        pairs_mid_high=pairs_mid_high,
        mid_key=str(mid_tf),
        eps_mtf=float(eps_mtf),
    )

    print(f"Triad: {triad}")
    print(f"TF windows days: {days_by_tf}")
    for tf in tfs:
        r = regs[str(tf)]
        print(f"Registry {tf}: events={len(r.events)} zones={len(r.zones)} last_ts={r.meta.get('last_ts')}")

    print(f"Pairs {low_tf}↔{mid_tf}: {len(pairs_low_mid)}")
    print(f"Pairs {mid_tf}↔{high_tf}: {len(pairs_mid_high)}")
    if str(triad) == "5m_1h_4h":
        print(f"Pairs {low_tf}↔{high_tf}: {len(pairs_low_high)}")
    print(f"Triples {low_tf}↔{mid_tf}↔{high_tf}: {len(triples)}")

    if log_pair == (str(low_tf), str(mid_tf)):
        _print_pairs(f"Confluent zones {low_tf}↔{mid_tf} (top by closeness)", pairs_low_mid, max_rows=30)
    else:
        if log_pair == (str(mid_tf), str(high_tf)):
            _print_pairs(f"Confluent zones {mid_tf}↔{high_tf} (top by closeness)", pairs_mid_high, max_rows=30)
        else:
            if log_pair == (str(low_tf), str(mid_tf)):
                _print_pairs(f"Confluent zones {low_tf}↔{mid_tf} (top by closeness)", pairs_low_mid, max_rows=30)

    if str(low_tf) == "5m":
        now_ms = int(now_ts_ms)
        bar_ms_5m = int(_interval_to_minutes(str(low_tf)) * 60 * 1000)
        bar_ms_mid = int(_interval_to_minutes(str(mid_tf)) * 60 * 1000)
        bar_ms_high = int(_interval_to_minutes(str(high_tf)) * 60 * 1000)
        _print_audit_top2_5m_aligned(
            f"Audit top2 youngest pairs {low_tf}↔{mid_tf} (ordered by 5m datetime)",
            reg_5m=reg_low,
            reg_other=reg_mid,
            pairs_5m_other=pairs_low_mid,
            other_tf_label=str(mid_tf),
            now_ts_ms=int(now_ms),
            bar_ms_5m=int(bar_ms_5m),
            bar_ms_other=int(bar_ms_mid),
        )
        if str(triad) == "5m_1h_4h":
            _print_audit_top2_5m_aligned(
                f"Audit top2 youngest pairs {low_tf}↔{high_tf} (ordered by 5m datetime)",
                reg_5m=reg_low,
                reg_other=reg_high,
                pairs_5m_other=pairs_low_high,
                other_tf_label=str(high_tf),
                now_ts_ms=int(now_ms),
                bar_ms_5m=int(bar_ms_5m),
                bar_ms_other=int(bar_ms_high),
            )

    mtf_payload = {
        "meta": {
            "symbol": str(args.symbol),
            "triad": str(triad),
            "tfs": list(tfs),
            "eps_mtf": float(eps_mtf),
            "eps_local": float(eps_pct),
            "updated_ts": int(time.time() * 1000),
        },
        "pair_low_mid": [
            {"role": p["role"], "zone_low": p["zone_a"], "zone_mid": p["zone_b"], "d_abs": p["d_abs"]}
            for p in pairs_low_mid
        ],
        "pair_mid_high": [
            {"role": p["role"], "zone_mid": p["zone_a"], "zone_high": p["zone_b"], "d_abs": p["d_abs"]}
            for p in pairs_mid_high
        ],
        "pair_low_high": [
            {"role": p["role"], "zone_low": p["zone_a"], "zone_high": p["zone_b"], "d_abs": p["d_abs"]}
            for p in pairs_low_high
        ]
        if str(triad) == "5m_1h_4h"
        else [],
        "triple": triples,
    }

    mtf_out: Path
    if str(args.mtf_out).strip():
        mtf_out = Path(str(args.mtf_out))
    else:
        mtf_out = pivot_dir / f"mtf_{str(args.symbol)}_{triad}.json"

    out_dir = mtf_out.parent
    pair_low_mid_out = out_dir / f"mtf_{str(args.symbol)}_{str(low_tf)}_{str(mid_tf)}.json"
    pair_mid_high_out = out_dir / f"mtf_{str(args.symbol)}_{str(mid_tf)}_{str(high_tf)}.json"
    pair_low_high_out = out_dir / f"mtf_{str(args.symbol)}_{str(low_tf)}_{str(high_tf)}.json"
    triple_out = out_dir / f"mtf_{str(args.symbol)}_{str(triad)}_triple.json"

    pair_low_mid_payload = {
        "meta": {
            "symbol": str(args.symbol),
            "tfs": [str(low_tf), str(mid_tf)],
            "eps_mtf": float(eps_mtf),
            "updated_ts": int(time.time() * 1000),
        },
        "pairs": [
            {
                "role": p["role"],
                "zone_a": p["zone_a"],
                "zone_b": p["zone_b"],
                "ref_a": p["ref_a"],
                "ref_b": p["ref_b"],
                "d_abs": p["d_abs"],
                "touches_a": p["touches_a"],
                "touches_b": p["touches_b"],
            }
            for p in pairs_low_mid
        ],
    }
    pair_mid_high_payload = {
        "meta": {
            "symbol": str(args.symbol),
            "tfs": [str(mid_tf), str(high_tf)],
            "eps_mtf": float(eps_mtf),
            "updated_ts": int(time.time() * 1000),
        },
        "pairs": [
            {
                "role": p["role"],
                "zone_a": p["zone_a"],
                "zone_b": p["zone_b"],
                "ref_a": p["ref_a"],
                "ref_b": p["ref_b"],
                "d_abs": p["d_abs"],
                "touches_a": p["touches_a"],
                "touches_b": p["touches_b"],
            }
            for p in pairs_mid_high
        ],
    }

    pair_low_high_payload = {
        "meta": {
            "symbol": str(args.symbol),
            "tfs": [str(low_tf), str(high_tf)],
            "eps_mtf": float(eps_mtf),
            "updated_ts": int(time.time() * 1000),
        },
        "pairs": [
            {
                "role": p["role"],
                "zone_a": p["zone_a"],
                "zone_b": p["zone_b"],
                "ref_a": p["ref_a"],
                "ref_b": p["ref_b"],
                "d_abs": p["d_abs"],
                "touches_a": p["touches_a"],
                "touches_b": p["touches_b"],
            }
            for p in pairs_low_high
        ],
    }
    triple_payload = {
        "meta": {
            "symbol": str(args.symbol),
            "tfs": [str(low_tf), str(mid_tf), str(high_tf)],
            "eps_mtf": float(eps_mtf),
            "updated_ts": int(time.time() * 1000),
        },
        "triple": triples,
    }

    _write_json(mtf_out, mtf_payload, write=not bool(args.skip_write), skip_unchanged=bool(args.skip_write_unchanged))
    _write_json(pair_low_mid_out, pair_low_mid_payload, write=not bool(args.skip_write), skip_unchanged=bool(args.skip_write_unchanged))
    _write_json(pair_mid_high_out, pair_mid_high_payload, write=not bool(args.skip_write), skip_unchanged=bool(args.skip_write_unchanged))
    if str(triad) == "5m_1h_4h":
        _write_json(pair_low_high_out, pair_low_high_payload, write=not bool(args.skip_write), skip_unchanged=bool(args.skip_write_unchanged))
    _write_json(triple_out, triple_payload, write=not bool(args.skip_write), skip_unchanged=bool(args.skip_write_unchanged))
    print(f"Wrote MTF triad registry: {mtf_out}")
    print(f"Wrote MTF pair registry: {pair_low_mid_out}")
    print(f"Wrote MTF pair registry: {pair_mid_high_out}")
    if str(triad) == "5m_1h_4h":
        print(f"Wrote MTF pair registry: {pair_low_high_out}")
    print(f"Wrote MTF triple registry: {triple_out}")

    if str(triad) == "5m_1h_4h":
        grid_pct = float(args.grid_pct)
        grid_mode = str(args.grid_mode).strip().lower()

        zones_cfg = {"macro": {"tf": "4h"}, "context": {"tf": "1h"}, "execution": {"tf": "5m"}}
        if args.zones_macro_radius_pct is not None:
            zones_cfg["macro"]["radius_pct"] = float(args.zones_macro_radius_pct)
        if args.zones_macro_padding_pct is not None:
            zones_cfg["macro"]["padding_pct"] = float(args.zones_macro_padding_pct)
        if args.zones_context_radius_pct is not None:
            zones_cfg["context"]["radius_pct"] = float(args.zones_context_radius_pct)
        if args.zones_context_padding_pct is not None:
            zones_cfg["context"]["padding_pct"] = float(args.zones_context_padding_pct)
        if args.zones_exec_radius_pct is not None:
            zones_cfg["execution"]["radius_pct"] = float(args.zones_exec_radius_pct)
        if args.zones_exec_padding_pct is not None:
            zones_cfg["execution"]["padding_pct"] = float(args.zones_exec_padding_pct)
        grid_payload = build_grid_confluence(
            symbol=str(args.symbol),
            current_price=float(current_price),
            now_ts_ms=int(now_ts_ms),
            grid_pct=float(grid_pct),
            regs_by_tf={"5m": reg_low, "1h": reg_mid, "4h": reg_high},
            tf_importance={"5m": 1, "1h": 2, "4h": 3},
            mode=str(grid_mode),
            zones_cfg=zones_cfg,
        )

        grid_out: Path
        if str(args.grid_out).strip():
            grid_out = Path(str(args.grid_out))
        else:
            if str(grid_mode) == "zones":
                meta_cfg = (grid_payload.get("meta") or {}).get("zones_cfg") or {}
                mr = float((meta_cfg.get("macro") or {}).get("radius_pct") or 0.0)
                cr = float((meta_cfg.get("context") or {}).get("radius_pct") or 0.0)
                er = float((meta_cfg.get("execution") or {}).get("radius_pct") or 0.0)
                grid_out = pivot_dir / f"zones_{str(args.symbol)}_{str(triad)}_M{int(round(mr*100))}C{int(round(cr*100))}E{int(round(er*100))}.json"
            else:
                grid_pct_int = int(round(float(grid_pct) * 100.0))
                grid_out = pivot_dir / f"grid_{str(args.symbol)}_{str(triad)}_{grid_pct_int}pct.json"

        _write_json(grid_out, grid_payload, write=not bool(args.skip_write), skip_unchanged=bool(args.skip_write_unchanged))

        pivot_table = extract_execution_pivot_price_weight_table(grid_payload, current_price=float(current_price))
        _print_pivot_price_weight_table(
            "Pivot table (execution picks: price+weight)",
            pivot_table,
            max_rows=40,
        )

        if str(grid_payload.get("meta", {}).get("mode")) == "zones":
            _print_grid_zones(grid_payload)
        else:
            print(f"Grid cells (confluent across 5m/1h/4h): {len(grid_payload.get('cells') or [])}")
            for cell in (grid_payload.get("cells") or [])[:20]:
                role = str(cell.get("role"))
                gk = int(cell.get("grid_key") or 0)
                gl = float(cell.get("grid_level") or 0.0)
                picks = cell.get("picks") or []
                print(f"Grid cell role={role} key={gk} level={gl:.6f} picks={len(picks)}")
                for pick in picks[:3]:
                    members = pick.get("members") or {}
                    p5 = members.get("5m") or {}
                    p1 = members.get("1h") or {}
                    p4 = members.get("4h") or {}
                    score = pick.get("score") or {}
                    print(
                        f"  pick global={int(score.get('importance_global') or 0)} local={int(score.get('importance_local') or 0)} "
                        f"5m(dt={p5.get('dt')} bars_ago={p5.get('bars_ago')} lvl={p5.get('level')}) "
                        f"1h(bars_ago={p1.get('bars_ago')} lvl={p1.get('level')}) "
                        f"4h(bars_ago={p4.get('bars_ago')} lvl={p4.get('level')})"
                    )
        print(f"Wrote grid confluence: {grid_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
