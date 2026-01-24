from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.agents.triple_vwma_cycle_agent import TripleVwmaCycleAgent, TripleVwmaCycleAgentConfig
from libs.agents.macd_hist_tranche_agent import HistTrancheAgentConfig, MacdHistTrancheAgent
from libs.agents.triple_cci_tranche_agent import TripleCciTrancheAgent, TripleCciTrancheAgentConfig
from libs.indicators.moving_averages.vwma_tv import vwma_tv
from libs.indicators.momentum.macd_tv import macd_tv
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


def _find_ms_crosses_from_sep_ms(sep_ms: np.ndarray) -> list[tuple[int, str]]:
    n = int(len(sep_ms))
    if n < 2:
        return []
    out: list[tuple[int, str]] = []
    prev = int(sep_ms[0])
    for i in range(1, n):
        cur = int(sep_ms[i])
        if cur == 0:
            continue
        if prev == 0:
            prev = int(cur)
            continue
        if int(cur) != int(prev):
            out.append((int(i), "MS_CROSS_UP" if int(cur) == 1 else "MS_CROSS_DOWN"))
            prev = int(cur)
    return out


def _idx_range(cycles: list[dict]) -> tuple[int | None, int | None]:
    if not cycles:
        return None, None
    a = min(int(c.get("start_i") or 0) for c in cycles)
    b = max(int(c.get("end_i") or 0) for c in cycles)
    return a, b


def _ts_range(cycles: list[dict]) -> tuple[int | None, int | None]:
    if not cycles:
        return None, None
    a = min(int(c.get("start_ts") or 0) for c in cycles)
    b = max(int(c.get("end_ts") or 0) for c in cycles)
    return a, b


def _fmt_ts(ts: int | None) -> str:
    if ts is None or int(ts) <= 0:
        return ""
    return str(pd.to_datetime(int(ts), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC"))


def _find_fs_crosses(*, fast: np.ndarray, slow: np.ndarray) -> list[tuple[int, str]]:
    n = int(len(fast))
    if n < 2:
        return []
    diff = fast - slow
    sign = np.sign(diff)
    out: list[tuple[int, str]] = []
    for i in range(1, n):
        if not np.isfinite(float(sign[i - 1])) or not np.isfinite(float(sign[i])):
            continue
        if float(sign[i - 1]) <= 0.0 and float(sign[i]) > 0.0:
            out.append((int(i), "CROSS_UP"))
        elif float(sign[i - 1]) >= 0.0 and float(sign[i]) < 0.0:
            out.append((int(i), "CROSS_DOWN"))
    return out


def _find_last_fs_cross(*, fast: np.ndarray, slow: np.ndarray) -> tuple[int | None, str | None]:
    n = int(len(fast))
    if n < 2:
        return None, None
    diff = fast - slow
    sign = np.sign(diff)
    last_i: int | None = None
    last_kind: str | None = None
    for i in range(1, n):
        if not np.isfinite(float(sign[i - 1])) or not np.isfinite(float(sign[i])):
            continue
        if float(sign[i - 1]) <= 0.0 and float(sign[i]) > 0.0:
            last_i = int(i)
            last_kind = "CROSS_UP"
        elif float(sign[i - 1]) >= 0.0 and float(sign[i]) < 0.0:
            last_i = int(i)
            last_kind = "CROSS_DOWN"
    return last_i, last_kind


def _zones_at(*, v: float, radius_pct: float) -> tuple[float, float]:
    r = float(radius_pct)
    if r < 0.0:
        r = 0.0
    return float(v) * (1.0 + r), float(v) * (1.0 - r)


def _to_global_event(e: dict, *, cycle_start_i: int) -> dict | None:
    try:
        local_pos = int(e.get("pos"))
    except Exception:
        return None
    out = dict(e)
    out["pos"] = int(cycle_start_i) + int(local_pos)
    meta = out.get("meta")
    if isinstance(meta, dict):
        meta2 = dict(meta)
        for k in ("start_pos", "end_pos", "extreme_pos"):
            if k in meta2 and meta2[k] is not None:
                try:
                    meta2[k] = int(cycle_start_i) + int(meta2[k])
                except Exception:
                    pass
        out["meta"] = meta2
    return out


def _print_cycle(prefix: str, c: dict, *, chrono_pos: int | None = None) -> None:
    start_i = int(c.get("start_i") or 0)
    end_i = int(c.get("end_i") or 0)
    macro = str(c.get("macro_trend_side") or "")
    micro = str(c.get("micro_trend_side") or "")
    start_dt = str(c.get("start_dt") or "")
    end_dt = str(c.get("end_dt") or "")
    spread_ms_max = c.get("spread_ms_max_pct")
    spread_ms_end = c.get("spread_ms_end_pct")
    spread_fm_max = c.get("spread_fm_max_pct")
    spread_fm_end = c.get("spread_fm_end_pct")
    harmony = c.get("vwma_slope_harmony_ratio")
    score = float(c.get("score") or 0.0)
    pbw = int(c.get("pullback_weak_count") or 0)
    pbm = int(c.get("pullback_medium_count") or 0)
    pbs = int(c.get("pullback_strong_count") or 0)
    last_pb = c.get("last_pullback_kind")
    rec = c.get("last_pullback_recency")
    bsc = bool(c.get("break_slow_confirmed"))

    chrono_txt = f" chrono_pos={int(chrono_pos)}" if chrono_pos is not None else ""
    print(
        f"{prefix} | cycle={int(c.get('cycle_id') or 0)} | macro={macro} micro={micro}{chrono_txt} | "
        f"idx=[{start_i}..{end_i}] | dt=[{start_dt}..{end_dt}] | "
        f"spread_ms(max/end)={spread_ms_max}/{spread_ms_end} spread_fm(max/end)={spread_fm_max}/{spread_fm_end} | "
        f"harmony={harmony} | pullbacks: weak={pbw} med={pbm} strong={pbs} | last_pb={last_pb} rec={rec} break_slow_confirmed={bsc} | score={score:.4f}"
    )

    events = c.get("events") or []
    for e in events:
        if not isinstance(e, dict):
            continue
        pos = int(e.get("pos") or 0)
        dt = str(e.get("dt") or "")
        kind = str(e.get("kind") or "")
        meta = e.get("meta")
        print(f"  {pos} | {dt} | {kind} | meta={meta}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="")
    ap.add_argument("--symbol", default="LINKUSDT")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--limit", type=int, default=900)
    ap.add_argument("--category", default="linear")
    ap.add_argument("--base-url", default="https://api.bybit.com")

    ap.add_argument("--vwma-fast-len", type=int, default=12)
    ap.add_argument("--vwma-mid-len", type=int, default=72)
    ap.add_argument("--vwma-slow-len", type=int, default=168)

    ap.add_argument("--zone-fast-radius-pct", type=float, default=0.001)
    ap.add_argument("--zone-mid-radius-pct", type=float, default=0.001)
    ap.add_argument("--zone-slow-radius-pct", type=float, default=0.001)
    ap.add_argument("--zone-large-mult", type=float, default=2.0)

    ap.add_argument("--break-confirm-bars", type=int, default=3)
    ap.add_argument("--spread-ref-pct", type=float, default=0.002)
    ap.add_argument("--min-cycle-len", type=int, default=20)
    ap.add_argument("--min-score", type=float, default=0.05)

    ap.add_argument("--max-cycles", type=int, default=8)
    ap.add_argument("--top-n", type=int, default=6)

    ap.add_argument("--mode", choices=["analyze", "current"], default="analyze")

    ap.add_argument("--bt-mode", default="scalp")
    ap.add_argument("--tp-pct", type=float, default=0.007)
    ap.add_argument("--sl-buffer-pct", type=float, default=0.0025)
    ap.add_argument("--max-sl-pct", type=float, default=-1.0)
    ap.add_argument("--max-signals", type=int, default=100)
    ap.add_argument("--hist-eps", type=float, default=0.0)

    ap.add_argument("--cci-fast", type=int, default=30)
    ap.add_argument("--cci-medium", type=int, default=120)
    ap.add_argument("--cci-slow", type=int, default=300)
    ap.add_argument("--cci-strength-ref", type=float, default=100.0)
    ap.add_argument("--cci-extreme-level", type=float, default=100.0)
    ap.add_argument("--cci-min-confluence", type=int, default=2)

    ap.add_argument("--tranche-min-abs-force", type=float, default=0.0)
    ap.add_argument("--cci-global-max-abs", type=float, default=100.0)

    ap.add_argument("--full", action="store_true")
    ap.add_argument("--signals-verbose", action="store_true")
    ap.add_argument("--show-rejected", action="store_true")
    ap.add_argument("--min-bars-after-cross", type=int, default=50)
    ap.add_argument("--scan-scope", default="cycle", choices=["cycle", "all"])
    ap.add_argument("--max-cycle-events", type=int, default=80)

    args = ap.parse_args()

    df: pd.DataFrame
    if str(args.csv).strip():
        df = pd.read_csv(Path(str(args.csv)).expanduser())
        for col in ("ts", "open", "high", "low", "close"):
            if col not in df.columns:
                raise ValueError(f"CSV missing required column: {col}")
        if "volume" not in df.columns:
            df["volume"] = 1.0
        df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    else:
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
    if bool(args.full):
        print(f"Now: {now_utc_dt}")
        print(f"Data range: {first_dt} -> {last_dt} | last_ts={last_ts} | lag_min={lag_min:.2f}")

    close = pd.to_numeric(df["close"], errors="coerce").astype(float).tolist()
    volume = pd.to_numeric(df["volume"], errors="coerce").astype(float).tolist()
    high_list = pd.to_numeric(df["high"], errors="coerce").astype(float).tolist()
    low_list = pd.to_numeric(df["low"], errors="coerce").astype(float).tolist()

    fast_len = int(args.vwma_fast_len)
    mid_len = int(args.vwma_mid_len)
    slow_len = int(args.vwma_slow_len)

    vwma_fast_col = f"vwma_{fast_len}"
    vwma_mid_col = f"vwma_{mid_len}"
    vwma_slow_col = f"vwma_{slow_len}"

    df[vwma_fast_col] = vwma_tv(close, volume, int(fast_len))
    df[vwma_mid_col] = vwma_tv(close, volume, int(mid_len))
    df[vwma_slow_col] = vwma_tv(close, volume, int(slow_len))

    fast_arr = pd.to_numeric(df[vwma_fast_col], errors="coerce").astype(float).to_numpy()
    mid_arr = pd.to_numeric(df[vwma_mid_col], errors="coerce").astype(float).to_numpy()
    slow_arr = pd.to_numeric(df[vwma_slow_col], errors="coerce").astype(float).to_numpy()
    ts_arr = pd.to_numeric(df["ts"], errors="coerce").astype("Int64").to_numpy()
    open_arr = pd.to_numeric(df["open"], errors="coerce").astype(float).to_numpy()

    zmid = float(args.zone_mid_radius_pct)
    zslow = float(args.zone_slow_radius_pct)
    mu = mid_arr * (1.0 + zmid)
    ml = mid_arr * (1.0 - zmid)
    su = slow_arr * (1.0 + zslow)
    sl = slow_arr * (1.0 - zslow)
    finite_ms = np.isfinite(mu) & np.isfinite(ml) & np.isfinite(su) & np.isfinite(sl)
    sep_ms = np.zeros(int(len(df)), dtype=int)
    sep_ms = np.where(finite_ms & (ml > su), 1, sep_ms)
    sep_ms = np.where(finite_ms & (mu < sl), -1, sep_ms)

    fs_crosses = _find_fs_crosses(fast=fast_arr, slow=slow_arr)
    ms_crosses = _find_ms_crosses_from_sep_ms(sep_ms)

    last_fs_cross_i, last_fs_cross_kind = (fs_crosses[-1] if fs_crosses else (None, None))
    last_ms_cross_i, last_ms_cross_kind = (ms_crosses[-1] if ms_crosses else (None, None))
    min_after = int(args.min_bars_after_cross)
    if min_after < 0:
        min_after = 0

    scan_scope = str(args.scan_scope).strip().lower()
    if scan_scope not in {"cycle", "all"}:
        raise ValueError(f"Unsupported --scan-scope: {args.scan_scope}")

    bt_mode = str(args.bt_mode).strip().lower()
    if bt_mode not in {"scalp", "swing"}:
        raise ValueError(f"Unsupported --bt-mode: {args.bt_mode}")

    cross_src = "mid_slow" if bt_mode == "scalp" else "fast_slow"
    crosses = ms_crosses if cross_src == "mid_slow" else fs_crosses

    chosen_cross_i: int | None = None
    chosen_cross_kind: str | None = None
    if crosses and scan_scope == "cycle":
        for ci, ck in reversed(crosses):
            if int(ci) <= int(len(df) - 1 - int(min_after)):
                chosen_cross_i = int(ci)
                chosen_cross_kind = str(ck)
                break
        if chosen_cross_i is None:
            chosen_cross_i, chosen_cross_kind = crosses[-1]

    last_fs_dt = str(df["dt"].iloc[int(last_fs_cross_i)]) if last_fs_cross_i is not None and 0 <= int(last_fs_cross_i) < len(df) else ""
    last_ms_dt = str(df["dt"].iloc[int(last_ms_cross_i)]) if last_ms_cross_i is not None and 0 <= int(last_ms_cross_i) < len(df) else ""
    chosen_cross_dt = str(df["dt"].iloc[int(chosen_cross_i)]) if chosen_cross_i is not None and 0 <= int(chosen_cross_i) < len(df) else ""

    if bool(args.full):
        print(f"Last FS cross: i={last_fs_cross_i} kind={last_fs_cross_kind} dt='{last_fs_dt}'")
        print(f"Last MS cross: i={last_ms_cross_i} kind={last_ms_cross_kind} dt='{last_ms_dt}'")
        print(
            f"Chosen cross(src={cross_src}): i={chosen_cross_i} kind={chosen_cross_kind} dt='{chosen_cross_dt}' "
            f"min_bars_after_cross={min_after} scan_scope={scan_scope}"
        )
    else:
        if scan_scope == "cycle":
            print(f"CROSS src={cross_src} i={chosen_cross_i} kind={chosen_cross_kind} dt='{chosen_cross_dt}'")
            if crosses and chosen_cross_i == crosses[-1][0] and int(crosses[-1][0]) > int(len(df) - 1 - int(min_after)):
                print(
                    f"WARN last cross too recent for min_bars_after_cross={min_after}; increase --limit or decrease --min-bars-after-cross"
                )
        else:
            print(f"SCAN scope=all n_bars={int(len(df))} range_dt=['{first_dt}'..'{last_dt}']")

    # New scalp mode: strict VWMA alignment + MACD_hist sign change; no pullbacks.
    if bt_mode == "scalp":
        max_sl_pct = None if float(args.max_sl_pct) <= 0.0 else float(args.max_sl_pct)
        tp_pct = float(args.tp_pct)
        sl_buf = float(args.sl_buffer_pct)

        hist_eps = float(args.hist_eps)
        if not np.isfinite(float(hist_eps)) or float(hist_eps) < 0.0:
            hist_eps = 0.0

        warmup = max(int(fast_len), int(mid_len), int(slow_len), int(args.cci_slow), 35)
        start_i = int(warmup)
        scan_end_i = int(len(df) - 1)

        if scan_scope == "cycle":
            # Scan window: after chosen MS cross, until next MS cross (or end).
            if chosen_cross_i is not None:
                start_i = max(int(start_i), int(chosen_cross_i) + 1)
            if chosen_cross_i is not None:
                for ci, _ck in ms_crosses:
                    if int(ci) > int(chosen_cross_i):
                        scan_end_i = min(int(scan_end_i), int(ci) - 1)
                        break

        _macd_line_list, _macd_signal_list, hist_list = macd_tv(close, 12, 26, 9)
        macd_hist = np.asarray(hist_list, dtype=float)
        df["macd_hist"] = macd_hist

        cci_fast_list = cci_tv(high_list, low_list, close, int(args.cci_fast))
        cci_medium_list = cci_tv(high_list, low_list, close, int(args.cci_medium))
        cci_slow_list = cci_tv(high_list, low_list, close, int(args.cci_slow))

        cci_fast_col = f"cci_{int(args.cci_fast)}"
        cci_medium_col = f"cci_{int(args.cci_medium)}"
        cci_slow_col = f"cci_{int(args.cci_slow)}"
        df[cci_fast_col] = np.asarray(cci_fast_list, dtype=float)
        df[cci_medium_col] = np.asarray(cci_medium_list, dtype=float)
        df[cci_slow_col] = np.asarray(cci_slow_list, dtype=float)

        tranche_min_abs_force = float(args.tranche_min_abs_force)
        if (not np.isfinite(float(tranche_min_abs_force))) or float(tranche_min_abs_force) < 0.0:
            tranche_min_abs_force = 0.0

        cci_global_max_abs = float(args.cci_global_max_abs)
        if (not np.isfinite(float(cci_global_max_abs))) or float(cci_global_max_abs) < 0.0:
            cci_global_max_abs = 0.0

        def _f(x: object) -> float:
            try:
                y = float(x)  # type: ignore[arg-type]
            except Exception:
                return float("nan")
            if not np.isfinite(float(y)):
                return float("nan")
            return float(y)

        scan_last_i = int(min(int(scan_end_i), int(len(df) - 1)) - 1)
        if int(scan_last_i) < int(start_i):
            scan_last_i = int(start_i)

        hist_cfg = HistTrancheAgentConfig(
            ts_col="ts",
            close_col="close",
            hist_col="macd_hist",
            min_abs_force=float(tranche_min_abs_force),
        )
        hist_agent = MacdHistTrancheAgent(cfg=hist_cfg)
        tranches = hist_agent.analyze_df(df, max_tranches=0)

        cci_cfg = TripleCciTrancheAgentConfig(
            ts_col="ts",
            high_col="high",
            low_col="low",
            close_col="close",
            hist_col="macd_hist",
            extremes_on="close",
            cci_fast_col=str(cci_fast_col),
            cci_medium_col=str(cci_medium_col),
            cci_slow_col=str(cci_slow_col),
            cci_fast_period=int(args.cci_fast),
            cci_medium_period=int(args.cci_medium),
            cci_slow_period=int(args.cci_slow),
        )
        cci_agent = TripleCciTrancheAgent(cfg=cci_cfg)
        cci_tranches = cci_agent.analyze_df(df, max_tranches=0)
        cci_by_id = {int(m.tranche_id): m for m in cci_tranches}

        high_arr = pd.to_numeric(df["high"], errors="coerce").astype(float).to_numpy()
        low_arr = pd.to_numeric(df["low"], errors="coerce").astype(float).to_numpy()
        open_arr = pd.to_numeric(df["open"], errors="coerce").astype(float).to_numpy()

        def _extreme_in_window(a: int, b: int, *, side: str) -> tuple[int | None, float | None]:
            if int(a) < 0:
                a = 0
            if int(b) >= int(len(df)):
                b = int(len(df) - 1)
            if int(a) > int(b) or int(len(df)) <= 0:
                return None, None

            if str(side) == "LONG":
                w = low_arr[int(a) : int(b) + 1]
                if w.size <= 0 or (not np.isfinite(w).any()):
                    return None, None
                j = int(np.nanargmin(w))
                return int(a + j), float(w[j])

            w = high_arr[int(a) : int(b) + 1]
            if w.size <= 0 or (not np.isfinite(w).any()):
                return None, None
            j = int(np.nanargmax(w))
            return int(a + j), float(w[j])

        n_signals = 0
        for idx, tm in enumerate(tranches):
            if n_signals >= int(args.max_signals):
                break

            tid = int(tm.tranche_id)
            signal_i = int(tm.tranche_start_i)
            exec_i = int(signal_i + 1)
            if int(signal_i) < int(start_i) or int(signal_i) > int(scan_last_i):
                continue
            if int(exec_i) < 0 or int(exec_i) >= int(len(df)):
                continue

            side = "LONG" if str(tm.tranche_sign) == "+" else "SHORT"
            if not bool(tm.is_interesting):
                if bool(args.signals_verbose) and bool(args.show_rejected):
                    print(
                        "REJECT_SIGNAL"
                        f" side={side}"
                        f" reason=TRANCHE_FORCE_TOO_LOW"
                        f" tranche_id={tid} force_mean_abs={float(tm.force_mean_abs):.8f} min_abs_force={float(tranche_min_abs_force):.8f}"
                        f" signal_i={signal_i} signal_dt='{str(df['dt'].iloc[int(signal_i)])}'",
                        flush=True,
                    )
                continue

            cci_m = cci_by_id.get(int(tid))
            cci_abs = None if cci_m is None else cci_m.cci_global_last_extreme_abs
            if float(cci_global_max_abs) > 0.0:
                if cci_abs is None or (not math.isfinite(float(cci_abs))) or float(cci_abs) >= float(cci_global_max_abs):
                    if bool(args.signals_verbose) and bool(args.show_rejected):
                        print(
                            "REJECT_SIGNAL"
                            f" side={side}"
                            f" reason=CCI_GLOBAL_EXTREME"
                            f" tranche_id={tid} cci_global_last_extreme_abs={cci_abs} cci_global_max_abs={float(cci_global_max_abs)}"
                            f" signal_i={signal_i} signal_dt='{str(df['dt'].iloc[int(signal_i)])}'",
                            flush=True,
                        )
                    continue

            prev_tm = tranches[int(idx - 1)] if int(idx) > 0 else None
            if prev_tm is None:
                if bool(args.signals_verbose) and bool(args.show_rejected):
                    print(
                        "REJECT_SIGNAL"
                        f" side={side}"
                        f" reason=MISSING_PREV_TRANCHE"
                        f" tranche_id={tid}"
                        f" signal_i={signal_i} signal_dt='{str(df['dt'].iloc[int(signal_i)])}'",
                        flush=True,
                    )
                continue

            prev_a = int(prev_tm.tranche_start_i)
            prev_b = int(prev_tm.tranche_end_i)
            extreme_i, extreme_price = _extreme_in_window(prev_a, prev_b, side=str(side))
            if extreme_i is None or extreme_price is None:
                if bool(args.signals_verbose) and bool(args.show_rejected):
                    print(
                        "REJECT_SIGNAL"
                        f" side={side}"
                        f" reason=MISSING_OPPOSITE_TRANCHE_EXTREME"
                        f" tranche_id={tid}"
                        f" signal_i={signal_i} signal_dt='{str(df['dt'].iloc[int(signal_i)])}'",
                        flush=True,
                    )
                continue

            entry = float(open_arr[int(exec_i)])
            if (not np.isfinite(float(entry))) or float(entry) <= 0.0:
                continue
            dist_ext = abs(float(entry) - float(extreme_price)) / float(entry)
            if max_sl_pct is not None and np.isfinite(float(dist_ext)) and float(dist_ext) > float(max_sl_pct):
                if bool(args.signals_verbose) and bool(args.show_rejected):
                    print(
                        "SKIP_ENTRY"
                        f" side={side}"
                        f" tranche_id={tid}"
                        f" signal_i={signal_i} signal_dt='{str(df['dt'].iloc[int(signal_i)])}'"
                        f" exec_i={exec_i} entry_dt='{str(df['dt'].iloc[int(exec_i)])}' entry={float(entry):.6f}"
                        f" extreme_i={int(extreme_i)} extreme_dt='{str(df['dt'].iloc[int(extreme_i)])}' extreme={float(extreme_price):.6f}"
                        f" dist_ext_pct={100.0 * float(dist_ext):.3f} max_sl_pct={100.0 * float(max_sl_pct):.3f}",
                        flush=True,
                    )
                continue

            if side == "LONG":
                sl_price = float(extreme_price) * (1.0 - float(sl_buf))
                tp_price = float(entry) * (1.0 + float(tp_pct))
            else:
                sl_price = float(extreme_price) * (1.0 + float(sl_buf))
                tp_price = float(entry) * (1.0 - float(tp_pct))

            print(
                "ENTRY"
                f" side={side}"
                f" tranche_id={tid} tranche_sign={str(tm.tranche_sign)}"
                f" force_mean_abs={float(tm.force_mean_abs):.8f}"
                + (f" cci_global_last_extreme_abs={float(cci_abs):.2f}" if cci_abs is not None and math.isfinite(float(cci_abs)) else "")
                + f" signal_i={signal_i} signal_dt='{str(df['dt'].iloc[int(signal_i)])}'"
                + f" exec_i={exec_i} entry_dt='{str(df['dt'].iloc[int(exec_i)])}' entry={float(entry):.6f}"
                + f" extreme_i={int(extreme_i)} extreme_dt='{str(df['dt'].iloc[int(extreme_i)])}' extreme={float(extreme_price):.6f}"
                + f" sl={float(sl_price):.6f} tp={float(tp_price):.6f}",
                flush=True,
            )
            n_signals += 1

        if int(n_signals) == 0:
            scan_end_dt = str(df["dt"].iloc[int(scan_end_i)]) if 0 <= int(scan_end_i) < len(df) else ""
            print(
                f"NO_ENTRY scan_scope={scan_scope} start_i={start_i} scan_end_i={scan_end_i} scan_end_dt='{scan_end_dt}'",
                flush=True,
            )

    agent_cfg = TripleVwmaCycleAgentConfig(
        ts_col="ts",
        open_col="open",
        high_col="high",
        low_col="low",
        close_col="close",
        vwma_fast_col=str(vwma_fast_col),
        vwma_mid_col=str(vwma_mid_col),
        vwma_slow_col=str(vwma_slow_col),
        zone_fast_radius_pct=float(args.zone_fast_radius_pct),
        zone_mid_radius_pct=float(args.zone_mid_radius_pct),
        zone_slow_radius_pct=float(args.zone_slow_radius_pct),
        zone_large_mult=float(args.zone_large_mult),
        break_confirm_bars=int(args.break_confirm_bars),
        spread_ref_pct=float(args.spread_ref_pct),
        min_cycle_len=int(args.min_cycle_len),
        min_score=float(args.min_score),
    )

    agent = TripleVwmaCycleAgent(cfg=agent_cfg)
    ans = agent.answer(
        question={
            "kind": str(args.mode),
            "max_cycles": int(args.max_cycles),
        },
        df=df,
    )

    if (not bool(args.full)) and str(args.mode) == "current":
        m = ans.get("metric")
        if m is None:
            print("(none)")
        else:
            _print_cycle("order=current", m)
        return 0

    cycles_all = agent.analyze_df(df, max_cycles=0)

    chosen_cycle_start_i: int | None = None
    chosen_cycle_end_i: int | None = None

    if (not bool(args.full)) and chosen_cross_i is not None:
        last_cycle = None
        for cyc in cycles_all:
            cd = asdict(cyc)
            a = int(cd.get("start_i") or 0)
            b = int(cd.get("end_i") or 0)
            if int(a) <= int(chosen_cross_i) <= int(b):
                last_cycle = cd
        if last_cycle is not None:
            cid = int(last_cycle.get("cycle_id") or 0)
            a = int(last_cycle.get("start_i") or 0)
            b = int(last_cycle.get("end_i") or 0)
            chosen_cycle_start_i = int(a)
            chosen_cycle_end_i = int(b)
            sdt = str(last_cycle.get("start_dt") or "")
            edt = str(last_cycle.get("end_dt") or "")
            macro = str(last_cycle.get("macro_trend_side") or "")
            micro = str(last_cycle.get("micro_trend_side") or "")
            print(f"CYCLE id={cid} idx=[{a}..{b}] dt=['{sdt}'..'{edt}'] macro={macro} micro={micro}")
            kinds_keep = {"vwma_cross", "pullback_start", "pullback_weak", "pullback_medium", "pullback_strong", "slow_break_confirmed"}
            evs = []
            for e in (last_cycle.get("events") or []):
                if not isinstance(e, dict):
                    continue
                try:
                    pos_local = int(e.get("pos") or 0)
                except Exception:
                    continue
                pos = int(a) + int(pos_local)
                kind = str(e.get("kind") or "")
                if kind not in kinds_keep:
                    continue
                if int(pos) < int(chosen_cross_i):
                    continue
                evs.append((int(pos), e))
            for pos, e in evs[: int(max(1, int(args.max_cycle_events)))] :
                kind = str(e.get("kind") or "")
                dt = str(e.get("dt") or "")
                meta = e.get("meta") or {}
                extra = ""
                if isinstance(meta, dict) and kind.startswith("pullback_"):
                    extra = f" extreme={meta.get('extreme_price')} stage={meta.get('stage')}"
                print(f"EV i={int(pos)} dt='{dt}' kind={kind}{extra}")

    if bool(args.full):
        ans_rank = agent.answer(
            question={
                "kind": "rank_triple_vwma_cycles",
                "top_n": int(args.top_n),
                "max_cycles": int(args.max_cycles),
            },
            df=df,
        )

        ranked = ans_rank.get("ranked") or []
        interesting = ans_rank.get("interesting") or []

        cycles_sel = agent.analyze_df(df, max_cycles=int(args.max_cycles))
        cycles_all_dicts = [asdict(x) for x in cycles_all]
        cycles_sel_dicts = [asdict(x) for x in cycles_sel]

        chrono_map = {int(c.get("cycle_id") or 0): i for i, c in enumerate(cycles_sel_dicts)}

        print("\n=== Selected cycles (chronological) ===")
        for i, c in enumerate(cycles_sel_dicts):
            _print_cycle(f"order=chrono pos={i}", c)

        if cycles_sel_dicts:
            print("\n=== Current (ongoing) cycle ===")
            _print_cycle("order=chrono current", cycles_sel_dicts[-1], chrono_pos=int(len(cycles_sel_dicts) - 1))

        print("\n=== Ranked cycles (by score; not chronological) ===")
        for r, c in enumerate(ranked, start=1):
            cp = chrono_map.get(int(c.get("cycle_id") or 0))
            _print_cycle(f"order=ranked(score) rank={r}", c, chrono_pos=cp)

        print("\n=== Interesting cycles (by score; not chronological) ===")
        for r, c in enumerate(interesting, start=1):
            cp = chrono_map.get(int(c.get("cycle_id") or 0))
            _print_cycle(f"order=interesting rank={r}", c, chrono_pos=cp)

        print("\n=== Summary ===")
        all_a, all_b = _idx_range(cycles_all_dicts)
        sel_a, sel_b = _idx_range(cycles_sel_dicts)
        all_ts0, all_ts1 = _ts_range(cycles_all_dicts)
        sel_ts0, sel_ts1 = _ts_range(cycles_sel_dicts)

        print(f"Cycles(all): n={len(cycles_all_dicts)} idx=[{all_a}..{all_b}] dt=[{_fmt_ts(all_ts0)}..{_fmt_ts(all_ts1)}]")
        print(f"Cycles(selected): n={len(cycles_sel_dicts)} idx=[{sel_a}..{sel_b}] dt=[{_fmt_ts(sel_ts0)}..{_fmt_ts(sel_ts1)}]")
    else:
        metrics = list(ans.get("metrics") or [])
        if not metrics:
            print("(none)")
            return 0
        for i, c in enumerate(metrics, start=1):
            _print_cycle(f"order=chrono pos={i}/{len(metrics)}", c)

    tp_pct = float(args.tp_pct)
    sl_buf = float(args.sl_buffer_pct)
    max_sl_pct = None if float(args.max_sl_pct) <= 0.0 else float(args.max_sl_pct)
    max_signals = int(args.max_signals)
    if max_signals < 1:
        max_signals = 1

    events_by_pos: dict[int, list[dict]] = {}
    for cyc in cycles_all:
        cd = asdict(cyc)
        start_i = int(cd.get("start_i") or 0)
        for e in (cd.get("events") or []):
            if not isinstance(e, dict):
                continue
            eg = _to_global_event(e, cycle_start_i=int(start_i))
            if eg is None:
                continue
            events_by_pos.setdefault(int(eg["pos"]), []).append(eg)

    warmup = max(int(fast_len), int(mid_len), int(slow_len))
    start_i = int(warmup)
    if scan_scope == "cycle" and chosen_cross_i is not None:
        start_i = max(int(start_i), int(chosen_cross_i) + 1)

    # In concise mode, keep the scan coherent with the printed cycle.
    scan_end_i = int(len(df) - 1)
    if (not bool(args.full)) and chosen_cycle_end_i is not None:
        scan_end_i = min(int(scan_end_i), int(chosen_cycle_end_i))

    scan_end_dt = str(df["dt"].iloc[int(scan_end_i)]) if 0 <= int(scan_end_i) < len(df) else ""

    if bool(args.full):
        print("\n=== Backtest-style entry candidates (after last cross) ===")
        print(
            f"mode={bt_mode} tp_pct={tp_pct} sl_buffer_pct={sl_buf} max_sl_pct={(None if max_sl_pct is None else max_sl_pct)} "
            f"start_i={start_i} start_dt='{str(df['dt'].iloc[int(start_i)]) if 0 <= int(start_i) < len(df) else ''}' "
            f"scan_end_i={scan_end_i} scan_end_dt='{scan_end_dt}'"
        )

    n_signals = 0
    n_pullback_events = 0
    n_clear_bars = 0
    n_side_none = 0
    for i in range(int(start_i), int(min(int(scan_end_i), int(len(df) - 1)))):
        if n_signals >= int(max_signals):
            break

        evs = events_by_pos.get(int(i), [])
        if not evs:
            continue

        side = None
        if int(sep_ms[i]) == 1:
            side = "LONG"
        elif int(sep_ms[i]) == -1:
            side = "SHORT"
        if side is None:
            n_side_none += 1
            continue
        n_clear_bars += 1

        for e in evs:
            kind = str(e.get("kind") or "")
            if bt_mode == "scalp":
                if kind not in {"pullback_weak", "pullback_medium"}:
                    continue
            else:
                if kind not in {"pullback_weak", "pullback_medium", "pullback_strong"}:
                    continue

            n_pullback_events += 1

            meta = e.get("meta") or {}
            extreme_price = meta.get("extreme_price")
            extreme_pos = meta.get("extreme_pos")
            try:
                exf = float(extreme_price) if extreme_price is not None else None
            except Exception:
                exf = None
            try:
                exp = int(extreme_pos) if extreme_pos is not None else None
            except Exception:
                exp = None
            if exf is None or (not np.isfinite(float(exf))) or float(exf) <= 0.0:
                continue
            if exp is None or int(exp) < 0 or int(exp) >= int(len(df)):
                exp = int(i)

            exec_i = int(i + 1)
            entry = float(open_arr[int(exec_i)])
            dist_ext = abs(float(entry) - float(exf)) / float(entry) if float(entry) > 0.0 else float("nan")
            if max_sl_pct is not None and np.isfinite(float(dist_ext)) and float(dist_ext) > float(max_sl_pct):
                continue

            if side == "LONG":
                sl_price = float(exf) * (1.0 - float(sl_buf))
                tp_price = float(entry) * (1.0 + float(tp_pct))
            else:
                sl_price = float(exf) * (1.0 + float(sl_buf))
                tp_price = float(entry) * (1.0 - float(tp_pct))

            fu, fl = _zones_at(v=float(fast_arr[int(exp)]), radius_pct=float(args.zone_fast_radius_pct))
            mu, ml = _zones_at(v=float(mid_arr[int(exp)]), radius_pct=float(args.zone_mid_radius_pct))
            su, slz = _zones_at(v=float(slow_arr[int(exp)]), radius_pct=float(args.zone_slow_radius_pct))
            in_fast = bool(float(fl) <= float(exf) <= float(fu))
            in_mid = bool(float(ml) <= float(exf) <= float(mu))
            in_slow = bool(float(slz) <= float(exf) <= float(su))

            ts_i = int(ts_arr[int(i)]) if 0 <= int(i) < len(ts_arr) and ts_arr[int(i)] is not None else 0
            dt_i = str(df["dt"].iloc[int(i)])
            dt_exec = str(df["dt"].iloc[int(exec_i)])
            dt_ex = str(df["dt"].iloc[int(exp)])

            if bool(args.signals_verbose) or bool(args.full):
                print(
                    "ENTRY_CAND"
                    f" ev_i={i} ev_dt='{dt_i}'"
                    f" kind={kind}"
                    f" side={side}"
                    f" exec_i={exec_i} exec_dt='{dt_exec}' entry={entry:.6f}"
                    f" extreme_i={int(exp)} extreme_dt='{dt_ex}' extreme={float(exf):.6f}"
                    f" extreme_in(fast/mid/slow)={int(in_fast)}/{int(in_mid)}/{int(in_slow)}"
                    f" sl={sl_price:.6f} tp={tp_price:.6f}"
                    f" dist_ext_pct={100.0 * float(dist_ext):.3f}",
                    flush=True,
                )
            else:
                print(
                    "ENTRY"
                    f" kind={kind}"
                    f" side={side}"
                    f" signal_i={i} signal_dt='{dt_i}'"
                    f" exec_i={exec_i}"
                    f" entry_dt='{dt_exec}' entry={entry:.6f}"
                    f" sl={sl_price:.6f}",
                    flush=True,
                )
            n_signals += 1
            if n_signals >= int(max_signals):
                break
        if n_signals >= int(max_signals):
            break

    if not bool(args.full) and int(n_signals) == 0:
        print(
            f"NO_ENTRY after_cross_i={chosen_cross_i} start_i={start_i} bars_after_cross={(None if chosen_cross_i is None else int(len(df) - 1 - int(chosen_cross_i)))} "
            f"scan_end_i={scan_end_i} scan_end_dt='{scan_end_dt}' clear_bars={n_clear_bars} side_none_bars={n_side_none} pullback_events_seen={n_pullback_events}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
