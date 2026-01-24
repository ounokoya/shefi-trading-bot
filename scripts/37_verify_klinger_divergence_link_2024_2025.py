from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.indicators.volume.klinger_oscillator_tv import klinger_oscillator_tv  # noqa: E402
from libs.indicators.volume.nvi_tv import nvi_tv  # noqa: E402
from libs.indicators.volume.pvi_tv import pvi_tv  # noqa: E402
from libs.indicators.volume.pvt_tv import pvt_tv  # noqa: E402
from libs.strategies.klinger_cci_extremes import KlingerCciExtremesConfig, analyze_klinger_cci_tranches  # noqa: E402


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


def _cache_path_for(
    *,
    cache_dir: Path,
    symbol: str,
    interval: str,
    start: str,
    end: str,
) -> Path:
    safe_symbol = str(symbol).replace("/", "_").replace(":", "_")
    safe_interval = str(interval).replace("/", "_").replace(":", "_")
    safe_start = str(start).replace(":", "-").replace(" ", "_")
    safe_end = str(end).replace(":", "-").replace(" ", "_")
    return cache_dir / f"bybit_klines_{safe_symbol}_{safe_interval}_{safe_start}_{safe_end}.csv"


def _fetch_bybit_klines_range(
    *,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    category: str,
    base_url: str,
    timeout_s: float,
    max_pages: int = 50,
) -> pd.DataFrame:
    import time

    import requests

    url = f"{base_url.rstrip('/')}/v5/market/kline"

    out: list[dict[str, object]] = []
    seen: set[int] = set()
    cursor_end = int(end_ms)

    pages = 0
    while pages < int(max_pages):
        pages += 1

        params: dict[str, Any] = {
            "category": str(category),
            "symbol": str(symbol),
            "interval": str(interval),
            "limit": "1000",
            "end": str(int(cursor_end)),
        }

        r = requests.get(url, params=params, timeout=float(timeout_s))
        r.raise_for_status()
        payload = r.json()
        if str(payload.get("retCode")) != "0":
            raise RuntimeError(f"Bybit retCode={payload.get('retCode')} retMsg={payload.get('retMsg')}")

        result = payload.get("result") or {}
        rows = result.get("list") or []
        if not rows:
            break

        min_ts_in_page: int | None = None
        for row in rows:
            if not isinstance(row, list) or len(row) < 6:
                continue
            try:
                ts = int(row[0])
            except Exception:
                continue
            if ts < int(start_ms) or ts > int(end_ms):
                continue
            if ts in seen:
                continue
            seen.add(ts)

            turnover = float("nan")
            if len(row) >= 7:
                try:
                    turnover = float(row[6])
                except Exception:
                    turnover = float("nan")

            out.append(
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
            if min_ts_in_page is None or ts < int(min_ts_in_page):
                min_ts_in_page = int(ts)

        if min_ts_in_page is None:
            break

        # move cursor backwards
        cursor_end = int(min_ts_in_page) - 1
        if cursor_end < int(start_ms):
            break

        # small sleep to be gentle on rate limits
        time.sleep(0.05)

    df = pd.DataFrame(out)
    if len(df):
        df = df.sort_values("ts").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return df


def _safe_float(v: object) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None

    if not math.isfinite(x):
        return None
    return float(x)


def _campaign_tranche_max_return(
    *,
    tranches: list[dict[str, object]],
    start_i: int,
    end_i: int,
    entry_price: float,
) -> dict[str, object]:
    if entry_price <= 0.0 or int(start_i) < 0 or int(end_i) <= int(start_i):
        return {
            "tranche_max_return": pd.NA,
            "tranche_max_level_reached_p": pd.NA,
            "tranche_max_dt": pd.NA,
            "tranche_max_i": pd.NA,
            "tranche_long_scanned": 0,
        }

    best_ret = None
    best_dt = None
    best_i = None
    scanned = 0

    for t in tranches:
        if str(t.get("side") or "") != "LONG":
            continue
        ext_i = int(t.get("kvo_extreme_i") or -1)
        if ext_i <= int(start_i) or ext_i > int(end_i):
            continue
        px = _safe_float(t.get("close_at_extreme"))
        if px is None:
            continue
        scanned += 1
        ret = (float(px) / float(entry_price)) - 1.0
        if best_ret is None or float(ret) > float(best_ret):
            best_ret = float(ret)
            best_dt = str(t.get("kvo_extreme_dt") or t.get("end_dt") or t.get("start_dt") or "")
            best_i = int(ext_i)

    return {
        "tranche_max_return": float(best_ret) if best_ret is not None else pd.NA,
        "tranche_max_level_reached_p": float(best_ret) * 100.0 if best_ret is not None else pd.NA,
        "tranche_max_dt": str(best_dt) if best_dt is not None else pd.NA,
        "tranche_max_i": int(best_i) if best_i is not None else pd.NA,
        "tranche_long_scanned": int(scanned),
    }


def _first_threshold_hit(*, ret: float, thresholds: list[int]) -> int | None:
    for t in sorted(thresholds):
        if float(ret) >= float(t) / 100.0:
            return int(t)
    return None


def _build_accum_campaigns(
    *,
    df: pd.DataFrame,
    dfp_acc: pd.DataFrame,
    close_s: pd.Series,
    kvo_s: pd.Series,
    ksig_s: pd.Series,
    thresholds: list[int],
    max_bottom_breaks: int,
    lookahead: int,
    require_signal: bool,
) -> pd.DataFrame:
    """Merge consecutive ACCUM signals into campaigns.

    Campaign rules:
    - Start at first ACCUM E2.
    - While not confirmed and not retested-bottom, any new ACCUM signal joins the campaign.
    - Confirmation: price reaches ANY threshold (e.g. +2%) from campaign entry price.
    - Retest-bottom: BOTH kvo and kvo_signal < campaign bottoms (min over included signals' formation bottoms).
    """
    if not len(dfp_acc):
        return pd.DataFrame()

    acc = dfp_acc.sort_values("e2_i").reset_index(drop=True)
    acc_rows = acc.to_dict("records")
    n = len(acc_rows)

    campaigns: list[dict[str, object]] = []
    i_sig = 0
    while i_sig < n:
        r0 = acc_rows[i_sig]
        start_i = int(r0.get("e2_i") or -1)
        if start_i < 0:
            i_sig += 1
            continue

        entry_close0 = _safe_float(close_s.iloc[int(start_i)])
        if entry_close0 is None or float(entry_close0) <= 0.0:
            i_sig += 1
            continue

        # DCA entry price: equal weight per absorbed ACCUM signal
        entry_sum = float(entry_close0)
        entry_n = 1
        entry_avg = float(entry_sum / entry_n)

        bottom_kvo = _safe_float(r0.get("v_kvo_bottom"))
        bottom_sig = _safe_float(r0.get("v_sig_bottom"))
        if bottom_kvo is None or bottom_sig is None:
            i_sig += 1
            continue

        last_sig_i = int(start_i)
        sig_count = 1

        breaks_used = 0
        break_dt_list: list[str] = []

        confirmed = False
        confirm_i = None
        confirm_level = None

        # After confirmation, we keep scanning prices for a limited window to measure
        # how far the move went. This does NOT change campaign merging rules.
        scan_after_confirm_until_i: int | None = None
        eval_end_i: int | None = None

        thr_first_days: dict[int, int] = {}

        best_ret = None
        best_day = None

        j = int(start_i) + 1
        next_sig = i_sig + 1
        # Scan forward until confirm or retest
        while j < int(len(close_s)):
            # absorb new ACCUM signals arriving before terminal event
            while (not bool(confirmed)) and next_sig < n:
                nxt = acc_rows[next_sig]
                nxt_i = int(nxt.get("e2_i") or -1)
                if nxt_i == j:
                    # DCA update: new ACCUM signal absorbed => new average entry
                    nxt_close = _safe_float(close_s.iloc[int(nxt_i)])
                    if nxt_close is not None and float(nxt_close) > 0.0:
                        entry_sum += float(nxt_close)
                        entry_n += 1
                        entry_avg = float(entry_sum / entry_n)

                    bk = _safe_float(nxt.get("v_kvo_bottom"))
                    bs = _safe_float(nxt.get("v_sig_bottom"))
                    if bk is not None:
                        bottom_kvo = float(min(float(bottom_kvo), float(bk)))
                    if bs is not None:
                        bottom_sig = float(min(float(bottom_sig), float(bs)))
                    last_sig_i = int(nxt_i)
                    sig_count += 1
                    next_sig += 1
                    continue
                if nxt_i < j:
                    next_sig += 1
                    continue
                break

            c = _safe_float(close_s.iloc[int(j)])
            if c is not None and float(entry_avg) > 0.0:
                ret = (float(c) / float(entry_avg)) - 1.0
                if best_ret is None or float(ret) > float(best_ret):
                    best_ret = float(ret)
                    best_day = int(j - int(start_i))

                # record first-hit day for each level
                for t in sorted(thresholds):
                    if int(t) not in thr_first_days and float(ret) >= float(t) / 100.0:
                        thr_first_days[int(t)] = int(j - int(start_i))

                hit = _first_threshold_hit(ret=float(ret), thresholds=thresholds)
                if hit is not None:
                    confirmed = True
                    confirm_i = int(j)
                    # at confirm bar, store the highest level reached
                    confirm_level = max([t for t in thresholds if float(ret) >= float(t) / 100.0])

                    # keep scanning for max return / max level within lookahead window
                    la = int(lookahead)
                    if la > 0:
                        scan_after_confirm_until_i = min(int(len(close_s) - 1), int(confirm_i) + int(la))
                    else:
                        scan_after_confirm_until_i = int(confirm_i)

                    eval_end_i = int(scan_after_confirm_until_i)

            kv = _safe_float(kvo_s.iloc[int(j)])
            sg = _safe_float(ksig_s.iloc[int(j)])
            if kv is not None and sg is not None:
                if (not bool(require_signal) and float(kv) < float(bottom_kvo)) or (
                    bool(require_signal) and float(kv) < float(bottom_kvo) and float(sg) < float(bottom_sig)
                ):
                    # bottom break
                    if int(breaks_used) < int(max_bottom_breaks):
                        breaks_used += 1
                        if "dt" in df.columns:
                            break_dt_list.append(str(df.iloc[int(j)]["dt"]))
                        # update bottom reference to this deeper break
                        bottom_kvo = float(min(float(bottom_kvo), float(kv)))
                        if bool(require_signal):
                            bottom_sig = float(min(float(bottom_sig), float(sg)))
                        j += 1
                        continue
                    break

            if bool(confirmed) and scan_after_confirm_until_i is not None:
                if int(j) >= int(scan_after_confirm_until_i):
                    break

            j += 1

        # terminal bar is either confirm_i (if confirmed) or retest at j (if not)
        retest_i = None if confirmed else (int(j) if j < int(len(close_s)) else None)
        end_i = int(confirm_i) if confirmed and confirm_i is not None else (int(retest_i) if retest_i is not None else int(len(close_s) - 1))
        if not bool(confirmed):
            eval_end_i = int(retest_i) if retest_i is not None else int(len(close_s) - 1)
        if eval_end_i is None:
            eval_end_i = int(end_i)

        mae_p = pd.NA
        mdd_p = pd.NA
        if float(entry_avg) > 0.0 and int(eval_end_i) > int(start_i):
            w = pd.to_numeric(close_s.iloc[int(start_i) + 1 : int(eval_end_i) + 1], errors="coerce").astype(float)
            w = w.dropna()
            if len(w):
                # MAE: worst drop below entry (close-only)
                mae = (float(w.min()) / float(entry_avg)) - 1.0
                mae_p = float(mae) * 100.0

                # MDD: maximum drawdown from a peak (close-only)
                peak = None
                best_dd = 0.0
                for x in w.tolist():
                    if peak is None or float(x) > float(peak):
                        peak = float(x)
                        continue
                    if float(peak) > 0.0:
                        dd = 1.0 - (float(x) / float(peak))
                        if float(dd) > float(best_dd):
                            best_dd = float(dd)
                mdd_p = float(best_dd) * 100.0

        max_level_reached = pd.NA
        if len(thr_first_days):
            max_level_reached = int(max(thr_first_days.keys()))

        campaigns.append(
            {
                "campaign_start_i": int(start_i),
                "campaign_start_dt": str(df.iloc[int(start_i)]["dt"]) if int(start_i) < int(len(df)) and "dt" in df.columns else pd.NA,
                "campaign_last_sig_i": int(last_sig_i),
                "campaign_last_sig_dt": str(df.iloc[int(last_sig_i)]["dt"]) if int(last_sig_i) < int(len(df)) and "dt" in df.columns else pd.NA,
                "campaign_signals": int(sig_count),
                "campaign_entry_close": float(entry_close0),
                "campaign_entry_avg": float(entry_avg),
                "campaign_bottom_kvo": float(bottom_kvo),
                "campaign_bottom_sig": float(bottom_sig),
                "campaign_breaks_used": int(breaks_used),
                "campaign_break_dts": ";".join(break_dt_list) if break_dt_list else "",
                "campaign_confirmed": bool(confirmed),
                "campaign_confirm_i": int(confirm_i) if confirm_i is not None else pd.NA,
                "campaign_confirm_dt": str(df.iloc[int(confirm_i)]["dt"]) if confirm_i is not None and int(confirm_i) < int(len(df)) and "dt" in df.columns else pd.NA,
                "campaign_confirm_level_p": int(confirm_level) if confirm_level is not None else pd.NA,
                "campaign_days_to_confirm": int(confirm_i - start_i) if confirmed and confirm_i is not None else pd.NA,
                "campaign_retest_i": int(retest_i) if retest_i is not None else pd.NA,
                "campaign_retest_dt": str(df.iloc[int(retest_i)]["dt"]) if retest_i is not None and int(retest_i) < int(len(df)) and "dt" in df.columns else pd.NA,
                "campaign_eval_end_i": int(eval_end_i) if eval_end_i is not None else pd.NA,
                "campaign_max_return": float(best_ret) if best_ret is not None else pd.NA,
                "campaign_days_to_max_return": int(best_day) if best_day is not None else pd.NA,
                "campaign_max_level_reached_p": max_level_reached,
                "campaign_mae_p": mae_p,
                "campaign_mdd_p": mdd_p,
                "campaign_thr_first_days": json.dumps({str(k): int(v) for k, v in thr_first_days.items()}),
                "campaign_end_i": int(end_i),
            }
        )

        # Move to next unused signal (signals with e2_i <= end_i are consumed)
        while next_sig < n:
            nxt_i = int(acc_rows[next_sig].get("e2_i") or -1)
            if nxt_i >= 0 and nxt_i <= int(end_i):
                next_sig += 1
            else:
                break
        i_sig = int(next_sig)

    return pd.DataFrame(campaigns)


def _find_retest_bottom_pos(
    *,
    kvo_s: pd.Series,
    ksig_s: pd.Series,
    start_i: int,
    bottom_kvo: float,
    bottom_sig: float,
    require_signal: bool,
) -> int | None:
    """Return first index >= start_i where BOTH kvo and kvo_signal are below their bottoms."""
    if int(start_i) < 0:
        return None
    if int(start_i) >= int(len(kvo_s)) or int(start_i) >= int(len(ksig_s)):
        return None
    for i in range(int(start_i), int(min(len(kvo_s), len(ksig_s)))):
        kv = _safe_float(kvo_s.iloc[int(i)])
        sg = _safe_float(ksig_s.iloc[int(i)])
        if kv is None or sg is None:
            continue
        if not bool(require_signal):
            if float(kv) < float(bottom_kvo):
                return int(i)
            continue
        if float(kv) < float(bottom_kvo) and float(sg) < float(bottom_sig):
            return int(i)
    return None


def _window_min(*, s: pd.Series, start_i: int, end_i: int) -> float | None:
    if int(start_i) < 0 or int(end_i) < 0:
        return None
    if int(end_i) < int(start_i):
        return None
    if int(start_i) >= int(len(s)):
        return None
    end_i = min(int(end_i), int(len(s) - 1))
    w = pd.to_numeric(s.iloc[int(start_i) : int(end_i) + 1], errors="coerce").astype(float)
    if not len(w.dropna()):
        return None
    return float(w.min())


def _outcome_after(
    *,
    df: pd.DataFrame,
    i: int,
    lookahead: int,
    up_pct: float,
    down_pct: float,
) -> str:
    if int(lookahead) <= 0:
        return ""
    if int(i) < 0 or int(i) >= int(len(df) - 1):
        return ""

    entry = _safe_float(df.iloc[int(i)]["close"])
    if entry is None:
        return ""

    end_i = min(int(len(df) - 1), int(i) + int(lookahead))
    w = df.iloc[int(i) + 1 : int(end_i) + 1]
    if not len(w):
        return ""

    highs = pd.to_numeric(w["high"], errors="coerce").astype(float)
    lows = pd.to_numeric(w["low"], errors="coerce").astype(float)

    max_up = (float(highs.max()) / float(entry)) - 1.0
    max_down = 1.0 - (float(lows.min()) / float(entry))

    if float(max_up) >= float(up_pct):
        return "reversal_up"
    if float(max_down) >= float(down_pct):
        return "continuation_down"
    return "noise"


@dataclass(frozen=True)
class PairDelta:
    kind: str  # continuation_vs_divergence
    price_delta: float
    kvo_delta: float


def _run_one_symbol(args: argparse.Namespace, *, symbol: str) -> dict[str, object]:
    start_dt = pd.to_datetime(str(args.year_start) + " 00:00:00", utc=True)
    end_dt = pd.to_datetime(str(args.year_end) + " 23:59:59", utc=True)
    start_ms = int(start_dt.value // 10**6)
    end_ms = int(end_dt.value // 10**6)

    interval = _interval_to_bybit(str(args.interval))

    cache_dir = Path(str(args.cache_dir)).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path_for(
        cache_dir=cache_dir,
        symbol=str(symbol),
        interval=str(args.interval),
        start=str(args.year_start),
        end=str(args.year_end),
    )

    df = pd.DataFrame()
    if cache_path.exists() and not bool(args.force_refresh):
        try:
            df = pd.read_csv(cache_path)
        except Exception:
            df = pd.DataFrame()

    if not len(df):
        df = _fetch_bybit_klines_range(
            symbol=str(symbol),
            interval=str(interval),
            start_ms=int(start_ms),
            end_ms=int(end_ms),
            category=str(args.category),
            base_url=str(args.base_url),
            timeout_s=30.0,
            max_pages=200,
        )
        try:
            df.to_csv(cache_path, index=False)
        except Exception:
            pass

    if not len(df):
        return {"symbol": str(symbol), "error": "no_klines"}

    vol_col = "volume" if str(args.volume_source) == "volume" else "turnover"
    if vol_col not in df.columns:
        return {"symbol": str(symbol), "error": f"missing_volume_col:{vol_col}"}
    df["volume"] = pd.to_numeric(df[vol_col], errors="coerce").astype(float)

    high_l = pd.to_numeric(df["high"], errors="coerce").astype(float).tolist()
    low_l = pd.to_numeric(df["low"], errors="coerce").astype(float).tolist()
    close_l = pd.to_numeric(df["close"], errors="coerce").astype(float).tolist()
    vol_l = pd.to_numeric(df["volume"], errors="coerce").astype(float).tolist()

    flow_indicator = str(args.flow_indicator).strip().lower()
    flow_signal_period = int(args.flow_signal_period)

    flow_has_signal = bool(flow_indicator in {"klinger", "kvo"})

    if flow_indicator in {"klinger", "kvo"}:
        kvo_l, ksig_l = klinger_oscillator_tv(
            high_l,
            low_l,
            close_l,
            vol_l,
            fast=int(args.kvo_fast),
            slow=int(args.kvo_slow),
            signal=int(args.kvo_signal),
            vf_use_abs_temp=True,
        )
    elif flow_indicator == "pvt":
        if int(flow_signal_period) != int(args.kvo_signal):
            pass
        kvo_l = pvt_tv(close_l, vol_l)
        ksig_l = kvo_l
    elif flow_indicator == "nvi":
        if int(flow_signal_period) != int(args.kvo_signal):
            pass
        kvo_l = nvi_tv(close_l, vol_l, start=float(args.nvi_start))
        ksig_l = kvo_l
    elif flow_indicator == "pvi":
        if int(flow_signal_period) != int(args.kvo_signal):
            pass
        kvo_l = pvi_tv(close_l, vol_l, start=float(args.pvi_start))
        ksig_l = kvo_l
    elif flow_indicator in {"pvt_pvi", "pvi_pvt"}:
        if int(flow_signal_period) != int(args.kvo_signal):
            pass
        pvt_l = pvt_tv(close_l, vol_l)
        pvi_l = pvi_tv(close_l, vol_l, start=float(args.pvi_start))
        kvo_l = [a + b for a, b in zip(pvt_l, pvi_l)]
        ksig_l = kvo_l
    else:
        return {"symbol": str(symbol), "error": f"unsupported_flow_indicator:{flow_indicator}"}

    kvo_s = pd.Series(kvo_l)
    ksig_s = pd.Series(ksig_l)

    confluence_mode = str(args.confluence_mode).strip()
    enable_cci_300 = bool(confluence_mode == "3")

    cci_fast = int(args.cci_fast)
    cci_medium = int(args.cci_medium)

    cfg = KlingerCciExtremesConfig(
        kvo_fast=int(args.kvo_fast),
        kvo_slow=int(args.kvo_slow),
        kvo_signal=int(args.kvo_signal),
        tranche_source=str(args.tranche_source),
        flow_indicator=str(args.flow_indicator),
        flow_signal_period=int(args.flow_signal_period),
        nvi_start=float(args.nvi_start),
        pvi_start=float(args.pvi_start),
        macd_fast=int(args.macd_fast),
        macd_slow=int(args.macd_slow),
        macd_signal=int(args.macd_signal),
        cci_extreme_level=float(args.cci_extreme),
        enable_cci_14=True,
        enable_cci_30=True,
        enable_cci_300=bool(enable_cci_300),
        cci_30_period=int(cci_fast),
        cci_30_col=f"cci_{int(cci_fast)}",
        cci_14_period=int(cci_medium),
        cci_14_col=f"cci_{int(cci_medium)}",
        reference_cci=30,
        dmi_period=int(args.dmi_period),
        dmi_adx_smoothing=int(args.dmi_adx_smoothing),
        volume_col="volume",
    )

    tr = analyze_klinger_cci_tranches(df, cfg=cfg)

    require_cat = str(args.require_dmi_category).strip()
    require_filter = str(getattr(args, "require_dmi_filter", "any") or "any").strip()

    if require_cat in {"impulsion", "respiration"} and require_filter.lower() in {"any", "all", "*", ""}:
        require_filter = str(require_cat)
        require_cat = ""
    if require_cat == "sans_force":
        require_cat = "plat"

    require_cat_enabled = bool(require_cat) and require_cat.lower() not in {"any", "all", "*"}
    require_filter_enabled = bool(require_filter) and require_filter.lower() not in {"any", "all", "*"}
    horizons = [int(x.strip()) for x in str(args.horizons).split(",") if x.strip()]
    horizons = [h for h in horizons if h > 0]
    if not horizons:
        horizons = [30, 60, 90]

    events = [
        t
        for t in tr
        if bool(t.get("cci_confluence_ok"))
        and str(t.get("side") or "") == "SHORT"
        and (not require_cat_enabled or str(t.get("dmi_category") or "") == require_cat)
        and (not require_filter_enabled or str(t.get("dmi_filter") or "") == require_filter)
        and int(t.get("kvo_extreme_i") or -1) >= 0
    ]

    close_s = pd.to_numeric(df["close"], errors="coerce").astype(float)

    if int(cfg.reference_cci) == 14:
        ref_cci_col = str(cfg.cci_14_col)
    elif int(cfg.reference_cci) == 300:
        ref_cci_col = str(cfg.cci_300_col)
    else:
        ref_cci_col = str(cfg.cci_30_col)

    events.sort(key=lambda x: int(x.get("kvo_extreme_i") or 0))

    extreme_seq_len = int(getattr(args, "extreme_seq_len", 2) or 2)
    if extreme_seq_len not in {2, 3, 4}:
        return {"symbol": str(symbol), "error": f"unsupported_extreme_seq_len:{extreme_seq_len}"}


    def _classify_extreme_sequence(seq: list[dict[str, object]]) -> str | None:
        if len(seq) < 2:
            return None
        ok_accum = True
        ok_no_acc = True
        for a, b in zip(seq[:-1], seq[1:]):
            k1 = _safe_float(a.get("kvo_extreme"))
            k2 = _safe_float(b.get("kvo_extreme"))
            s1 = _safe_float(a.get("kvo_signal_at_extreme"))
            s2 = _safe_float(b.get("kvo_signal_at_extreme"))
            if k1 is None or k2 is None or s1 is None or s2 is None:
                return None
            d_k = float(k2) - float(k1)
            d_s = float(s2) - float(s1)
            if bool(flow_has_signal):
                if not (d_k > 0.0 and d_s > 0.0):
                    ok_accum = False
                if not (d_k < 0.0 and d_s < 0.0):
                    ok_no_acc = False
                continue

            # No true signal line => classify using kvo only
            if not (d_k > 0.0):
                ok_accum = False
            if not (d_k < 0.0):
                ok_no_acc = False

        if ok_accum:
            return "ACCUM"
        if ok_no_acc:
            return "NO_ACC"
        return "other"

    paired: list[dict[str, object]] = []
    for k in range(int(extreme_seq_len) - 1, len(events)):
        seq = events[int(k) - int(extreme_seq_len) + 1 : int(k) + 1]
        if len(seq) != int(extreme_seq_len):
            continue

        e_first = seq[0]
        e_last = seq[-1]

        i1 = int(e_first.get("kvo_extreme_i") or -1)
        i2 = int(e_last.get("kvo_extreme_i") or -1)
        if i1 < 0 or i2 < 0 or i2 <= i1:
            continue

        kind = _classify_extreme_sequence(seq)
        if kind is None:
            continue

        kvo1 = _safe_float(e_first.get("kvo_extreme"))
        kvo2 = _safe_float(e_last.get("kvo_extreme"))
        sig1 = _safe_float(e_first.get("kvo_signal_at_extreme"))
        sig2 = _safe_float(e_last.get("kvo_signal_at_extreme"))
        if kvo1 is None or kvo2 is None or sig1 is None or sig2 is None:
            continue

        w_kvo_form = pd.to_numeric(kvo_s.iloc[int(i1) : int(i2) + 1], errors="coerce").astype(float)
        w_sig_form = pd.to_numeric(ksig_s.iloc[int(i1) : int(i2) + 1], errors="coerce").astype(float)
        v_kvo_bottom = float(w_kvo_form.min()) if len(w_kvo_form.dropna()) else float(min(float(kvo1), float(kvo2)))
        v_sig_bottom = float(w_sig_form.min()) if len(w_sig_form.dropna()) else float(min(float(sig1), float(sig2)))

        post_has_2_events = False
        post_ok = True
        post_reason = ""
        next_events = events[int(k) + 1 : int(k) + 3]
        if len(next_events) >= 2:
            post_has_2_events = True
            e3 = next_events[0]
            e4 = next_events[1]

            c2 = _safe_float((e_last.get("cci_meta") or {}).get(f"{ref_cci_col}_extreme_value"))
            c3 = _safe_float((e3.get("cci_meta") or {}).get(f"{ref_cci_col}_extreme_value"))
            c4 = _safe_float((e4.get("cci_meta") or {}).get(f"{ref_cci_col}_extreme_value"))

            k3 = _safe_float(e3.get("kvo_extreme"))
            k4 = _safe_float(e4.get("kvo_extreme"))
            s3 = _safe_float(e3.get("kvo_signal_at_extreme"))
            s4 = _safe_float(e4.get("kvo_signal_at_extreme"))

            cci_weaker = False
            if c2 is not None and c3 is not None and c4 is not None:
                cci_weaker = bool(float(c3) > float(c2) and float(c4) > float(c3))

            klinger_rechute = False
            if k3 is not None and k4 is not None and s3 is not None and s4 is not None:
                if bool(flow_has_signal):
                    klinger_rechute = bool(float(k4) < float(k3) and float(s4) < float(s3))
                else:
                    klinger_rechute = bool(float(k4) < float(k3))

            if bool(cci_weaker and klinger_rechute):
                post_ok = False
                post_reason = "post_rechute_with_weaker_cci"

        validation: dict[str, object] = {}

        paired.append(
            {
                "e2_dt": e_last.get("kvo_extreme_dt"),
                "e2_i": int(i2),
                "kind": str(kind),
                "extreme_seq_len": int(extreme_seq_len),
                "v_kvo_bottom": float(v_kvo_bottom),
                "v_sig_bottom": float(v_sig_bottom),
                "post_has_2_events": bool(post_has_2_events),
                "post_ok": bool(post_ok) if bool(post_has_2_events) else pd.NA,
                "post_reason": str(post_reason),
                **validation,
            }
        )

    dfp = pd.DataFrame(paired)

    thresholds = [2, 5, 10, 15, 20, 25, 30]
    thr_fracs = {t: float(t) / 100.0 for t in thresholds}

    out: dict[str, object] = {
        "symbol": str(symbol),
        "bars": int(len(df)),
        "events_short": int(len(events)),
        "pairs": int(len(dfp)),
        "accum_pairs": int((dfp["kind"] == "ACCUM").sum()) if len(dfp) else 0,
        "require_dmi_category": str(require_cat) if require_cat_enabled else "any",
        "require_dmi_filter": str(require_filter) if require_filter_enabled else "any",
    }

    # Price-based confirmations within the natural window: from E2+1 until retest bottom.
    if len(dfp):
        acc = dfp[dfp["kind"] == "ACCUM"].copy()
        acc["max_return_before_retest"] = pd.NA
        acc["days_to_max_return"] = pd.NA

        for t in thresholds:
            out[f"accum_reached_{t}p"] = float("nan")
            out[f"accum_avg_days_to_{t}p"] = float("nan")

        out["accum_maxret_mean"] = float("nan")
        out["accum_maxret_median"] = float("nan")
        out["accum_maxret_p25"] = float("nan")
        out["accum_maxret_p75"] = float("nan")
        out["accum_avg_days_to_maxret"] = float("nan")

        reached_counts: dict[int, int] = {t: 0 for t in thresholds}
        days_lists: dict[int, list[int]] = {t: [] for t in thresholds}
        maxret_list: list[float] = []
        maxret_days_list: list[int] = []

        for _, r in acc.iterrows():
            i2 = int(r.get("e2_i") or -1)
            if i2 < 0 or i2 >= int(len(df)):
                continue

            entry_close = _safe_float(df.loc[int(i2), "close"])
            if entry_close is None or float(entry_close) <= 0.0:
                continue

            # bottoms were not kept in dfp; recompute from cached r if present, else skip.
            # In this script, we still have vbottom fields in dfp rows if we keep them.
            bottom_kvo = _safe_float(r.get("v_kvo_bottom"))
            bottom_sig = _safe_float(r.get("v_sig_bottom"))
            if bottom_kvo is None or bottom_sig is None:
                continue

            retest_i = _find_retest_bottom_pos(
                kvo_s=kvo_s,
                ksig_s=ksig_s,
                start_i=int(i2) + 1,
                bottom_kvo=float(bottom_kvo),
                bottom_sig=float(bottom_sig),
                require_signal=bool(flow_has_signal),
            )
            end_i = int(retest_i - 1) if retest_i is not None else int(len(df) - 1)
            if end_i <= int(i2):
                continue

            # Compute max return before retest bottom
            best_ret = None
            best_day = None
            for j in range(int(i2) + 1, int(end_i) + 1):
                c = _safe_float(close_s.iloc[int(j)])
                if c is None:
                    continue
                ret = (float(c) / float(entry_close)) - 1.0
                if best_ret is None or float(ret) > float(best_ret):
                    best_ret = float(ret)
                    best_day = int(j - int(i2))

            if best_ret is not None and best_day is not None:
                maxret_list.append(float(best_ret))
                maxret_days_list.append(int(best_day))

            for t in thresholds:
                thr = float(thr_fracs[t])
                found_day: int | None = None
                for j in range(int(i2) + 1, int(end_i) + 1):
                    c = _safe_float(close_s.iloc[int(j)])
                    if c is None:
                        continue
                    ret = (float(c) / float(entry_close)) - 1.0

                    if float(ret) >= float(thr):
                        found_day = int(j - int(i2))
                        break

                if found_day is not None:
                    reached_counts[t] += 1
                    days_lists[t].append(int(found_day))

        n_acc = int(len(acc))
        if n_acc > 0:
            for t in thresholds:
                out[f"accum_reached_{t}p"] = float(reached_counts[t] / n_acc)
                if len(days_lists[t]):
                    out[f"accum_avg_days_to_{t}p"] = float(sum(days_lists[t]) / len(days_lists[t]))
                else:
                    out[f"accum_avg_days_to_{t}p"] = float("nan")

        if len(maxret_list):
            s = pd.Series(maxret_list, dtype=float)
            out["accum_maxret_mean"] = float(s.mean())
            out["accum_maxret_median"] = float(s.median())
            out["accum_maxret_p25"] = float(s.quantile(0.25))
            out["accum_maxret_p75"] = float(s.quantile(0.75))
        if len(maxret_days_list):
            out["accum_avg_days_to_maxret"] = float(sum(maxret_days_list) / len(maxret_days_list))

        # Campaign-level stats (this answers: how many ACCUM give NO profit before retest bottom?)
        campaigns = _build_accum_campaigns(
            df=df,
            dfp_acc=acc,
            close_s=close_s,
            kvo_s=kvo_s,
            ksig_s=ksig_s,
            thresholds=thresholds,
            max_bottom_breaks=int(args.max_bottom_breaks),
            lookahead=int(getattr(args, "lookahead", 0) or 0),
            require_signal=bool(flow_has_signal),
        )

        if len(campaigns):
            tmax_rets: list[float] = []
            tmax_pcts: list[float] = []
            for ix in range(int(len(campaigns))):
                r = campaigns.iloc[int(ix)].to_dict()
                st_i = int(r.get("campaign_start_i") or -1)
                ev_end_i = int(r.get("campaign_eval_end_i") or -1)
                entry = _safe_float(r.get("campaign_entry_avg"))
                if st_i < 0 or ev_end_i < 0 or entry is None:
                    continue
                m = _campaign_tranche_max_return(tranches=tr, start_i=st_i, end_i=ev_end_i, entry_price=float(entry))
                for k, v in m.items():
                    campaigns.loc[int(ix), str(k)] = v
                vv = _safe_float(m.get("tranche_max_return"))
                if vv is not None:
                    tmax_rets.append(float(vv))
                    tmax_pcts.append(float(vv) * 100.0)

            if len(tmax_pcts):
                out["tranche_max_captured_sum_p"] = float(sum(tmax_pcts))
                out["tranche_max_captured_mean_p"] = float(sum(tmax_pcts) / len(tmax_pcts))
                out["tranche_max_captured_median_p"] = float(pd.Series(tmax_pcts, dtype=float).median())
            else:
                out["tranche_max_captured_sum_p"] = float("nan")
                out["tranche_max_captured_mean_p"] = float("nan")
                out["tranche_max_captured_median_p"] = float("nan")
        out["campaigns_total"] = int(len(campaigns))
        if len(campaigns):
            s_conf = campaigns["campaign_confirmed"].astype("boolean")
            out["campaigns_confirmed"] = int(s_conf.sum(skipna=True))
            out["campaigns_no_confirm_retest"] = int((~s_conf.fillna(False)).sum())
            out["campaign_confirm_rate"] = float(s_conf.mean(skipna=True))

            mae_s = pd.to_numeric(campaigns.get("campaign_mae_p"), errors="coerce").astype(float)
            mae_s = mae_s.dropna()
            out["campaign_mae_mean_p"] = float(mae_s.mean()) if len(mae_s) else float("nan")
            out["campaign_mae_median_p"] = float(mae_s.median()) if len(mae_s) else float("nan")

            mdd_s = pd.to_numeric(campaigns.get("campaign_mdd_p"), errors="coerce").astype(float)
            mdd_s = mdd_s.dropna()
            out["campaign_mdd_mean_p"] = float(mdd_s.mean()) if len(mdd_s) else float("nan")
            out["campaign_mdd_median_p"] = float(mdd_s.median()) if len(mdd_s) else float("nan")
            subc = campaigns[campaigns["campaign_confirmed"] == True]
            if len(subc) and "campaign_days_to_confirm" in subc.columns:
                d = pd.to_numeric(subc["campaign_days_to_confirm"], errors="coerce").astype(float)
                out["campaign_avg_days_to_confirm"] = float(d.mean()) if len(d.dropna()) else float("nan")
            else:
                out["campaign_avg_days_to_confirm"] = float("nan")
            out["campaigns_no_confirm_list"] = (
                campaigns[campaigns["campaign_confirmed"] == False]["campaign_start_dt"].astype(str).tolist()
            )
            # Compact detail for debugging/inspection
            no_c = campaigns[campaigns["campaign_confirmed"] == False]
            det: list[str] = []
            for _, rr in no_c.iterrows():
                thr = str(rr.get("campaign_thr_first_days") or "")
                det.append(
                    f"{rr.get('campaign_start_dt')}|last_sig={rr.get('campaign_last_sig_dt')}|signals={int(rr.get('campaign_signals') or 0)}|breaks={int(rr.get('campaign_breaks_used') or 0)}|break_dts={rr.get('campaign_break_dts')}|entry_avg={rr.get('campaign_entry_avg')}|maxlvl={rr.get('campaign_max_level_reached_p')}|maxret={rr.get('campaign_max_return')}|retest={rr.get('campaign_retest_dt')}|thr_first_days={thr}"
                )
            out["campaigns_no_confirm_detail"] = det

            ok_c = campaigns[campaigns["campaign_confirmed"] == True]
            det2: list[str] = []
            for _, rr in ok_c.iterrows():
                thr = str(rr.get("campaign_thr_first_days") or "")
                det2.append(
                    f"{rr.get('campaign_start_dt')}|last_sig={rr.get('campaign_last_sig_dt')}|signals={int(rr.get('campaign_signals') or 0)}|breaks={int(rr.get('campaign_breaks_used') or 0)}|entry_avg={rr.get('campaign_entry_avg')}|confirm={rr.get('campaign_confirm_dt')}|level={rr.get('campaign_confirm_level_p')}|days={rr.get('campaign_days_to_confirm')}|thr_first_days={thr}"
                )
            out["campaigns_confirm_detail"] = det2
        else:
            out["campaigns_confirmed"] = 0
            out["campaigns_no_confirm_retest"] = 0
            out["campaign_confirm_rate"] = float("nan")
            out["campaign_avg_days_to_confirm"] = float("nan")
            out["campaigns_no_confirm_list"] = []
            out["campaigns_no_confirm_detail"] = []
            out["campaigns_confirm_detail"] = []

        export_dir = str(getattr(args, "export_dir", "") or "").strip()
        if export_dir:
            p = Path(export_dir).expanduser().resolve()
            p.mkdir(parents=True, exist_ok=True)
            base = (
                f"zone_{symbol}_{args.interval}_{args.year_start}_{args.year_end}"
                f"_tr_{args.tranche_source}_flow_{args.flow_indicator}_cci{args.confluence_mode}"
                f"_dmi{args.require_dmi_category}_{args.require_dmi_filter}_breaks{args.max_bottom_breaks}"
            )
            try:
                campaigns_out = campaigns.copy()
                campaigns_out["symbol"] = str(symbol)
                campaigns_out["interval"] = str(args.interval)
                campaigns_out["year_start"] = str(args.year_start)
                campaigns_out["year_end"] = str(args.year_end)
                campaigns_out["tranche_source"] = str(args.tranche_source)
                campaigns_out["flow_indicator"] = str(args.flow_indicator)
                campaigns_out["confluence_mode"] = str(args.confluence_mode)
                campaigns_out["require_dmi_category"] = str(args.require_dmi_category)
                campaigns_out["require_dmi_filter"] = str(getattr(args, "require_dmi_filter", "any") or "any")
                campaigns_out["max_bottom_breaks"] = int(args.max_bottom_breaks)

                # Optional: if the campaigns df already contains per-event dmi_force_brute columns (from earlier steps), keep them.
                # Otherwise nothing to do.
                campaigns_out.to_csv(p / f"{base}_campaigns.csv", index=False)
            except Exception:
                pass

    if len(dfp) and "post_ok" in dfp.columns:
        sub = dfp[(dfp["kind"] == "ACCUM") & (dfp["post_has_2_events"] == True)]
        if len(sub):
            s = sub["post_ok"].astype("boolean")
            out["accum_post_ok_2ev"] = float(s.mean(skipna=True))
        else:
            out["accum_post_ok_2ev"] = float("nan")

    # Start points for accumulating = E2 of ACCUM patterns
    if len(dfp):
        acc = dfp[dfp["kind"] == "ACCUM"].copy()
        out["accum_start_points"] = acc["e2_dt"].tolist()
    else:
        out["accum_start_points"] = []

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="LINKUSDT")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--category", default="linear")
    ap.add_argument("--base-url", default="https://api.bybit.com")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--limit", type=int, default=1000)

    ap.add_argument("--cache-dir", default=str(PROJECT_ROOT / "data" / "cache" / "klines"))
    ap.add_argument("--force-refresh", action="store_true")

    ap.add_argument("--max-bottom-breaks", type=int, default=1)

    ap.add_argument("--print-campaigns", choices=["none", "failed", "all"], default="failed")

    ap.add_argument("--print-start-points", choices=["no", "yes"], default="no")

    ap.add_argument("--year-start", default="2024-01-01")
    ap.add_argument("--year-end", default="2025-12-31")

    ap.add_argument("--kvo-fast", type=int, default=34)
    ap.add_argument("--kvo-slow", type=int, default=55)
    ap.add_argument("--kvo-signal", type=int, default=13)

    ap.add_argument("--flow-indicator", choices=["klinger", "pvt", "nvi", "pvi", "pvt_pvi"], default="klinger")
    ap.add_argument("--flow-signal-period", type=int, default=13)
    ap.add_argument("--nvi-start", type=float, default=1000.0)
    ap.add_argument("--pvi-start", type=float, default=1000.0)

    ap.add_argument("--volume-source", choices=["volume", "turnover"], default="volume")

    ap.add_argument("--tranche-source", choices=["kvo_diff", "macd_hist"], default="kvo_diff")
    ap.add_argument("--macd-fast", type=int, default=12)
    ap.add_argument("--macd-slow", type=int, default=26)
    ap.add_argument("--macd-signal", type=int, default=9)

    ap.add_argument("--cci-fast", type=int, default=30)
    ap.add_argument("--cci-medium", type=int, default=90)
    ap.add_argument("--cci-extreme", type=float, default=100.0)
    ap.add_argument("--confluence-mode", choices=["2", "3"], default="2")
    ap.add_argument("--dmi-period", type=int, default=14)
    ap.add_argument("--dmi-adx-smoothing", type=int, default=14)

    ap.add_argument("--extreme-seq-len", type=int, default=2)

    ap.add_argument("--lookahead", type=int, default=30)
    ap.add_argument("--reversal-up-pct", type=float, default=0.15)
    ap.add_argument("--continuation-down-pct", type=float, default=0.10)

    ap.add_argument("--require-dmi-category", default="any")
    ap.add_argument("--require-dmi-filter", default="any")
    ap.add_argument("--horizons", default="30,60,90")

    ap.add_argument("--max-print", type=int, default=25)
    ap.add_argument("--export-dir", default="")
    args = ap.parse_args()

    symbols_s = str(args.symbols).strip()
    if symbols_s:
        symbols = [s.strip() for s in symbols_s.split(",") if s.strip()]
    else:
        symbols = [str(args.symbol)]

    results: list[dict[str, object]] = []
    for sym in symbols:
        results.append(_run_one_symbol(args, symbol=sym))

    horizons = [int(x.strip()) for x in str(args.horizons).split(",") if x.strip()]
    horizons = [h for h in horizons if h > 0]
    if not horizons:
        horizons = [30, 60, 90]

    print(
        f"PERIOD {args.year_start}..{args.year_end} TF={args.interval} CCI={args.cci_fast}/{args.cci_medium} "
        f"confluence_mode={args.confluence_mode} require_cat={args.require_dmi_category} require_filter={args.require_dmi_filter}"
    )
    for r in results:
        if r.get("error"):
            print(f"{r.get('symbol')}: ERROR {r.get('error')}")
            continue
        parts = [
            f"{r.get('symbol')}",
            f"bars={r.get('bars')}",
            f"events_short={r.get('events_short')}",
            f"accum_pairs={r.get('accum_pairs')}",
            f"campaigns={r.get('campaigns_total')}",
            f"campaigns_confirmed={r.get('campaigns_confirmed')}",
            f"campaigns_no_confirm_retest={r.get('campaigns_no_confirm_retest')}",
        ]
        if "accum_maxret_median" in r:
            v = r.get("accum_maxret_median")
            parts.append(f"accum_maxret_median={v:.3f}" if isinstance(v, float) and math.isfinite(v) else "accum_maxret_median=nan")
        if "accum_maxret_p25" in r and "accum_maxret_p75" in r:
            v1 = r.get("accum_maxret_p25")
            v2 = r.get("accum_maxret_p75")
            p25s = f"{v1:.3f}" if isinstance(v1, float) and math.isfinite(v1) else "nan"
            p75s = f"{v2:.3f}" if isinstance(v2, float) and math.isfinite(v2) else "nan"
            parts.append(f"accum_maxret_p25_p75={p25s}/{p75s}")
        if "accum_avg_days_to_maxret" in r:
            v = r.get("accum_avg_days_to_maxret")
            parts.append(f"accum_days_to_maxret={v:.1f}" if isinstance(v, float) and math.isfinite(v) else "accum_days_to_maxret=nan")
        for t in [2, 5, 10, 15, 20, 25, 30]:
            k = f"accum_reached_{int(t)}p"
            k2 = f"accum_avg_days_to_{int(t)}p"
            if k in r:
                v = r.get(k)
                parts.append(f"{k}={v:.3f}" if isinstance(v, float) and math.isfinite(v) else f"{k}=nan")
            if k2 in r:
                v2 = r.get(k2)
                parts.append(f"{k2}={v2:.1f}" if isinstance(v2, float) and math.isfinite(v2) else f"{k2}=nan")
        if "accum_post_ok_2ev" in r:
            v = r.get("accum_post_ok_2ev")
            parts.append(f"accum_post_ok_2ev={v:.3f}" if isinstance(v, float) and math.isfinite(v) else "accum_post_ok_2ev=nan")
        print(" | ".join(parts))

        if str(args.print_start_points) == "yes":
            sp = r.get("accum_start_points") or []
            if sp:
                print(f"  ACCUM_START_POINTS: {', '.join(str(x) for x in sp[:12])}{' ...' if len(sp) > 12 else ''}")

        ncl = r.get("campaigns_no_confirm_list") or []
        if ncl:
            print(f"  CAMPAIGNS_NO_CONFIRM_RETEST: {', '.join(str(x) for x in ncl[:12])}{' ...' if len(ncl) > 12 else ''}")

        mode = str(args.print_campaigns)
        if mode not in {"none", "failed", "all"}:
            mode = "failed"

        det_failed = r.get("campaigns_no_confirm_detail") or []
        det_conf = r.get("campaigns_confirm_detail") or []

        if mode in {"failed", "all"} and det_failed:
            print("  CAMPAIGNS_FAILED_DETAIL:")
            for line in det_failed[:25]:
                print(f"    {line}")

        if mode == "all" and det_conf:
            print("  CAMPAIGNS_CONFIRMED_DETAIL:")
            for line in det_conf[:25]:
                print(f"    {line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
