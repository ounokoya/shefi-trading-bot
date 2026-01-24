from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def _safe_float(v: object) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return float(x)


def _mfe_mae(
    *,
    df: pd.DataFrame,
    start_i: int,
    side: str,
    window: int,
    entry_price: float,
    high_col: str = "high",
    low_col: str = "low",
) -> tuple[float | None, float | None]:
    if int(window) <= 0:
        return None, None

    end_i = min(int(len(df) - 1), int(start_i) + int(window))
    if end_i <= start_i:
        return None, None

    w = df.iloc[int(start_i) + 1 : int(end_i) + 1]
    if not len(w):
        return None, None

    highs = pd.to_numeric(w[str(high_col)], errors="coerce").astype(float)
    lows = pd.to_numeric(w[str(low_col)], errors="coerce").astype(float)

    if str(side) == "LONG":
        mfe = (float(highs.max()) / float(entry_price)) - 1.0
        mae = (float(lows.min()) / float(entry_price)) - 1.0
    else:
        mfe = 1.0 - (float(lows.min()) / float(entry_price))
        mae = 1.0 - (float(highs.max()) / float(entry_price))

    return float(mfe), float(mae)


@dataclass(frozen=True)
class BtResult:
    n: int
    winrate: float | None
    avg_r: float | None
    sum_r: float | None


def _mini_backtest(
    *,
    df: pd.DataFrame,
    events: list[dict[str, object]],
    tp_pct: float,
    sl_pct: float,
    max_hold_bars: int,
) -> BtResult:
    if not events:
        return BtResult(n=0, winrate=None, avg_r=None, sum_r=None)

    returns: list[float] = []

    for ev in events:
        side = str(ev.get("side") or "")
        i = int(ev.get("kvo_extreme_i") or 0)
        entry = _safe_float(ev.get("close_at_extreme"))
        if not side or entry is None or not math.isfinite(float(entry)):
            continue

        tp = None
        sl = None
        if str(side) == "LONG":
            tp = float(entry) * (1.0 + float(tp_pct))
            sl = float(entry) * (1.0 - float(sl_pct))
        else:
            tp = float(entry) * (1.0 - float(tp_pct))
            sl = float(entry) * (1.0 + float(sl_pct))

        end_i = min(int(len(df) - 1), int(i) + int(max_hold_bars))
        if end_i <= i:
            continue

        exit_price = None
        for k in range(int(i) + 1, int(end_i) + 1):
            hi = _safe_float(df.iloc[k]["high"])
            lo = _safe_float(df.iloc[k]["low"])
            if hi is None or lo is None:
                continue

            if str(side) == "LONG":
                if float(lo) <= float(sl):
                    exit_price = float(sl)
                    break
                if float(hi) >= float(tp):
                    exit_price = float(tp)
                    break
            else:
                if float(hi) >= float(sl):
                    exit_price = float(sl)
                    break
                if float(lo) <= float(tp):
                    exit_price = float(tp)
                    break

        if exit_price is None:
            exit_price = _safe_float(df.iloc[end_i]["close"])

        if exit_price is None:
            continue

        if str(side) == "LONG":
            r = (float(exit_price) / float(entry)) - 1.0
        else:
            r = (float(entry) / float(exit_price)) - 1.0
        returns.append(float(r))

    if not returns:
        return BtResult(n=0, winrate=None, avg_r=None, sum_r=None)

    wins = sum(1 for r in returns if float(r) > 0.0)
    return BtResult(
        n=int(len(returns)),
        winrate=float(wins) / float(len(returns)),
        avg_r=float(sum(returns)) / float(len(returns)),
        sum_r=float(sum(returns)),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="LINKUSDT")
    ap.add_argument("--category", default="linear")
    ap.add_argument("--base-url", default="https://api.bybit.com")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--volume-col", choices=["volume", "turnover"], default="volume")

    ap.add_argument("--kvo-fast", type=int, default=34)
    ap.add_argument("--kvo-slow", type=int, default=55)
    ap.add_argument("--kvo-signal", type=int, default=13)

    ap.add_argument("--dmi-period", type=int, default=14)
    ap.add_argument("--dmi-adx-smoothing", type=int, default=14)
    ap.add_argument("--adx-force-threshold", type=float, default=20.0)
    ap.add_argument("--adx-force-confirm-bars", type=int, default=3)

    ap.add_argument("--cci-extreme", type=float, default=100.0)

    ap.add_argument("--confluence-mode", choices=["2", "3"], default="2")
    ap.add_argument("--ref-cci", type=int, choices=[14, 30, 300], default=30)

    ap.add_argument("--only-side", choices=["", "LONG", "SHORT"], default="")
    ap.add_argument(
        "--only-dmi-category",
        choices=["", "plat", "tendenciel", "impulsion", "respiration", "sans_force"],
        default="",
    )
    ap.add_argument("--only-dmi-filter", choices=["", "impulsion", "respiration"], default="")
    ap.add_argument("--only-dmi-force", choices=["", "true", "false"], default="")

    ap.add_argument("--horizons", default="5,10,20")
    ap.add_argument("--mfe-window", type=int, default=20)

    ap.add_argument("--bt-tp-pct", type=float, default=0.05)
    ap.add_argument("--bt-sl-pct", type=float, default=0.03)
    ap.add_argument("--bt-max-hold-bars", type=int, default=20)

    ap.add_argument("--out-csv", default="")
    args = ap.parse_args()

    df = _fetch_bybit_klines_last_n(
        symbol=str(args.symbol),
        interval=_interval_to_bybit(str(args.interval)),
        limit=int(args.limit),
        category=str(args.category),
        base_url=str(args.base_url),
        timeout_s=30.0,
    )
    if not len(df):
        raise RuntimeError("no klines fetched")

    vol_col = str(args.volume_col)
    if vol_col not in df.columns:
        raise RuntimeError(f"volume column not found: {vol_col}")
    df["volume"] = df[vol_col].astype(float)

    confluence_mode = str(args.confluence_mode)
    enable_cci_300 = bool(confluence_mode == "3")

    cfg = KlingerCciExtremesConfig(
        kvo_fast=int(args.kvo_fast),
        kvo_slow=int(args.kvo_slow),
        kvo_signal=int(args.kvo_signal),
        dmi_period=int(args.dmi_period),
        dmi_adx_smoothing=int(args.dmi_adx_smoothing),
        adx_force_threshold=float(args.adx_force_threshold),
        adx_force_confirm_bars=int(args.adx_force_confirm_bars),
        cci_extreme_level=float(args.cci_extreme),
        enable_cci_14=True,
        enable_cci_30=True,
        enable_cci_300=bool(enable_cci_300),
        reference_cci=int(args.ref_cci),
        volume_col="volume",
    )

    tranches = analyze_klinger_cci_tranches(df, cfg=cfg)

    # Apply filters (event-level)
    filtered: list[dict[str, object]] = []
    only_cat = str(args.only_dmi_category).strip()
    only_filter = str(args.only_dmi_filter).strip()
    if only_cat in {"impulsion", "respiration"} and not only_filter:
        only_filter = str(only_cat)
        only_cat = ""
    if only_cat == "sans_force":
        only_cat = "plat"
    for t in tranches:
        side = str(t.get("side") or "")
        if str(args.only_side).strip() and side != str(args.only_side).strip():
            continue

        dmi_cat = str(t.get("dmi_category") or "")
        if only_cat and dmi_cat != str(only_cat):
            continue

        dmi_filter = str(t.get("dmi_filter") or "")
        if only_filter and dmi_filter != str(only_filter):
            continue

        dmi_force = bool(t.get("dmi_force_confirmed"))
        if str(args.only_dmi_force).strip():
            want = str(args.only_dmi_force).strip().lower()
            if want == "true" and not bool(dmi_force):
                continue
            if want == "false" and bool(dmi_force):
                continue

        # confluence in chosen mode (2/2 or 3/3 based on enabled CCIs)
        if not bool(t.get("cci_confluence_ok")):
            continue

        filtered.append(t)

    # Compute outcome metrics
    horizons = []
    for x in str(args.horizons).split(","):
        x = str(x).strip()
        if not x:
            continue
        try:
            horizons.append(int(x))
        except Exception:
            continue
    horizons = sorted(set(h for h in horizons if h > 0))

    rows: list[dict[str, object]] = []
    for t in filtered:
        i = int(t.get("kvo_extreme_i") or 0)
        side = str(t.get("side") or "")
        entry = _safe_float(t.get("close_at_extreme"))
        if entry is None:
            continue

        row: dict[str, object] = {
            "symbol": str(args.symbol),
            "interval": str(args.interval),
            "confluence_mode": str(confluence_mode),
            "start_dt": t.get("start_dt"),
            "end_dt": t.get("end_dt"),
            "dt_ext": t.get("kvo_extreme_dt"),
            "pos_ext": int(i),
            "side": str(side),
            "close_ext": float(entry),
            "kvo_ext": t.get("kvo_extreme"),
            "kvo_sig_ext": t.get("kvo_signal_at_extreme"),
            "kvo_diff_ext": (
                None
                if _safe_float(t.get("kvo_extreme")) is None or _safe_float(t.get("kvo_signal_at_extreme")) is None
                else float(_safe_float(t.get("kvo_extreme")) or 0.0) - float(_safe_float(t.get("kvo_signal_at_extreme")) or 0.0)
            ),
            "dmi_category": t.get("dmi_category"),
            "dmi_filter": t.get("dmi_filter"),
            "dmi_force_brute": t.get("dmi_force_brute"),
            "dmi_force_confirmed": bool(t.get("dmi_force_confirmed")),
            "adx_ext": t.get("dmi_adx_at_extreme"),
            "dx_ext": t.get("dmi_dx_at_extreme"),
            "pdi_ext": t.get("dmi_plus_di_at_extreme"),
            "mdi_ext": t.get("dmi_minus_di_at_extreme"),
            "dmi_aligned": bool(t.get("dmi_aligned")),
            "ref_episodes": (t.get("cci_meta") or {}).get("ref_episodes"),
            "deeper": (t.get("cci_meta") or {}).get("ref_deeper_progression"),
        }

        for h in horizons:
            j = min(int(len(df) - 1), int(i) + int(h))
            if j <= i:
                row[f"ret_{h}"] = None
                continue
            px = _safe_float(df.iloc[j]["close"])
            if px is None:
                row[f"ret_{h}"] = None
                continue
            if str(side) == "LONG":
                row[f"ret_{h}"] = (float(px) / float(entry)) - 1.0
            else:
                row[f"ret_{h}"] = (float(entry) / float(px)) - 1.0

        mfe, mae = _mfe_mae(df=df, start_i=int(i), side=str(side), window=int(args.mfe_window), entry_price=float(entry))
        row["mfe"] = mfe
        row["mae"] = mae
        rows.append(row)

    out_csv = str(args.out_csv).strip()
    if not out_csv:
        side_part = (str(args.only_side).strip() or "ANY")
        cat_part = (only_cat or "ANY")
        filter_part = (only_filter or "ANY")
        force_part = (str(args.only_dmi_force).strip() or "ANY")
        out_csv = f"events_{args.symbol}_{args.interval}_conf{confluence_mode}_side{side_part}_cat{cat_part}_filter{filter_part}_force{force_part}.csv"

    if rows:
        cols = sorted({k for r in rows for k in r.keys()})
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    print(f"events_total={len(tranches)} events_filtered={len(rows)} out_csv={out_csv}")

    # Quick aggregated stats
    if rows:
        df_ev = pd.DataFrame(rows)
        if horizons:
            for h in horizons:
                c = f"ret_{h}"
                if c in df_ev.columns:
                    s = pd.to_numeric(df_ev[c], errors="coerce").astype(float)
                    print(f"{c}: mean={s.mean():.6f} median={s.median():.6f} winrate={(s > 0).mean():.4f} n={int(s.count())}")

        s_mfe = pd.to_numeric(df_ev["mfe"], errors="coerce").astype(float)
        s_mae = pd.to_numeric(df_ev["mae"], errors="coerce").astype(float)
        print(f"mfe: mean={s_mfe.mean():.6f} median={s_mfe.median():.6f}")
        print(f"mae: mean={s_mae.mean():.6f} median={s_mae.median():.6f}")

    # Mini-backtest on filtered events
    bt = _mini_backtest(
        df=df,
        events=filtered,
        tp_pct=float(args.bt_tp_pct),
        sl_pct=float(args.bt_sl_pct),
        max_hold_bars=int(args.bt_max_hold_bars),
    )
    print(f"mini_bt: n={bt.n} winrate={bt.winrate} avg_r={bt.avg_r} sum_r={bt.sum_r} tp={args.bt_tp_pct} sl={args.bt_sl_pct} hold={args.bt_max_hold_bars}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
