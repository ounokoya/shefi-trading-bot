from __future__ import annotations

import argparse
import sys
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="LINKUSDT")
    ap.add_argument("--category", default="linear")
    ap.add_argument("--base-url", default="https://api.bybit.com")
    ap.add_argument("--interval", default="15m")
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

    ap.add_argument("--enable-cci-14", default=True, action=argparse.BooleanOptionalAction)
    ap.add_argument("--enable-cci-30", default=True, action=argparse.BooleanOptionalAction)
    ap.add_argument("--enable-cci-300", default=False, action=argparse.BooleanOptionalAction)

    ap.add_argument("--ref-cci", type=int, choices=[14, 30, 300], default=30)
    ap.add_argument("--tail", type=int, default=25)
    ap.add_argument("--only-cci-confluence", default=False, action=argparse.BooleanOptionalAction)
    ap.add_argument("--only-side", choices=["", "LONG", "SHORT"], default="")
    ap.add_argument(
        "--only-dmi-category",
        choices=["", "plat", "tendenciel", "impulsion", "respiration", "sans_force"],
        default="",
    )
    ap.add_argument("--only-dmi-filter", choices=["", "impulsion", "respiration"], default="")
    ap.add_argument("--only-dmi-force", choices=["", "true", "false"], default="")
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

    cfg = KlingerCciExtremesConfig(
        kvo_fast=int(args.kvo_fast),
        kvo_slow=int(args.kvo_slow),
        kvo_signal=int(args.kvo_signal),
        dmi_period=int(args.dmi_period),
        dmi_adx_smoothing=int(args.dmi_adx_smoothing),
        adx_force_threshold=float(args.adx_force_threshold),
        adx_force_confirm_bars=int(args.adx_force_confirm_bars),
        cci_extreme_level=float(args.cci_extreme),
        enable_cci_14=bool(args.enable_cci_14),
        enable_cci_30=bool(args.enable_cci_30),
        enable_cci_300=bool(args.enable_cci_300),
        reference_cci=int(args.ref_cci),
        volume_col="volume",
    )

    tranches = analyze_klinger_cci_tranches(df, cfg=cfg)

    if int(args.tail) > 0:
        tranches = tranches[-int(args.tail) :]

    if not tranches:
        print("<no tranches>")
        return 0

    only_cat = str(args.only_dmi_category).strip()
    only_filter = str(args.only_dmi_filter).strip()
    if only_cat in {"impulsion", "respiration"} and not only_filter:
        only_filter = str(only_cat)
        only_cat = ""
    if only_cat == "sans_force":
        only_cat = "plat"

    for t in tranches:
        meta = t.get("cci_meta") or {}
        ref_episodes = meta.get("ref_episodes")
        deeper = meta.get("ref_deeper_progression")
        confluence_ok = t.get("cci_confluence_ok")

        if bool(args.only_cci_confluence) and (not bool(confluence_ok)):
            continue

        side = str(t.get("side") or "")
        if str(args.only_side).strip() and side != str(args.only_side).strip():
            continue

        dmi_side = t.get("dmi_side")
        dmi_aligned = t.get("dmi_aligned")
        dmi_category = t.get("dmi_category")
        dmi_filter = t.get("dmi_filter")
        dmi_force_brute = t.get("dmi_force_brute")
        dmi_force_confirmed = t.get("dmi_force_confirmed")
        dmi_adx = t.get("dmi_adx_at_extreme")
        dmi_dx = t.get("dmi_dx_at_extreme")
        dmi_pdi = t.get("dmi_plus_di_at_extreme")
        dmi_mdi = t.get("dmi_minus_di_at_extreme")

        if only_cat and str(dmi_category) != str(only_cat):
            continue

        if only_filter and str(dmi_filter) != str(only_filter):
            continue

        if str(args.only_dmi_force).strip():
            want = str(args.only_dmi_force).strip().lower()
            if want == "true" and not bool(dmi_force_confirmed):
                continue
            if want == "false" and bool(dmi_force_confirmed):
                continue

        kvo_ext = t.get("kvo_extreme")
        ksig = t.get("kvo_signal_at_extreme")
        close = t.get("close_at_extreme")
        dt = t.get("kvo_extreme_dt")

        print(
            " | ".join(
                [
                    f"{t.get('start_dt')} -> {t.get('end_dt')}",
                    f"side={t.get('side')}",
                    f"dmi_side={dmi_side}",
                    f"dmi_aligned={dmi_aligned}",
                    f"dmi_category={dmi_category}",
                    f"dmi_filter={dmi_filter}",
                    f"dmi_force_brute={dmi_force_brute}",
                    f"dmi_force={dmi_force_confirmed}",
                    f"adx@ext={dmi_adx}",
                    f"dx@ext={dmi_dx}",
                    f"+di@ext={dmi_pdi}",
                    f"-di@ext={dmi_mdi}",
                    f"kvo_ext={kvo_ext}",
                    f"sig_at_ext={ksig}",
                    f"close_at_ext={close}",
                    f"dt_ext={dt}",
                    f"cci_confluence_ok={confluence_ok}",
                    f"ref_episodes={ref_episodes}",
                    f"deeper={deeper}",
                ]
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
