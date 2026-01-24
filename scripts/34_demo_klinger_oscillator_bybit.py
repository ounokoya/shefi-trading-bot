from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.indicators.volume.klinger_oscillator_tv import klinger_oscillator_tv  # noqa: E402


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
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--volume-source", choices=["volume", "turnover"], default="volume")
    ap.add_argument("--fast", type=int, default=34)
    ap.add_argument("--slow", type=int, default=55)
    ap.add_argument("--signal", type=int, default=13)
    ap.add_argument(
        "--vf-use-abs-temp",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    ap.add_argument("--tail", type=int, default=50)
    ap.add_argument(
        "--print-bars",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    ap.add_argument("--crosses-tail", type=int, default=20)
    ap.add_argument("--segments-tail", type=int, default=10)
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

    vol_col = "volume" if str(args.volume_source) == "volume" else "turnover"
    if vol_col not in df.columns:
        raise RuntimeError(f"volume column not found: {vol_col}")

    kvo, ksig = klinger_oscillator_tv(
        df["high"].astype(float).to_list(),
        df["low"].astype(float).to_list(),
        df["close"].astype(float).to_list(),
        df[vol_col].astype(float).to_list(),
        fast=int(args.fast),
        slow=int(args.slow),
        signal=int(args.signal),
        vf_use_abs_temp=bool(args.vf_use_abs_temp),
    )

    df["klinger"] = kvo
    df["klinger_signal"] = ksig
    df["klinger_diff"] = df["klinger"].astype(float) - df["klinger_signal"].astype(float)

    tail = int(args.tail)
    if tail > 0:
        df2 = df.tail(tail).reset_index(drop=True)
    else:
        df2 = df

    if bool(args.print_bars):
        for _, r in df2.iterrows():
            dt = str(r.get("dt") or "")
            close = float(r.get("close") or float("nan"))
            k = float(r.get("klinger") or float("nan"))
            s = float(r.get("klinger_signal") or float("nan"))
            print(f"{dt} close={close:.6f} klinger={k:.6f} signal={s:.6f}")

    df_cross = df[["ts", "dt", "close", "klinger", "klinger_signal", "klinger_diff"]].copy()
    df_cross["diff_prev"] = df_cross["klinger_diff"].shift(1)
    df_cross["cross_kind"] = ""
    up_mask = (df_cross["diff_prev"] <= 0) & (df_cross["klinger_diff"] > 0)
    dn_mask = (df_cross["diff_prev"] >= 0) & (df_cross["klinger_diff"] < 0)
    df_cross.loc[up_mask, "cross_kind"] = "UP"
    df_cross.loc[dn_mask, "cross_kind"] = "DOWN"
    crosses = df_cross[df_cross["cross_kind"] != ""].reset_index(drop=False).rename(columns={"index": "i"})

    crosses_tail = int(args.crosses_tail)
    if crosses_tail > 0:
        crosses_to_print = crosses.tail(crosses_tail)
    else:
        crosses_to_print = crosses

    print("\nCROSSES (KVO vs Signal)")
    if not len(crosses_to_print):
        print("<none>")
    else:
        for _, r in crosses_to_print.iterrows():
            dt = str(r.get("dt") or "")
            close = float(r.get("close") or float("nan"))
            k = float(r.get("klinger") or float("nan"))
            s = float(r.get("klinger_signal") or float("nan"))
            d = float(r.get("klinger_diff") or float("nan"))
            ck = str(r.get("cross_kind") or "")
            i = int(r.get("i") or 0)
            print(f"{dt} i={i} close={close:.6f} cross={ck} kvo={k:.6f} sig={s:.6f} diff={d:.6f}")

    segments: list[dict[str, object]] = []
    if len(crosses) >= 2:
        cross_pos = crosses["i"].astype(int).to_list()
        cross_kind = crosses["cross_kind"].astype(str).to_list()
        for j in range(len(cross_pos) - 1):
            start_i = int(cross_pos[j])
            end_i = int(cross_pos[j + 1])
            seg = df.iloc[start_i : end_i + 1].copy()
            if not len(seg):
                continue

            k_max_i = int(seg["klinger"].astype(float).idxmax())
            k_min_i = int(seg["klinger"].astype(float).idxmin())
            s_max_i = int(seg["klinger_signal"].astype(float).idxmax())
            s_min_i = int(seg["klinger_signal"].astype(float).idxmin())

            segments.append(
                {
                    "start_i": start_i,
                    "end_i": end_i,
                    "start_dt": str(seg.iloc[0].get("dt") or ""),
                    "end_dt": str(seg.iloc[-1].get("dt") or ""),
                    "start_cross": str(cross_kind[j]),
                    "end_cross": str(cross_kind[j + 1]),
                    "kvo_max": float(df.loc[k_max_i, "klinger"]),
                    "kvo_max_dt": str(df.loc[k_max_i, "dt"]),
                    "kvo_min": float(df.loc[k_min_i, "klinger"]),
                    "kvo_min_dt": str(df.loc[k_min_i, "dt"]),
                    "sig_max": float(df.loc[s_max_i, "klinger_signal"]),
                    "sig_max_dt": str(df.loc[s_max_i, "dt"]),
                    "sig_min": float(df.loc[s_min_i, "klinger_signal"]),
                    "sig_min_dt": str(df.loc[s_min_i, "dt"]),
                }
            )

    seg_tail = int(args.segments_tail)
    if seg_tail > 0:
        segments_to_print = segments[-seg_tail:]
    else:
        segments_to_print = segments

    print("\nSEGMENTS (between crosses) with extrema")
    if not len(segments_to_print):
        print("<none>")
    else:
        for seg in segments_to_print:
            print(
                " | ".join(
                    [
                        f"{seg['start_dt']} -> {seg['end_dt']}",
                        f"i={seg['start_i']}..{seg['end_i']}",
                        f"cross={seg['start_cross']}..{seg['end_cross']}",
                        f"kvo_max={seg['kvo_max']:.6f}@{seg['kvo_max_dt']}",
                        f"kvo_min={seg['kvo_min']:.6f}@{seg['kvo_min_dt']}",
                        f"sig_max={seg['sig_max']:.6f}@{seg['sig_max_dt']}",
                        f"sig_min={seg['sig_min']:.6f}@{seg['sig_min_dt']}",
                    ]
                )
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
