from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.blocks.add_macd_tv_columns_df import add_macd_tv_columns_df
from libs.blocks.get_current_tranche_extreme_zone_confluence_signal import (
    get_current_tranche_extreme_zone_confluence_signal,
    get_current_tranche_extreme_zone_confluence_tranche_last_signal,
)
from libs.indicators.momentum.cci_tv import cci_tv
from libs.indicators.momentum.dmi_tv import dmi_tv
from libs.indicators.momentum.vortex_tv import vortex_tv
from libs.indicators.moving_averages.vwma_tv import vwma_tv
from libs.market_data.binance.build_cumulative_klines_csv import build_cumulative_klines_csv
from libs.market_data.binance.dump_um_klines import dump_um_klines
from libs.presets.extreme_confluence_presets import get_extreme_confluence_preset


def _parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _ms_to_iso_utc(ms: object) -> str:
    if ms is None:
        return ""
    try:
        msi = int(ms)
    except Exception:
        return ""
    return pd.to_datetime(msi, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")


def _parse_cols_csv(s: str | None) -> list[str]:
    if s is None:
        return []
    parts = [p.strip() for p in str(s).split(",")]
    return [p for p in parts if p]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="LINKUSDT")
    ap.add_argument("--interval", default="5m")
    ap.add_argument("--start", default="2025-12-01", type=_parse_date)
    ap.add_argument("--end", default="2025-12-31", type=_parse_date, help="Inclusive end date")
    ap.add_argument("--root-dir", default="data/raw/binance_data_vision")
    ap.add_argument("--update-existing", action="store_true")
    ap.add_argument("--window", type=int, default=600)
    ap.add_argument("--macd-fast", type=int, default=12)
    ap.add_argument("--macd-slow", type=int, default=26)
    ap.add_argument("--macd-signal", type=int, default=9)
    ap.add_argument(
        "--trend-filter",
        default="none",
        choices=["none", "vortex", "dmi", "both"],
        help="Optional trend filter: restrict zones to one side only based on trend. vortex: vi+/vi- ; dmi: di+/di- ; both: AND.",
    )
    ap.add_argument("--vortex-period", type=int, default=300)
    ap.add_argument("--dmi-period", type=int, default=300)
    ap.add_argument("--dmi-adx-smoothing", type=int, default=14)
    ap.add_argument(
        "--mode",
        default="long",
        choices=["long", "short", "both"],
        help="Which close extreme kind to target: long=LOW close, short=HIGH close, both=accept both",
    )
    ap.add_argument(
        "--confluence-type",
        default="instant",
        choices=["instant", "tranche_last"],
        help="instant: all extremes on same candidate candle; tranche_last: zone triggers when last required series gets its first tranche extreme confirmed",
    )
    ap.add_argument(
        "--preset",
        default=None,
        help="Optional preset defining default confluence columns and CCI thresholds. Examples: scalping, intraday_strict, intraday_standard, swing_strict, swing_standard",
    )
    ap.add_argument(
        "--confluence-cols",
        default=None,
        help="Comma-separated list of columns to use for confluence. If omitted, uses default set.",
    )
    ap.add_argument(
        "--exclude-cols",
        default=None,
        help="Comma-separated list of columns to exclude from confluence.",
    )
    ap.add_argument(
        "--min-confirmed",
        type=int,
        default=None,
        help="Minimum number of series that must confirm (default: all selected series).",
    )
    ap.add_argument(
        "--cci-fast-threshold",
        type=float,
        default=None,
        help="Symmetric CCI threshold for cci_30: LONG requires cci_30 <= -T, SHORT requires cci_30 >= +T.",
    )
    ap.add_argument(
        "--cci-medium-threshold",
        type=float,
        default=None,
        help="Symmetric CCI threshold for cci_120.",
    )
    ap.add_argument(
        "--cci-slow-threshold",
        type=float,
        default=None,
        help="Symmetric CCI threshold for cci_300.",
    )
    ap.add_argument(
        "--out-csv",
        default=None,
    )
    args = ap.parse_args()

    dumped_root = dump_um_klines(
        root_dir=args.root_dir,
        ticker=str(args.ticker),
        interval=str(args.interval),
        date_start=args.start,
        date_end=args.end,
        update_existing=bool(args.update_existing),
    )

    cumulative_csv = Path("data/processed/klines") / f"{str(args.ticker)}_{str(args.interval)}_{args.start}_{args.end}.csv"
    cumulative_csv.parent.mkdir(parents=True, exist_ok=True)

    build_cumulative_klines_csv(
        dumped_root_dir=dumped_root,
        ticker=str(args.ticker),
        interval=str(args.interval),
        date_start=args.start,
        date_end=args.end,
        out_csv=cumulative_csv,
    )

    df = pd.read_csv(cumulative_csv)
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)

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
    volume = pd.to_numeric(df["volume"], errors="coerce").astype(float).tolist()

    df["cci_30"] = cci_tv(high, low, close, 30)
    df["cci_120"] = cci_tv(high, low, close, 120)
    df["cci_300"] = cci_tv(high, low, close, 300)
    df["vwma_4"] = vwma_tv(close, volume, 4)
    df["vwma_12"] = vwma_tv(close, volume, 12)

    vi_plus, vi_minus = vortex_tv(high, low, close, int(args.vortex_period))
    df["vi_plus"] = vi_plus
    df["vi_minus"] = vi_minus

    adx, di_plus, di_minus = dmi_tv(
        high,
        low,
        close,
        int(args.dmi_period),
        adx_smoothing=int(args.dmi_adx_smoothing),
    )
    df["adx"] = adx
    df["di_plus"] = di_plus
    df["di_minus"] = di_minus

    win = int(args.window)
    if win < 2:
        raise ValueError("--window must be >= 2")
    if len(df) < win:
        raise ValueError(f"Not enough rows for window={win}: got {len(df)}")

    preset = None
    if args.preset is not None:
        preset = get_extreme_confluence_preset(str(args.preset))

    default_series_cols = (
        list(preset.series_cols)
        if preset is not None
        else [
            "close",
            "macd_hist",
            "macd_line",
            "cci_30",
            "cci_120",
            "cci_300",
            "vwma_4",
            "vwma_12",
        ]
    )
    series_cols = _parse_cols_csv(args.confluence_cols) or list(default_series_cols)
    excluded = set(_parse_cols_csv(args.exclude_cols))
    series_cols = [c for c in series_cols if c not in excluded]
    if "close" not in series_cols:
        series_cols = ["close"] + series_cols

    mode = str(args.mode).lower()
    if mode == "long":
        target_close_extreme_kind = "LOW"
        out_suffix = "LOW"
    elif mode == "short":
        target_close_extreme_kind = "HIGH"
        out_suffix = "HIGH"
    elif mode == "both":
        target_close_extreme_kind = None
        out_suffix = "BOTH"
    else:
        raise ValueError(f"Unexpected --mode: {mode}")

    confluence_type = str(args.confluence_type).lower()
    min_confirmed = args.min_confirmed
    cci_fast_threshold = (
        args.cci_fast_threshold
        if args.cci_fast_threshold is not None
        else (preset.cci_fast_threshold if preset is not None else None)
    )
    cci_medium_threshold = (
        args.cci_medium_threshold
        if args.cci_medium_threshold is not None
        else (preset.cci_medium_threshold if preset is not None else None)
    )
    cci_slow_threshold = (
        args.cci_slow_threshold
        if args.cci_slow_threshold is not None
        else (preset.cci_slow_threshold if preset is not None else None)
    )

    rows: list[dict[str, object]] = []

    for end_idx in range(win, len(df) + 1):
        w = df.iloc[end_idx - win : end_idx]
        required_min = (len(series_cols) if min_confirmed is None else int(min_confirmed))

        if confluence_type == "tranche_last":
            z = get_current_tranche_extreme_zone_confluence_tranche_last_signal(
                w,
                ts_col="ts",
                hist_col="macd_hist",
                close_col="close",
                series_cols=series_cols,
                target_close_extreme_kind=target_close_extreme_kind,
                cci_fast_threshold=cci_fast_threshold,
                cci_medium_threshold=cci_medium_threshold,
                cci_slow_threshold=cci_slow_threshold,
                min_confirmed=required_min,
                trend_filter=str(args.trend_filter),
            )
        else:
            z = get_current_tranche_extreme_zone_confluence_signal(
                w,
                ts_col="ts",
                hist_col="macd_hist",
                close_col="close",
                series_cols=series_cols,
                target_close_extreme_kind=target_close_extreme_kind,
                cci_fast_threshold=cci_fast_threshold,
                cci_medium_threshold=cci_medium_threshold,
                cci_slow_threshold=cci_slow_threshold,
                min_confirmed=required_min,
                trend_filter=str(args.trend_filter),
            )
        if not bool(z.get("is_zone")):
            continue

        cand = w.iloc[-2]
        now = w.iloc[-1]

        cand_ts = int(cand["ts"])
        now_ts = int(now["ts"])

        row = {
            "confluence_type": confluence_type,
            "cand_ts": cand_ts,
            "cand_dt": _ms_to_iso_utc(cand_ts),
            "now_ts": now_ts,
            "now_dt": _ms_to_iso_utc(now_ts),
            "open_side": z.get("open_side"),
            "close_extreme_kind": z.get("close_extreme_kind"),
            "tranche_sign": z.get("tranche_sign"),
            "tranche_start_ts": z.get("tranche_start_ts"),
            "tranche_start_dt": _ms_to_iso_utc(z.get("tranche_start_ts")),
            "tranche_len": z.get("tranche_len"),
            "confirmed_count": z.get("confirmed_count"),
            "confirmed_series": "|".join(list(z.get("confirmed_series") or [])),
            "newly_confirmed_series": "|".join(list(z.get("newly_confirmed_series") or [])),
            "trend_filter": z.get("trend_filter"),
            "trend_ok": z.get("trend_ok"),
            "trend_vortex_side": z.get("trend_vortex_side"),
            "trend_dmi_side": z.get("trend_dmi_side"),
            "cand_close": float(cand["close"]),
            "now_close": float(now["close"]),
            "cand_macd_hist": float(cand["macd_hist"]),
            "cand_macd_line": float(cand["macd_line"]),
            "cand_cci_30": float(cand["cci_30"]),
            "cand_cci_120": float(cand["cci_120"]),
            "cand_cci_300": float(cand["cci_300"]),
            "cand_vwma_4": float(cand["vwma_4"]),
            "cand_vwma_12": float(cand["vwma_12"]),
            "cand_vi_plus": float(cand["vi_plus"]),
            "cand_vi_minus": float(cand["vi_minus"]),
            "cand_di_plus": float(cand["di_plus"]),
            "cand_di_minus": float(cand["di_minus"]),
            "cand_adx": float(cand["adx"]),
            "now_macd_hist": float(now["macd_hist"]),
            "now_macd_line": float(now["macd_line"]),
            "now_cci_30": float(now["cci_30"]),
            "now_cci_120": float(now["cci_120"]),
            "now_cci_300": float(now["cci_300"]),
            "now_vwma_4": float(now["vwma_4"]),
            "now_vwma_12": float(now["vwma_12"]),
            "now_vi_plus": float(now["vi_plus"]),
            "now_vi_minus": float(now["vi_minus"]),
            "now_di_plus": float(now["di_plus"]),
            "now_di_minus": float(now["di_minus"]),
            "now_adx": float(now["adx"]),
        }
        rows.append(row)

        print(
            f"ZONE confluence | type={row['confluence_type']} kind={row['close_extreme_kind']} cand={row['cand_dt']} now={row['now_dt']} side={row['open_side']} "
            f"tranche={row['tranche_sign']} start={row['tranche_start_dt']} len={row['tranche_len']} newly={row['newly_confirmed_series']} "
            f"trend={row['trend_filter']} ok={row['trend_ok']} vortex={row['trend_vortex_side']} dmi={row['trend_dmi_side']}"
        )

    out_path = Path(
        str(args.out_csv)
        if args.out_csv is not None
        else f"data/processed/klines/extreme_zone_confluence_{str(args.ticker)}_{str(args.interval)}_{args.start}_{args.end}_{out_suffix}_{confluence_type}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} confluence zones to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
