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
from libs.blocks.get_current_tranche_series_extreme_signal import get_current_tranche_series_extreme_signal
from libs.indicators.momentum.cci_tv import cci_tv
from libs.indicators.moving_averages.vwma_tv import vwma_tv
from libs.market_data.binance.build_cumulative_klines_csv import build_cumulative_klines_csv
from libs.market_data.binance.dump_um_klines import dump_um_klines


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
    ap.add_argument("--series-col", default="close")
    ap.add_argument(
        "--out-candidates-csv",
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

    if "macd_hist" not in df.columns or "macd_line" not in df.columns or "macd_signal" not in df.columns:
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

    if "cci_30" not in df.columns:
        df["cci_30"] = cci_tv(high, low, close, 30)
    if "cci_120" not in df.columns:
        df["cci_120"] = cci_tv(high, low, close, 120)
    if "cci_300" not in df.columns:
        df["cci_300"] = cci_tv(high, low, close, 300)
    if "vwma_4" not in df.columns:
        df["vwma_4"] = vwma_tv(close, volume, 4)

    win = int(args.window)
    if win < 2:
        raise ValueError("--window must be >= 2")
    if len(df) < win:
        raise ValueError(f"Not enough rows for window={win}: got {len(df)}")

    rows: list[dict[str, object]] = []
    series_col = str(args.series_col)
    if series_col not in df.columns:
        raise ValueError(f"--series-col {series_col} not found in dataframe")

    for end_idx in range(win, len(df) + 1):
        w = df.iloc[end_idx - win : end_idx]
        sig = get_current_tranche_series_extreme_signal(
            w,
            series_col=series_col,
            ts_col="ts",
            hist_col="macd_hist",
        )
        if not bool(sig.get("is_extreme_confirmed_now")):
            continue

        cand = w.iloc[-2]
        now = w.iloc[-1]

        window_end_ts = int(now["ts"])
        window_end_close = float(now["close"])
        tranche_start_ts = sig.get("tranche_start_ts")
        extreme_ts = sig.get("extreme_ts")

        cand_ts = int(cand["ts"])
        cand_close = float(cand["close"])
        cand_value = float(pd.to_numeric(cand[series_col], errors="coerce"))
        now_value = float(pd.to_numeric(now[series_col], errors="coerce"))

        row = {
            "series_col": series_col,
            "window_end_ts": window_end_ts,
            "window_end_dt": _ms_to_iso_utc(window_end_ts),
            "window_end_close": window_end_close,
            "window_end_value": now_value,
            "open_side": sig.get("open_side"),
            "tranche_sign": sig.get("tranche_sign"),
            "tranche_start_ts": tranche_start_ts,
            "tranche_start_dt": _ms_to_iso_utc(tranche_start_ts),
            "tranche_len": sig.get("tranche_len"),
            "extreme_kind": sig.get("extreme_kind"),
            "extreme_ts": extreme_ts,
            "extreme_dt": _ms_to_iso_utc(extreme_ts),
            "cand_ts": cand_ts,
            "cand_dt": _ms_to_iso_utc(cand_ts),
            "cand_close": cand_close,
            "cand_value": cand_value,
            "cand_macd_hist": float(pd.to_numeric(cand["macd_hist"], errors="coerce")),
            "cand_macd_line": float(pd.to_numeric(cand["macd_line"], errors="coerce")),
            "cand_cci_30": float(pd.to_numeric(cand["cci_30"], errors="coerce")),
            "cand_cci_120": float(pd.to_numeric(cand["cci_120"], errors="coerce")),
            "cand_cci_300": float(pd.to_numeric(cand["cci_300"], errors="coerce")),
            "cand_vwma_4": float(pd.to_numeric(cand["vwma_4"], errors="coerce")),
            "now_macd_hist": float(pd.to_numeric(now["macd_hist"], errors="coerce")),
            "now_macd_line": float(pd.to_numeric(now["macd_line"], errors="coerce")),
            "now_cci_30": float(pd.to_numeric(now["cci_30"], errors="coerce")),
            "now_cci_120": float(pd.to_numeric(now["cci_120"], errors="coerce")),
            "now_cci_300": float(pd.to_numeric(now["cci_300"], errors="coerce")),
            "now_vwma_4": float(pd.to_numeric(now["vwma_4"], errors="coerce")),
        }
        rows.append(row)

        print(
            f"candidate confirmed | series={row['series_col']} end={row['window_end_dt']} side={row['open_side']} "
            f"tranche={row['tranche_sign']} start={row['tranche_start_dt']} len={row['tranche_len']} "
            f"cand={row['cand_dt']} cand_value={row['cand_value']} now_value={row['window_end_value']}"
        )

    out_path = Path(
        str(args.out_candidates_csv)
        if args.out_candidates_csv is not None
        else f"data/processed/klines/candidates_extremes_{str(args.ticker)}_{str(args.interval)}_{args.start}_{args.end}_{series_col}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} candidates to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
