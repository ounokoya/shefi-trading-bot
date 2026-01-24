from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import datetime as dt

from libs.market_data.binance.build_cumulative_klines_csv import build_cumulative_klines_csv


def _parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="LINKUSDT")
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--start", default="2020-01-01", type=_parse_date)
    parser.add_argument("--end", default="2024-12-31", type=_parse_date)
    parser.add_argument("--dumped-root-dir", default="data/raw/binance_data_vision")
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    out_csv = args.out_csv
    if not out_csv:
        out_csv = (
            Path("data/processed/klines")
            / f"{args.ticker}_{args.interval}_{args.start.isoformat()}_{args.end.isoformat()}.csv"
        )

    out_path = build_cumulative_klines_csv(
        dumped_root_dir=args.dumped_root_dir,
        ticker=args.ticker,
        interval=args.interval,
        date_start=args.start,
        date_end=args.end,
        out_csv=out_csv,
    )

    print(str(out_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
