from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

import pandas as pd

from libs.indicators.momentum.macd_tv import macd_tv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-csv",
        default="data/processed/klines/LINKUSDT_5m_2020-01-01_2024-12-31.csv",
    )
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--macd-fast", type=int, default=12)
    parser.add_argument("--macd-slow", type=int, default=26)
    parser.add_argument("--macd-signal", type=int, default=9)
    args = parser.parse_args()

    in_path = Path(args.in_csv)
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    out_csv = args.out_csv
    if not out_csv:
        out_csv = str(in_path).replace(
            ".csv",
            f"_with_macd_{args.macd_fast}_{args.macd_slow}_{args.macd_signal}.csv",
        )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    if "close" not in df.columns:
        raise ValueError("Input CSV must contain a 'close' column")

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    close = df["close"].astype(float).tolist()

    macd_line, macd_signal, macd_hist = macd_tv(
        close,
        fast_period=int(args.macd_fast),
        slow_period=int(args.macd_slow),
        signal_period=int(args.macd_signal),
    )

    df["macd_line"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    df.to_csv(out_path, index=False)
    print(str(out_path))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
