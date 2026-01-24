from __future__ import annotations

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VENV_ROOT = PROJECT_ROOT / "venv_optuna"
VENV_PY = VENV_ROOT / "bin" / "python"
if VENV_PY.exists() and Path(sys.prefix).resolve() != VENV_ROOT.resolve():
    os.execv(str(VENV_PY), [str(VENV_PY), *sys.argv])

import argparse

import pandas as pd

from libs.features.add_quadratic_shape_features_df import add_quadratic_shape_features_df
from libs.features.add_rolling_quantile_features_df import add_rolling_quantile_features_df
from libs.blocks.add_macd_tv_columns_df import add_macd_tv_columns_df
from libs.indicators.momentum.cci_tv import cci_tv
from libs.indicators.momentum.dmi_tv import dmi_tv
from libs.indicators.momentum.vortex_tv import vortex_tv
from libs.indicators.moving_averages.ema_tv import ema_tv
from libs.indicators.moving_averages.sma_tv import sma_tv
from libs.indicators.moving_averages.vwma_tv import vwma_tv
from libs.indicators.volatility.atr_tv import atr_tv
from libs.indicators.volume.mfi_tv import mfi_tv
from libs.indicators.volume.volume_oscillator_tv import volume_oscillator_tv, percentage_volume_oscillator_tv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-csv",
        default="data/processed/klines/LINKUSDT_4h_2020-01-01_2025-12-31.csv",
    )
    ap.add_argument(
        "--out-csv",
        default="data/processed/features/LINKUSDT_4h_2020-01-01_2025-12-31_with_rolling_quantiles.csv",
    )
    args = ap.parse_args()

    in_csv = Path(str(args.in_csv))
    out_csv = Path(str(args.out_csv))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    if "ts" in df.columns:
        df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("Int64")
        df = df.sort_values("ts").reset_index(drop=True)

    if "tp" not in df.columns:
        df["tp"] = (
            pd.to_numeric(df["high"], errors="coerce").astype(float)
            + pd.to_numeric(df["low"], errors="coerce").astype(float)
            + pd.to_numeric(df["close"], errors="coerce").astype(float)
        ) / 3.0

    need = {"macd_line", "macd_signal", "macd_hist"}
    if not need.issubset(set(df.columns)):
        df = add_macd_tv_columns_df(df)

    high = pd.to_numeric(df["high"], errors="coerce").astype(float).tolist()
    low = pd.to_numeric(df["low"], errors="coerce").astype(float).tolist()
    close = pd.to_numeric(df["close"], errors="coerce").astype(float).tolist()
    volume = pd.to_numeric(df["volume"], errors="coerce").astype(float).tolist()

    if "atr_14" not in df.columns:
        df["atr_14"] = atr_tv(high, low, close, 14)
    if "cci_20" not in df.columns:
        df["cci_20"] = cci_tv(high, low, close, 20)
    if "mfi_14" not in df.columns:
        df["mfi_14"] = mfi_tv(high, low, close, volume, 14)
    if "sma_20" not in df.columns:
        df["sma_20"] = sma_tv(close, 20)
    if "ema_20" not in df.columns:
        df["ema_20"] = ema_tv(close, 20)
    if "vwma_20" not in df.columns:
        df["vwma_20"] = vwma_tv(close, volume, 20)

    # Ajout Vortex Indicator (période 14 par défaut TradingView)
    if "vortex_plus_14" not in df.columns or "vortex_minus_14" not in df.columns:
        vortex_plus, vortex_minus = vortex_tv(high, low, close, 14)
        df["vortex_plus_14"] = vortex_plus
        df["vortex_minus_14"] = vortex_minus

    # Ajout Volume Oscillator (fast=10, slow=30 par défaut TradingView)
    if "vol_osc_10_30" not in df.columns:
        df["vol_osc_10_30"] = volume_oscillator_tv(volume, 10, 30)
    if "vol_osc_pct_10_30" not in df.columns:
        df["vol_osc_pct_10_30"] = percentage_volume_oscillator_tv(volume, 10, 30)

    # Ajout indicateurs période 300
    if "vwma_300" not in df.columns:
        df["vwma_300"] = vwma_tv(close, volume, 300)
    if "sma_300" not in df.columns:
        df["sma_300"] = sma_tv(close, 300)
    if "ema_300" not in df.columns:
        df["ema_300"] = ema_tv(close, 300)
    if "cci_300" not in df.columns:
        df["cci_300"] = cci_tv(high, low, close, 300)
    if "vol_osc_12_300" not in df.columns:
        df["vol_osc_12_300"] = volume_oscillator_tv(volume, 12, 300)
    if "vol_osc_pct_12_300" not in df.columns:
        df["vol_osc_pct_12_300"] = percentage_volume_oscillator_tv(volume, 12, 300)
    
    # Vortex période 300
    if "vortex_plus_300" not in df.columns or "vortex_minus_300" not in df.columns:
        vortex_plus_300, vortex_minus_300 = vortex_tv(high, low, close, 300)
        df["vortex_plus_300"] = vortex_plus_300
        df["vortex_minus_300"] = vortex_minus_300

    # DMI (ADX) période 300 et 14
    if "adx_300" not in df.columns:
        adx_300, plus_di_300, minus_di_300 = dmi_tv(high, low, close, 300)
        df["adx_300"] = adx_300
        df["plus_di_300"] = plus_di_300
        df["minus_di_300"] = minus_di_300
    
    if "adx_14" not in df.columns:
        adx_14, plus_di_14, minus_di_14 = dmi_tv(high, low, close, 14)
        df["adx_14"] = adx_14
        df["plus_di_14"] = plus_di_14
        df["minus_di_14"] = minus_di_14

    quantile_cols = [
        "close",
        "volume",
        "tp",
        "macd_line",
        "macd_signal",
        "macd_hist",
        "atr_14",
        "cci_20",
        "mfi_14",
        "sma_20",
        "ema_20",
        "vwma_20",
        "vortex_plus_14",
        "vortex_minus_14",
        "vol_osc_10_30",
        "vol_osc_pct_10_30",
        # Nouveaux indicateurs période 300
        "vwma_300",
        "sma_300",
        "ema_300",
        "cci_300",
        "vol_osc_12_300",
        "vol_osc_pct_12_300",
        "vortex_plus_300",
        "vortex_minus_300",
        "adx_300",
        "plus_di_300",
        "minus_di_300",
        "adx_14",
        "plus_di_14",
        "minus_di_14",
    ]
    out = add_rolling_quantile_features_df(
        df,
        cols=quantile_cols,
        window=180,
        quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )

    shape_cols = [
        "close",
        "tp",
        "macd_line",
        "macd_signal",
        "macd_hist",
        "atr_14",
        "cci_20",
        "mfi_14",
        "sma_20",
        "ema_20",
        "vwma_20",
        "vortex_plus_14",
        "vortex_minus_14",
        "vol_osc_10_30",
        "vol_osc_pct_10_30",
        # Nouveaux indicateurs période 300
        "vwma_300",
        "sma_300",
        "ema_300",
        "cci_300",
        "vol_osc_12_300",
        "vol_osc_pct_12_300",
        "vortex_plus_300",
        "vortex_minus_300",
        "adx_300",
        "plus_di_300",
        "minus_di_300",
        "adx_14",
        "plus_di_14",
        "minus_di_14",
    ]
    out = add_quadratic_shape_features_df(out, cols=shape_cols, windows=[3, 6, 12])

    out.to_csv(out_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
