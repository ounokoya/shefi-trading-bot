from __future__ import annotations

import numpy as np
import pandas as pd

from libs.blocks.add_macd_tv_columns_df import add_macd_tv_columns_df
from libs.indicators.momentum.cci_tv import cci_tv
from libs.indicators.momentum.dmi_tv import dmi_tv
from libs.indicators.momentum.vortex_tv import vortex_tv
from libs.indicators.moving_averages.vwma_tv import vwma_tv
from libs.indicators.volatility.atr_tv import atr_tv

from libs.backtest_price_action_tranche.config import FullConfig


def ensure_indicators_df(df: pd.DataFrame, *, cfg: FullConfig, force: bool = False) -> pd.DataFrame:
    df2 = df.copy()

    ts_col = cfg.data.ts_col
    for c in (ts_col, cfg.data.ohlc.open, cfg.data.ohlc.high, cfg.data.ohlc.low, cfg.data.ohlc.close):
        if c not in df2.columns:
            raise ValueError(f"Missing required column: {c}")

    df2[ts_col] = pd.to_numeric(df2[ts_col], errors="coerce").astype("Int64")
    df2 = df2.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    for c in (cfg.data.ohlc.open, cfg.data.ohlc.high, cfg.data.ohlc.low, cfg.data.ohlc.close):
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    if cfg.data.ohlc.volume in df2.columns:
        df2[cfg.data.ohlc.volume] = pd.to_numeric(df2[cfg.data.ohlc.volume], errors="coerce")
    else:
        df2[cfg.data.ohlc.volume] = 0.0

    df2 = df2.dropna(subset=[cfg.data.ohlc.open, cfg.data.ohlc.high, cfg.data.ohlc.low, cfg.data.ohlc.close]).reset_index(drop=True)

    if bool(force) or (not {"macd_line", "macd_hist"}.issubset(set(df2.columns))):
        df2 = add_macd_tv_columns_df(
            df2,
            close_col=cfg.data.ohlc.close,
            fast_period=int(cfg.indicators.macd_fast),
            slow_period=int(cfg.indicators.macd_slow),
            signal_period=int(cfg.indicators.macd_signal),
        )

    high = pd.to_numeric(df2[cfg.data.ohlc.high], errors="coerce").astype(float).tolist()
    low = pd.to_numeric(df2[cfg.data.ohlc.low], errors="coerce").astype(float).tolist()
    close = pd.to_numeric(df2[cfg.data.ohlc.close], errors="coerce").astype(float).tolist()
    volume = pd.to_numeric(df2[cfg.data.ohlc.volume], errors="coerce").astype(float).tolist()

    cci_fast_col = f"cci_{int(cfg.indicators.cci_fast)}"
    cci_medium_col = f"cci_{int(cfg.indicators.cci_medium)}"
    cci_slow_col = f"cci_{int(cfg.indicators.cci_slow)}"
    if bool(force) or (cci_fast_col not in df2.columns):
        df2[cci_fast_col] = cci_tv(high, low, close, int(cfg.indicators.cci_fast))
    if bool(force) or (cci_medium_col not in df2.columns):
        df2[cci_medium_col] = cci_tv(high, low, close, int(cfg.indicators.cci_medium))
    if bool(force) or (cci_slow_col not in df2.columns):
        df2[cci_slow_col] = cci_tv(high, low, close, int(cfg.indicators.cci_slow))

    vwma_fast_col = f"vwma_{int(cfg.indicators.vwma_fast)}"
    vwma_medium_col = f"vwma_{int(cfg.indicators.vwma_medium)}"
    if bool(force) or (vwma_fast_col not in df2.columns):
        df2[vwma_fast_col] = vwma_tv(close, volume, int(cfg.indicators.vwma_fast))
    if bool(force) or (vwma_medium_col not in df2.columns):
        df2[vwma_medium_col] = vwma_tv(close, volume, int(cfg.indicators.vwma_medium))

    if bool(force) or (not {"stoch_k", "stoch_d"}.issubset(set(df2.columns))):
        k_period = int(cfg.indicators.stoch_k)
        d_period = int(cfg.indicators.stoch_d)
        if k_period < 1:
            raise ValueError("indicators.stoch.k must be >= 1")
        if d_period < 1:
            raise ValueError("indicators.stoch.d must be >= 1")

        low_s = pd.to_numeric(df2[cfg.data.ohlc.low], errors="coerce").astype(float)
        high_s = pd.to_numeric(df2[cfg.data.ohlc.high], errors="coerce").astype(float)
        close_s = pd.to_numeric(df2[cfg.data.ohlc.close], errors="coerce").astype(float)

        ll = low_s.rolling(window=k_period, min_periods=k_period).min()
        hh = high_s.rolling(window=k_period, min_periods=k_period).max()
        denom = (hh - ll).astype(float)
        numer = (close_s - ll).astype(float)

        k = 100.0 * (numer / denom.replace(0.0, np.nan))
        df2["stoch_k"] = k
        df2["stoch_d"] = df2["stoch_k"].rolling(window=d_period, min_periods=d_period).mean()

    if bool(force) or (not {"vi_plus", "vi_minus"}.issubset(set(df2.columns))):
        vi_plus, vi_minus = vortex_tv(high, low, close, int(cfg.indicators.vortex_period))
        df2["vi_plus"] = vi_plus
        df2["vi_minus"] = vi_minus

    if bool(force) or (not {"di_plus", "di_minus", "adx"}.issubset(set(df2.columns))):
        adx, di_plus, di_minus = dmi_tv(
            high,
            low,
            close,
            int(cfg.indicators.dmi_period),
            adx_smoothing=int(cfg.indicators.dmi_adx_smoothing),
        )
        df2["adx"] = adx
        df2["di_plus"] = di_plus
        df2["di_minus"] = di_minus

    atr_lens = {int(cfg.indicators.atr_len)}
    if cfg.sl.atr_len is not None:
        atr_lens.add(int(cfg.sl.atr_len))
    for atr_len in sorted(list(atr_lens)):
        atr_col = f"atr_{int(atr_len)}"
        if bool(force) or (atr_col not in df2.columns):
            df2[atr_col] = atr_tv(high, low, close, int(atr_len))

    return df2
