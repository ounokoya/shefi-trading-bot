from __future__ import annotations

import math

import numpy as np
import pandas as pd

from libs.blocks.add_macd_tv_columns_df import add_macd_tv_columns_df
from libs.indicators.momentum.cci_tv import cci_tv
from libs.indicators.momentum.dmi_tv import dmi_tv
from libs.new_strategie.config import NewStrategieConfig


def _compute_dx_from_di(*, plus_di: pd.Series, minus_di: pd.Series) -> pd.Series:
    di_sum = (plus_di + minus_di).astype(float)
    num = (plus_di - minus_di).abs().astype(float)
    dx = 100.0 * (num / di_sum.replace(0.0, np.nan))
    return dx


def _compute_stoch(
    df: pd.DataFrame,
    *,
    high_col: str,
    low_col: str,
    close_col: str,
    k_period: int,
    k_smooth_period: int,
    d_period: int,
) -> tuple[pd.Series, pd.Series]:
    low_s = pd.to_numeric(df[str(low_col)], errors="coerce").astype(float)
    high_s = pd.to_numeric(df[str(high_col)], errors="coerce").astype(float)
    close_s = pd.to_numeric(df[str(close_col)], errors="coerce").astype(float)

    ll = low_s.rolling(window=int(k_period), min_periods=int(k_period)).min()
    hh = high_s.rolling(window=int(k_period), min_periods=int(k_period)).max()
    denom = (hh - ll).astype(float)
    numer = (close_s - ll).astype(float)

    k_raw = 100.0 * (numer / denom.replace(0.0, np.nan))

    ks = int(k_smooth_period)
    if ks <= 1:
        k = k_raw
    else:
        k = k_raw.rolling(window=int(ks), min_periods=int(ks)).mean()

    d = k.rolling(window=int(d_period), min_periods=int(d_period)).mean()
    return k, d


def ensure_indicators_df(df: pd.DataFrame, *, cfg: NewStrategieConfig, force: bool = False) -> pd.DataFrame:
    df2 = df.copy()

    # Basic normalize
    for c in (cfg.ts_col, cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col):
        if str(c) not in df2.columns:
            raise ValueError(f"Missing required column: {c}")

    df2[str(cfg.ts_col)] = pd.to_numeric(df2[str(cfg.ts_col)], errors="coerce").astype("Int64")
    df2 = df2.dropna(subset=[str(cfg.ts_col)]).sort_values(str(cfg.ts_col)).reset_index(drop=True)

    for c in (cfg.open_col, cfg.high_col, cfg.low_col, cfg.close_col):
        df2[str(c)] = pd.to_numeric(df2[str(c)], errors="coerce").astype(float)

    if str(cfg.volume_col) in df2.columns:
        df2[str(cfg.volume_col)] = pd.to_numeric(df2[str(cfg.volume_col)], errors="coerce").astype(float)
    else:
        df2[str(cfg.volume_col)] = 0.0

    if str(cfg.macd_hist_col) not in df2.columns or bool(force):
        df2 = add_macd_tv_columns_df(
            df2,
            close_col=str(cfg.close_col),
            fast_period=int(cfg.macd_fast),
            slow_period=int(cfg.macd_slow),
            signal_period=int(cfg.macd_signal),
        )

    # DMI
    need_di = {str(cfg.plus_di_col), str(cfg.minus_di_col), str(cfg.adx_col)}
    alt_di = {str(cfg.alt_plus_di_col), str(cfg.alt_minus_di_col), str(cfg.adx_col)}
    have_main = need_di.issubset(set(df2.columns))
    have_alt = alt_di.issubset(set(df2.columns))

    if bool(force) or (not have_main and not have_alt):
        high = df2[str(cfg.high_col)].astype(float).tolist()
        low = df2[str(cfg.low_col)].astype(float).tolist()
        close = df2[str(cfg.close_col)].astype(float).tolist()

        adx, di_plus, di_minus = dmi_tv(
            high,
            low,
            close,
            int(cfg.dmi_period),
            adx_smoothing=int(cfg.dmi_adx_smoothing),
        )
        df2[str(cfg.adx_col)] = adx
        df2[str(cfg.plus_di_col)] = di_plus
        df2[str(cfg.minus_di_col)] = di_minus

    # normalize di column names
    if str(cfg.plus_di_col) not in df2.columns and str(cfg.alt_plus_di_col) in df2.columns:
        df2[str(cfg.plus_di_col)] = df2[str(cfg.alt_plus_di_col)]
    if str(cfg.minus_di_col) not in df2.columns and str(cfg.alt_minus_di_col) in df2.columns:
        df2[str(cfg.minus_di_col)] = df2[str(cfg.alt_minus_di_col)]

    # DX
    if (str(cfg.dx_col) not in df2.columns) or bool(force):
        pdi = pd.to_numeric(df2[str(cfg.plus_di_col)], errors="coerce").astype(float)
        mdi = pd.to_numeric(df2[str(cfg.minus_di_col)], errors="coerce").astype(float)
        df2[str(cfg.dx_col)] = _compute_dx_from_di(plus_di=pdi, minus_di=mdi)

    # Stoch
    if bool(force) or (not {str(cfg.stoch_k_col), str(cfg.stoch_d_col)}.issubset(set(df2.columns))):
        k, d = _compute_stoch(
            df2,
            high_col=str(cfg.high_col),
            low_col=str(cfg.low_col),
            close_col=str(cfg.close_col),
            k_period=int(cfg.stoch_k_period),
            k_smooth_period=int(cfg.stoch_k_smooth_period),
            d_period=int(cfg.stoch_d_period),
        )
        df2[str(cfg.stoch_k_col)] = k
        df2[str(cfg.stoch_d_col)] = d

    # CCI
    if str(cfg.cci_col) not in df2.columns or bool(force):
        high = df2[str(cfg.high_col)].astype(float).tolist()
        low = df2[str(cfg.low_col)].astype(float).tolist()
        close = df2[str(cfg.close_col)].astype(float).tolist()
        df2[str(cfg.cci_col)] = cci_tv(high, low, close, int(cfg.cci_period))

    if "dt" not in df2.columns:
        try:
            df2["dt"] = pd.to_datetime(df2[str(cfg.ts_col)].astype(int), unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except Exception:
            df2["dt"] = ""

    # final sanity
    for c in (cfg.macd_hist_col, cfg.adx_col, cfg.dx_col, cfg.plus_di_col, cfg.minus_di_col, cfg.stoch_k_col, cfg.stoch_d_col, cfg.cci_col):
        if str(c) not in df2.columns:
            raise ValueError(f"Indicator missing after ensure_indicators_df: {c}")

    # replace inf
    for c in (cfg.macd_hist_col, cfg.adx_col, cfg.dx_col, cfg.plus_di_col, cfg.minus_di_col, cfg.stoch_k_col, cfg.stoch_d_col, cfg.cci_col):
        s = pd.to_numeric(df2[str(c)], errors="coerce").astype(float)
        s = s.replace([np.inf, -np.inf], np.nan)
        df2[str(c)] = s

    return df2
