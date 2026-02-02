from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from libs.data_loader import get_crypto_data
from libs.indicators.asi import asi_by_market
from libs.indicators.momentum.cci_tv import cci_tv
from libs.indicators.momentum.dmi_tv import dmi_tv
from libs.indicators.momentum.macd_tv import macd_tv
from libs.indicators.volume.klinger_oscillator_tv import klinger_oscillator_tv
from libs.indicators.volume.mfi_tv import mfi_tv
from libs.indicators.volume.pvt_tv import pvt_tv
from libs.new_strategie.indicators import _compute_stoch
from libs.zerem.timeframes import indicator_params_for_tf, tf_to_timedelta


def _offline_find_covering_cache(
    *,
    cache_dir: Path,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
) -> Optional[Path]:
    if cache_dir is None or not Path(cache_dir).exists():
        return None

    want_start = pd.Timestamp(str(start_date), tz="UTC")
    want_end = pd.Timestamp(str(end_date), tz="UTC")

    pat = re.compile(
        rf"^{re.escape(str(symbol))}_{re.escape(str(timeframe))}_(\d{{4}}-\d{{2}}-\d{{2}})_(\d{{4}}-\d{{2}}-\d{{2}})\.csv$"
    )

    best_path: Optional[Path] = None
    best_span_days: Optional[int] = None
    for p in cache_dir.glob(f"{symbol}_{timeframe}_*.csv"):
        m = pat.match(p.name)
        if not m:
            continue
        s0, e0 = m.group(1), m.group(2)
        try:
            have_start = pd.Timestamp(s0, tz="UTC")
            have_end = pd.Timestamp(e0, tz="UTC")
        except Exception:
            continue
        if have_start <= want_start and have_end >= want_end:
            span_days = int((have_end - have_start).days)
            if best_span_days is None or span_days < best_span_days:
                best_span_days = span_days
                best_path = Path(p)
    return best_path


def load_zerem_df(
    *,
    project_root: Path,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    allow_warmup_before_start: bool,
    warmup_bars: int,
    offline: bool,
    cache_validate: bool,
    cache_max_missing: int,
) -> Optional[pd.DataFrame]:
    start_dt = pd.Timestamp(start_date, tz="UTC")
    end_dt = pd.Timestamp(end_date, tz="UTC")
    if end_dt < start_dt:
        raise ValueError("end_date must be >= start_date")

    if bool(allow_warmup_before_start) and int(warmup_bars) > 0:
        start_fetch = (start_dt - tf_to_timedelta(str(timeframe), int(warmup_bars) + 10)).strftime("%Y-%m-%d")
    else:
        start_fetch = start_dt.strftime("%Y-%m-%d")

    if bool(offline):
        cache_dir = Path(project_root) / "data" / "raw" / "klines_cache"
        cache_path = cache_dir / f"{symbol}_{timeframe}_{start_fetch}_{end_date}.csv"
        if not cache_path.exists():
            alt = _offline_find_covering_cache(
                cache_dir=cache_dir,
                symbol=str(symbol),
                timeframe=str(timeframe),
                start_date=str(start_fetch),
                end_date=str(end_date),
            )
            if alt is None:
                return None
            cache_path = alt

        df = pd.read_csv(cache_path)
        if "ts" in df.columns:
            df["ts"] = pd.to_numeric(df["ts"], errors="coerce").dropna().astype(int)
            df = df.sort_values("ts").reset_index(drop=True)
        if "dt" not in df.columns and "ts" in df.columns:
            df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    else:
        df = get_crypto_data(
            symbol=str(symbol),
            start_date=str(start_fetch),
            end_date=str(end_date),
            timeframe=str(timeframe),
            project_root=Path(project_root),
            cache_max_missing=int(cache_max_missing),
            cache_validate=bool(cache_validate),
        )

    if df is None or df.empty:
        return None

    if "dt" not in df.columns:
        df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["dt"] = pd.to_datetime(df["dt"], utc=True, errors="coerce")

    tf_params = indicator_params_for_tf(str(timeframe))

    high = df["high"].astype(float).tolist()
    low = df["low"].astype(float).tolist()
    close = df["close"].astype(float).tolist()
    volume = df["volume"].astype(float).tolist()

    df = df.copy()
    df["cci"] = cci_tv(high, low, close, period=int(tf_params["cci_period"]))
    df["asi"] = asi_by_market(df, market="crypto")["ASI"]
    df["pvt"] = pvt_tv(close, volume)

    macd_line, macd_signal, macd_hist = macd_tv(close, fast_period=12, slow_period=26, signal_period=9)
    df["macd_line"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    adx, di_plus, di_minus = dmi_tv(high, low, close, period=14, adx_smoothing=14)
    df["dx"] = [abs(p - m) for p, m in zip(di_plus, di_minus)]

    kvo, ks = klinger_oscillator_tv(high, low, close, volume, fast=34, slow=55, signal=13)
    df["kvo"] = kvo
    df["klinger_signal"] = ks

    df["mfi"] = mfi_tv(high, low, close, volume, period=14)

    stoch_k, stoch_d = _compute_stoch(
        df,
        high_col="high",
        low_col="low",
        close_col="close",
        k_period=int(tf_params["stoch_k_period"]),
        k_smooth_period=int(tf_params["stoch_k_smooth"]),
        d_period=int(tf_params["stoch_d_period"]),
    )
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_d

    df = df[(df["dt"] >= pd.Timestamp(start_date, tz="UTC")) & (df["dt"] <= pd.Timestamp(end_date, tz="UTC"))].reset_index(drop=True)
    if df.empty:
        return None

    return df
