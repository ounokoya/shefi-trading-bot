from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NewStrategieConfig:
    ts_col: str = "ts"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: str = "volume"

    macd_hist_col: str = "macd_hist"
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # DMI
    adx_col: str = "adx"
    dx_col: str = "dx"

    # Accept both naming conventions.
    plus_di_col: str = "plus_di"
    minus_di_col: str = "minus_di"
    alt_plus_di_col: str = "di_plus"
    alt_minus_di_col: str = "di_minus"

    dmi_period: int = 14
    dmi_adx_smoothing: int = 14

    # Stoch
    stoch_k_col: str = "stoch_k"
    stoch_d_col: str = "stoch_d"
    stoch_k_period: int = 14
    stoch_k_smooth_period: int = 3
    stoch_d_period: int = 3
    stoch_extreme_high: float = 80.0
    stoch_extreme_low: float = 20.0

    # CCI
    cci_col: str = "cci"
    cci_period: int = 20
    cci_extreme_level: float = 100.0

    # Pivots
    pivot_zone_pct: float = 0.01
    pivot_merge_pct: float = 0.01
    max_pivots: int = 10

    # Signal condition window: conditions must occur within this many bars (rolling window).
    signal_condition_window_bars: int = 30
