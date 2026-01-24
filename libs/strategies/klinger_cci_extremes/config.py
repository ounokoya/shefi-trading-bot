from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KlingerCciExtremesConfig:
    ts_col: str = "ts"
    dt_col: str = "dt"

    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: str = "volume"

    tranche_source: str = "kvo_diff"

    flow_indicator: str = "klinger"
    flow_signal_period: int = 13
    nvi_start: float = 1000.0
    pvi_start: float = 1000.0

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    macd_line_col: str = "macd_line"
    macd_signal_col: str = "macd_signal"
    macd_hist_col: str = "macd_hist"

    kvo_fast: int = 34
    kvo_slow: int = 55
    kvo_signal: int = 13
    vf_use_abs_temp: bool = True

    dmi_period: int = 14
    dmi_adx_smoothing: int = 14
    adx_force_threshold: float = 20.0
    adx_force_confirm_bars: int = 3

    cci_extreme_level: float = 100.0

    enable_cci_14: bool = True
    enable_cci_30: bool = True
    enable_cci_300: bool = True

    cci_14_period: int = 14
    cci_30_period: int = 30
    cci_300_period: int = 300

    cci_14_col: str = "cci_14"
    cci_30_col: str = "cci_30"
    cci_300_col: str = "cci_300"

    reference_cci: int = 30

    confirm_bars_weak: int = 1
    confirm_bars_strong: int = 2
