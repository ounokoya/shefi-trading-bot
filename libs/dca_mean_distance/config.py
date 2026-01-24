from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DcaConfig:
    symbol: str
    start_date: str
    end_date: str
    timeframe: str
    capital_usdt: float
    leverage: float
    fee_rate: float
    side: str
    cci_period: int
    cci12_enabled: bool
    cci_long_threshold: float
    cci_short_threshold: float
    mfi_period: int
    mfi_enabled: bool
    mfi_long_threshold: float
    mfi_short_threshold: float
    macd_fast_period: int
    macd_slow_period: int
    macd_signal_period: int
    macd_hist_flip_enabled: bool
    macd_prev_opposite_tranche_enabled: bool
    macd_prev_tranche_cci_enabled: bool
    macd_prev_tranche_cci_period: int
    macd_prev_tranche_cci_bull_threshold: float
    macd_prev_tranche_cci_bear_threshold: float
    macd_prev_tranche_cci_medium_enabled: bool
    macd_prev_tranche_cci_medium_period: int
    macd_prev_tranche_cci_medium_bull_threshold: float
    macd_prev_tranche_cci_medium_bear_threshold: float
    macd_prev_tranche_cci_slow_enabled: bool
    macd_prev_tranche_cci_slow_period: int
    macd_prev_tranche_cci_slow_bull_threshold: float
    macd_prev_tranche_cci_slow_bear_threshold: float
    macd_prev_tranche_mfi_enabled: bool
    macd_prev_tranche_mfi_period: int
    macd_prev_tranche_mfi_low_threshold: float
    macd_prev_tranche_mfi_high_threshold: float
    macd_prev_tranche_dmi_enabled: bool
    macd_prev_tranche_dmi_period: int
    macd_prev_tranche_dmi_adx_smoothing: int
    max_portions: int
    d_start_pct: float
    d_step_pct: float
    tp_pct: float
    tp_mode: str
    tp_d_start_pct: float
    tp_d_step_pct: float
    tp_close_ratio: float
    tp_new_cycle_min_distance_pct: float
    tp_partial_macd_hist_flip_enabled: bool
    tp_partial_prev_opposite_tranche_enabled: bool
    tp_partial_prev_tranche_cci_enabled: bool
    tp_partial_prev_tranche_cci_period: int
    tp_partial_prev_tranche_cci_bull_threshold: float
    tp_partial_prev_tranche_cci_bear_threshold: float
    tp_partial_prev_tranche_cci_medium_enabled: bool
    tp_partial_prev_tranche_cci_medium_period: int
    tp_partial_prev_tranche_cci_medium_bull_threshold: float
    tp_partial_prev_tranche_cci_medium_bear_threshold: float
    tp_partial_prev_tranche_cci_slow_enabled: bool
    tp_partial_prev_tranche_cci_slow_period: int
    tp_partial_prev_tranche_cci_slow_bull_threshold: float
    tp_partial_prev_tranche_cci_slow_bear_threshold: float
    tp_partial_prev_tranche_mfi_enabled: bool
    tp_partial_prev_tranche_mfi_period: int
    tp_partial_prev_tranche_mfi_low_threshold: float
    tp_partial_prev_tranche_mfi_high_threshold: float
    tp_partial_prev_tranche_dmi_enabled: bool
    tp_partial_prev_tranche_dmi_period: int
    tp_partial_prev_tranche_dmi_adx_smoothing: int
    tp_partial_mode: str
    tp_partial_stoch_k_period: int
    tp_partial_stoch_d_period: int
    tp_partial_stoch_min_k: float | None
    tp_partial_stoch_max_k: float | None
    reentry_cooldown_minutes: int
    max_cycles: int
    liquidation_threshold_pct: float
    output_dir: str
    png: bool


def _get(d: dict, path: list[str], default: Any) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_one_side_dca_config_yaml(path: str | Path) -> DcaConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping")

    def _float_or(val: Any, default: float) -> float:
        if val is None:
            return float(default)
        try:
            return float(val)
        except Exception:
            return float(default)

    def _bool_or(val: Any, default: bool, key: str) -> bool:
        if val is None:
            return bool(default)
        if isinstance(val, bool):
            return bool(val)
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            v = val.strip().lower()
            if v in {"true", "1", "yes", "y", "on"}:
                return True
            if v in {"false", "0", "no", "n", "off"}:
                return False
        raise ValueError(f"Invalid boolean for {key}: {val!r} (expected true|false)")

    bt_raw = _get(raw, ["backtest"], {})
    strat_raw = _get(raw, ["strategy"], {})
    ind_raw = _get(raw, ["indicators"], {})
    filt_raw = _get(raw, ["filters"], {})
    out_raw = _get(raw, ["output"], {})

    side = str(strat_raw.get("side", "long") or "long").strip().lower()
    if side not in {"long", "short"}:
        raise ValueError(f"Unexpected strategy.side: {side} (expected: long|short)")

    cci12_raw = _get(filt_raw, ["cci12"], {})
    cci12_enabled = _bool_or(cci12_raw.get("enabled", False), False, "filters.cci12.enabled")
    cci_raw = _get(ind_raw, ["cci"], {})
    cci_period = int(cci12_raw.get("period", cci_raw.get("period", 12)) or 12)
    cci_long_threshold = _float_or(cci12_raw.get("long_threshold", 100.0), 100.0)
    cci_short_threshold = _float_or(cci12_raw.get("short_threshold", 100.0), 100.0)

    mfi_filt_raw = _get(filt_raw, ["mfi"], {})
    mfi_enabled = _bool_or(mfi_filt_raw.get("enabled", False), False, "filters.mfi.enabled")
    mfi_raw = _get(ind_raw, ["mfi"], {})
    mfi_period = int(mfi_filt_raw.get("period", mfi_raw.get("period", 14)) or 14)
    mfi_long_threshold = _float_or(mfi_filt_raw.get("long_threshold", 20.0), 20.0)
    mfi_short_threshold = _float_or(mfi_filt_raw.get("short_threshold", 80.0), 80.0)

    macd_hist_flip_raw = _get(filt_raw, ["macd_hist_flip"], {})
    macd_tp_partial_raw = _get(filt_raw, ["tp_partial_macd_hist_flip"], {})
    macd_entry_cfg_raw = _get(macd_hist_flip_raw, ["macd"], {})
    macd_tp_cfg_raw = _get(macd_tp_partial_raw, ["macd"], {})
    macd_ind_cfg_raw = _get(ind_raw, ["macd"], {})
    macd_cfg_raw = (
        macd_entry_cfg_raw
        if isinstance(macd_entry_cfg_raw, dict) and macd_entry_cfg_raw
        else (macd_tp_cfg_raw if isinstance(macd_tp_cfg_raw, dict) and macd_tp_cfg_raw else macd_ind_cfg_raw)
    )
    macd_fast_period = int(macd_cfg_raw.get("fast_period", 12) or 12)
    macd_slow_period = int(macd_cfg_raw.get("slow_period", 26) or 26)
    macd_signal_period = int(macd_cfg_raw.get("signal_period", 9) or 9)
    macd_hist_flip_enabled = _bool_or(
        macd_hist_flip_raw.get("enabled", False),
        False,
        "filters.macd_hist_flip.enabled",
    )

    macd_prev_opposite_tranche_raw = _get(macd_hist_flip_raw, ["prev_opposite_tranche"], {})
    macd_prev_opposite_tranche_enabled = _bool_or(
        macd_prev_opposite_tranche_raw.get("enabled", False),
        False,
        "filters.macd_hist_flip.prev_opposite_tranche.enabled",
    )

    macd_prev_tranche_cci_raw = _get(macd_prev_opposite_tranche_raw, ["cci"], {})
    macd_prev_tranche_cci_enabled = _bool_or(
        macd_prev_tranche_cci_raw.get("enabled", False),
        False,
        "filters.macd_hist_flip.prev_opposite_tranche.cci.enabled",
    )
    macd_prev_tranche_cci_period = int(macd_prev_tranche_cci_raw.get("period", 12) or 12)
    macd_prev_tranche_cci_bull_threshold = _float_or(macd_prev_tranche_cci_raw.get("bull_threshold", 100.0), 100.0)
    macd_prev_tranche_cci_bear_threshold = _float_or(macd_prev_tranche_cci_raw.get("bear_threshold", 100.0), 100.0)

    macd_prev_tranche_cci_medium_raw = _get(macd_prev_opposite_tranche_raw, ["cci_medium"], {})
    macd_prev_tranche_cci_medium_enabled = _bool_or(
        macd_prev_tranche_cci_medium_raw.get("enabled", False),
        False,
        "filters.macd_hist_flip.prev_opposite_tranche.cci_medium.enabled",
    )
    macd_prev_tranche_cci_medium_period = int(macd_prev_tranche_cci_medium_raw.get("period", 96) or 96)
    macd_prev_tranche_cci_medium_bull_threshold = _float_or(
        macd_prev_tranche_cci_medium_raw.get("bull_threshold", 100.0),
        100.0,
    )
    macd_prev_tranche_cci_medium_bear_threshold = _float_or(
        macd_prev_tranche_cci_medium_raw.get("bear_threshold", 100.0),
        100.0,
    )

    macd_prev_tranche_cci_slow_raw = _get(macd_prev_opposite_tranche_raw, ["cci_slow"], {})
    macd_prev_tranche_cci_slow_enabled = _bool_or(
        macd_prev_tranche_cci_slow_raw.get("enabled", False),
        False,
        "filters.macd_hist_flip.prev_opposite_tranche.cci_slow.enabled",
    )
    macd_prev_tranche_cci_slow_period = int(macd_prev_tranche_cci_slow_raw.get("period", 192) or 192)
    macd_prev_tranche_cci_slow_bull_threshold = _float_or(
        macd_prev_tranche_cci_slow_raw.get("bull_threshold", 100.0),
        100.0,
    )
    macd_prev_tranche_cci_slow_bear_threshold = _float_or(
        macd_prev_tranche_cci_slow_raw.get("bear_threshold", 100.0),
        100.0,
    )

    macd_prev_tranche_mfi_raw = _get(macd_prev_opposite_tranche_raw, ["mfi"], {})
    macd_prev_tranche_mfi_enabled = _bool_or(
        macd_prev_tranche_mfi_raw.get("enabled", False),
        False,
        "filters.macd_hist_flip.prev_opposite_tranche.mfi.enabled",
    )
    macd_prev_tranche_mfi_period = int(macd_prev_tranche_mfi_raw.get("period", 14) or 14)
    macd_prev_tranche_mfi_low_threshold = _float_or(
        macd_prev_tranche_mfi_raw.get("low_threshold", macd_prev_tranche_mfi_raw.get("long_threshold", 20.0)),
        20.0,
    )
    macd_prev_tranche_mfi_high_threshold = _float_or(
        macd_prev_tranche_mfi_raw.get("high_threshold", macd_prev_tranche_mfi_raw.get("short_threshold", 80.0)),
        80.0,
    )

    macd_prev_tranche_dmi_raw = _get(macd_prev_opposite_tranche_raw, ["dmi"], {})
    macd_prev_tranche_dmi_enabled = _bool_or(
        macd_prev_tranche_dmi_raw.get("enabled", False),
        False,
        "filters.macd_hist_flip.prev_opposite_tranche.dmi.enabled",
    )
    macd_prev_tranche_dmi_period = int(macd_prev_tranche_dmi_raw.get("period", 14) or 14)
    macd_prev_tranche_dmi_adx_smoothing = int(macd_prev_tranche_dmi_raw.get("adx_smoothing", 14) or 14)

    symbol = str(bt_raw.get("symbol", ""))
    if not symbol:
        raise ValueError("Missing config: backtest.symbol")
    start_date = str(bt_raw.get("start_date", ""))
    end_date = str(bt_raw.get("end_date", ""))
    timeframe = str(bt_raw.get("timeframe", ""))
    if not start_date or not end_date or not timeframe:
        raise ValueError("Missing config: backtest.start_date/end_date/timeframe")

    out_dir = str(out_raw.get("out_dir", "data/processed/backtests/one_side_dca"))
    png = _bool_or(out_raw.get("png", True), True, "output.png")

    leverage = _float_or(bt_raw.get("leverage", 1.0), 1.0)
    parts_multiplier = _float_or(strat_raw.get("parts_multiplier", 1.0), 1.0)
    max_portions = int(max(100, int(float(leverage) * 10.0 * float(parts_multiplier))))

    tp_mode = str(strat_raw.get("tp_mode", "tp_full") or "tp_full").strip().lower()
    if tp_mode not in {"tp_full", "tp_cycles"}:
        raise ValueError(f"Unexpected strategy.tp_mode: {tp_mode} (expected: tp_full|tp_cycles)")

    tp_d_start_pct = _float_or(strat_raw.get("tp_d_start_pct", strat_raw.get("tp_pct", 2.0)), 2.0)
    tp_d_step_pct = _float_or(strat_raw.get("tp_d_step_pct", 0.0), 0.0)
    tp_close_ratio = _float_or(strat_raw.get("tp_close_ratio", 0.5), 0.5)
    tp_new_cycle_min_distance_pct = _float_or(strat_raw.get("tp_new_cycle_min_distance_pct", 0.0), 0.0)
    tp_partial_macd_hist_flip_raw = _get(filt_raw, ["tp_partial_macd_hist_flip"], {})
    tp_partial_macd_hist_flip_enabled = _bool_or(
        tp_partial_macd_hist_flip_raw.get("enabled", True),
        True,
        "filters.tp_partial_macd_hist_flip.enabled",
    )

    tp_partial_mode = str(tp_partial_macd_hist_flip_raw.get("mode", "macd_hist_flip") or "macd_hist_flip").strip().lower()
    if tp_partial_mode in {"stoch", "stoch_cross", "stoch_kd_cross"}:
        tp_partial_mode = "stoch_cross"
    if tp_partial_mode not in {"macd_hist_flip", "stoch_cross"}:
        raise ValueError(
            f"Unexpected filters.tp_partial_macd_hist_flip.mode: {tp_partial_mode} (expected: macd_hist_flip|stoch_cross)"
        )

    tp_partial_stoch_raw = _get(tp_partial_macd_hist_flip_raw, ["stoch"], {})
    tp_partial_stoch_k_period = int(tp_partial_stoch_raw.get("k_period", 14) or 14)
    tp_partial_stoch_d_period = int(tp_partial_stoch_raw.get("d_period", 3) or 3)
    tp_partial_stoch_min_k_val = tp_partial_stoch_raw.get("min_k", None)
    tp_partial_stoch_max_k_val = tp_partial_stoch_raw.get("max_k", None)
    tp_partial_stoch_min_k = (None if tp_partial_stoch_min_k_val is None else _float_or(tp_partial_stoch_min_k_val, 0.0))
    tp_partial_stoch_max_k = (None if tp_partial_stoch_max_k_val is None else _float_or(tp_partial_stoch_max_k_val, 0.0))
    tp_partial_prev_opposite_tranche_raw = _get(tp_partial_macd_hist_flip_raw, ["prev_opposite_tranche"], {})
    tp_partial_prev_opposite_tranche_enabled = _bool_or(
        tp_partial_prev_opposite_tranche_raw.get("enabled", False),
        False,
        "filters.tp_partial_macd_hist_flip.prev_opposite_tranche.enabled",
    )

    tp_partial_prev_tranche_cci_raw = _get(tp_partial_prev_opposite_tranche_raw, ["cci"], {})
    tp_partial_prev_tranche_cci_enabled = _bool_or(
        tp_partial_prev_tranche_cci_raw.get("enabled", False),
        False,
        "filters.tp_partial_macd_hist_flip.prev_opposite_tranche.cci.enabled",
    )
    tp_partial_prev_tranche_cci_period = int(tp_partial_prev_tranche_cci_raw.get("period", 12) or 12)
    tp_partial_prev_tranche_cci_bull_threshold = _float_or(
        tp_partial_prev_tranche_cci_raw.get("bull_threshold", 100.0),
        100.0,
    )
    tp_partial_prev_tranche_cci_bear_threshold = _float_or(
        tp_partial_prev_tranche_cci_raw.get("bear_threshold", 100.0),
        100.0,
    )

    tp_partial_prev_tranche_cci_medium_raw = _get(tp_partial_prev_opposite_tranche_raw, ["cci_medium"], {})
    tp_partial_prev_tranche_cci_medium_enabled = _bool_or(
        tp_partial_prev_tranche_cci_medium_raw.get("enabled", False),
        False,
        "filters.tp_partial_macd_hist_flip.prev_opposite_tranche.cci_medium.enabled",
    )
    tp_partial_prev_tranche_cci_medium_period = int(tp_partial_prev_tranche_cci_medium_raw.get("period", 96) or 96)
    tp_partial_prev_tranche_cci_medium_bull_threshold = _float_or(
        tp_partial_prev_tranche_cci_medium_raw.get("bull_threshold", 100.0),
        100.0,
    )
    tp_partial_prev_tranche_cci_medium_bear_threshold = _float_or(
        tp_partial_prev_tranche_cci_medium_raw.get("bear_threshold", 100.0),
        100.0,
    )

    tp_partial_prev_tranche_cci_slow_raw = _get(tp_partial_prev_opposite_tranche_raw, ["cci_slow"], {})
    tp_partial_prev_tranche_cci_slow_enabled = _bool_or(
        tp_partial_prev_tranche_cci_slow_raw.get("enabled", False),
        False,
        "filters.tp_partial_macd_hist_flip.prev_opposite_tranche.cci_slow.enabled",
    )
    tp_partial_prev_tranche_cci_slow_period = int(tp_partial_prev_tranche_cci_slow_raw.get("period", 192) or 192)
    tp_partial_prev_tranche_cci_slow_bull_threshold = _float_or(
        tp_partial_prev_tranche_cci_slow_raw.get("bull_threshold", 100.0),
        100.0,
    )
    tp_partial_prev_tranche_cci_slow_bear_threshold = _float_or(
        tp_partial_prev_tranche_cci_slow_raw.get("bear_threshold", 100.0),
        100.0,
    )

    tp_partial_prev_tranche_mfi_raw = _get(tp_partial_prev_opposite_tranche_raw, ["mfi"], {})
    tp_partial_prev_tranche_mfi_enabled = _bool_or(
        tp_partial_prev_tranche_mfi_raw.get("enabled", False),
        False,
        "filters.tp_partial_macd_hist_flip.prev_opposite_tranche.mfi.enabled",
    )
    tp_partial_prev_tranche_mfi_period = int(tp_partial_prev_tranche_mfi_raw.get("period", 14) or 14)
    tp_partial_prev_tranche_mfi_low_threshold = _float_or(
        tp_partial_prev_tranche_mfi_raw.get("low_threshold", tp_partial_prev_tranche_mfi_raw.get("long_threshold", 20.0)),
        20.0,
    )
    tp_partial_prev_tranche_mfi_high_threshold = _float_or(
        tp_partial_prev_tranche_mfi_raw.get("high_threshold", tp_partial_prev_tranche_mfi_raw.get("short_threshold", 80.0)),
        80.0,
    )

    tp_partial_prev_tranche_dmi_raw = _get(tp_partial_prev_opposite_tranche_raw, ["dmi"], {})
    tp_partial_prev_tranche_dmi_enabled = _bool_or(
        tp_partial_prev_tranche_dmi_raw.get("enabled", False),
        False,
        "filters.tp_partial_macd_hist_flip.prev_opposite_tranche.dmi.enabled",
    )
    tp_partial_prev_tranche_dmi_period = int(tp_partial_prev_tranche_dmi_raw.get("period", 14) or 14)
    tp_partial_prev_tranche_dmi_adx_smoothing = int(tp_partial_prev_tranche_dmi_raw.get("adx_smoothing", 14) or 14)

    return DcaConfig(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        capital_usdt=_float_or(bt_raw.get("capital_usdt", 0.0), 0.0),
        leverage=leverage,
        fee_rate=_float_or(bt_raw.get("fee_rate", 0.0015), 0.0015),
        side=side,
        cci_period=cci_period,
        cci12_enabled=cci12_enabled,
        cci_long_threshold=cci_long_threshold,
        cci_short_threshold=cci_short_threshold,
        mfi_period=mfi_period,
        mfi_enabled=mfi_enabled,
        mfi_long_threshold=mfi_long_threshold,
        mfi_short_threshold=mfi_short_threshold,
        macd_fast_period=macd_fast_period,
        macd_slow_period=macd_slow_period,
        macd_signal_period=macd_signal_period,
        macd_hist_flip_enabled=macd_hist_flip_enabled,
        macd_prev_opposite_tranche_enabled=macd_prev_opposite_tranche_enabled,
        macd_prev_tranche_cci_enabled=macd_prev_tranche_cci_enabled,
        macd_prev_tranche_cci_period=macd_prev_tranche_cci_period,
        macd_prev_tranche_cci_bull_threshold=macd_prev_tranche_cci_bull_threshold,
        macd_prev_tranche_cci_bear_threshold=macd_prev_tranche_cci_bear_threshold,
        macd_prev_tranche_cci_medium_enabled=macd_prev_tranche_cci_medium_enabled,
        macd_prev_tranche_cci_medium_period=macd_prev_tranche_cci_medium_period,
        macd_prev_tranche_cci_medium_bull_threshold=macd_prev_tranche_cci_medium_bull_threshold,
        macd_prev_tranche_cci_medium_bear_threshold=macd_prev_tranche_cci_medium_bear_threshold,
        macd_prev_tranche_cci_slow_enabled=macd_prev_tranche_cci_slow_enabled,
        macd_prev_tranche_cci_slow_period=macd_prev_tranche_cci_slow_period,
        macd_prev_tranche_cci_slow_bull_threshold=macd_prev_tranche_cci_slow_bull_threshold,
        macd_prev_tranche_cci_slow_bear_threshold=macd_prev_tranche_cci_slow_bear_threshold,
        macd_prev_tranche_mfi_enabled=macd_prev_tranche_mfi_enabled,
        macd_prev_tranche_mfi_period=macd_prev_tranche_mfi_period,
        macd_prev_tranche_mfi_low_threshold=macd_prev_tranche_mfi_low_threshold,
        macd_prev_tranche_mfi_high_threshold=macd_prev_tranche_mfi_high_threshold,
        macd_prev_tranche_dmi_enabled=macd_prev_tranche_dmi_enabled,
        macd_prev_tranche_dmi_period=macd_prev_tranche_dmi_period,
        macd_prev_tranche_dmi_adx_smoothing=macd_prev_tranche_dmi_adx_smoothing,
        liquidation_threshold_pct=_float_or(bt_raw.get("liquidation_threshold_pct", 0.05), 0.05),
        max_portions=max_portions,
        d_start_pct=_float_or(strat_raw.get("d_start_pct", 0.5), 0.5),
        d_step_pct=_float_or(strat_raw.get("d_step_pct", 0.5), 0.5),
        tp_pct=_float_or(strat_raw.get("tp_pct", 2.0), 2.0),
        tp_mode=tp_mode,
        tp_d_start_pct=tp_d_start_pct,
        tp_d_step_pct=tp_d_step_pct,
        tp_close_ratio=tp_close_ratio,
        tp_new_cycle_min_distance_pct=tp_new_cycle_min_distance_pct,
        tp_partial_macd_hist_flip_enabled=tp_partial_macd_hist_flip_enabled,
        tp_partial_prev_opposite_tranche_enabled=tp_partial_prev_opposite_tranche_enabled,
        tp_partial_prev_tranche_cci_enabled=tp_partial_prev_tranche_cci_enabled,
        tp_partial_prev_tranche_cci_period=tp_partial_prev_tranche_cci_period,
        tp_partial_prev_tranche_cci_bull_threshold=tp_partial_prev_tranche_cci_bull_threshold,
        tp_partial_prev_tranche_cci_bear_threshold=tp_partial_prev_tranche_cci_bear_threshold,
        tp_partial_prev_tranche_cci_medium_enabled=tp_partial_prev_tranche_cci_medium_enabled,
        tp_partial_prev_tranche_cci_medium_period=tp_partial_prev_tranche_cci_medium_period,
        tp_partial_prev_tranche_cci_medium_bull_threshold=tp_partial_prev_tranche_cci_medium_bull_threshold,
        tp_partial_prev_tranche_cci_medium_bear_threshold=tp_partial_prev_tranche_cci_medium_bear_threshold,
        tp_partial_prev_tranche_cci_slow_enabled=tp_partial_prev_tranche_cci_slow_enabled,
        tp_partial_prev_tranche_cci_slow_period=tp_partial_prev_tranche_cci_slow_period,
        tp_partial_prev_tranche_cci_slow_bull_threshold=tp_partial_prev_tranche_cci_slow_bull_threshold,
        tp_partial_prev_tranche_cci_slow_bear_threshold=tp_partial_prev_tranche_cci_slow_bear_threshold,
        tp_partial_prev_tranche_mfi_enabled=tp_partial_prev_tranche_mfi_enabled,
        tp_partial_prev_tranche_mfi_period=tp_partial_prev_tranche_mfi_period,
        tp_partial_prev_tranche_mfi_low_threshold=tp_partial_prev_tranche_mfi_low_threshold,
        tp_partial_prev_tranche_mfi_high_threshold=tp_partial_prev_tranche_mfi_high_threshold,
        tp_partial_prev_tranche_dmi_enabled=tp_partial_prev_tranche_dmi_enabled,
        tp_partial_prev_tranche_dmi_period=tp_partial_prev_tranche_dmi_period,
        tp_partial_prev_tranche_dmi_adx_smoothing=tp_partial_prev_tranche_dmi_adx_smoothing,
        tp_partial_mode=tp_partial_mode,
        tp_partial_stoch_k_period=tp_partial_stoch_k_period,
        tp_partial_stoch_d_period=tp_partial_stoch_d_period,
        tp_partial_stoch_min_k=tp_partial_stoch_min_k,
        tp_partial_stoch_max_k=tp_partial_stoch_max_k,
        reentry_cooldown_minutes=int(strat_raw.get("reentry_cooldown_minutes", 0)),
        max_cycles=int(strat_raw.get("max_cycles", 0)),
        output_dir=out_dir,
        png=png,
    )
