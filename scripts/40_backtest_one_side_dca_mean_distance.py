import sys
from pathlib import Path
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import yaml

# Setup Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.data_loader import get_crypto_data
from libs.indicators.momentum.cci_tv import cci_tv
from libs.indicators.volume.mfi_tv import mfi_tv
from libs.indicators.momentum.macd_tv import macd_tv
from libs.indicators.momentum.dmi_tv import dmi_tv
from libs.dca_mean_distance.config import load_one_side_dca_config_yaml as load_core_config
from libs.dca_mean_distance.engine import run_backtest_one_side_df


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
    reentry_cooldown_minutes: int
    max_cycles: int
    liquidation_threshold_pct: float
    output_dir: str
    png: bool


@dataclass
class MiniCycle:
    cycle_id: int
    created_ts: int
    size: float = 0.0
    avg_open: float = 0.0
    avg_close: float = 0.0
    closed_qty: float = 0.0
    tp_index: int = 0
    next_tp_price: float = 0.0
    tp_reached: bool = False
    current_d_index: int = 0
    next_target_price: float = 0.0


@dataclass
class DcaState:
    wallet: float
    position_size: float = 0.0   # taille en base (ex: BTC)
    avg_price: float = 0.0
    margin_invested: float = 0.0
    portions_used: int = 0
    current_d_index: int = 0     # 0 = pas encore de sécurité, 1 = d_start, etc.
    next_target_price: float = 0.0
    cycles_completed: int = 0
    last_exit_ts: int = 0
    is_liquidated: bool = False
    liquidation_reason: str = ""

    tp_phase: str = "OPEN"  # OPEN|TP (only used in tp_cycles mode)
    tp_active: Optional[MiniCycle] = None
    tp_bucket: List[MiniCycle] = field(default_factory=list)
    tp_next_cycle_id: int = 1
    prev_entry_signal_raw: bool = False
    prev_tp_partial_signal_raw: bool = False


def _get(d: dict, path: List[str], default: Any) -> Any:
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
    macd_prev_tranche_mfi_low_threshold = _float_or(macd_prev_tranche_mfi_raw.get("low_threshold", 20.0), 20.0)
    macd_prev_tranche_mfi_high_threshold = _float_or(macd_prev_tranche_mfi_raw.get("high_threshold", 80.0), 80.0)

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
        tp_partial_prev_tranche_mfi_raw.get("low_threshold", 20.0),
        20.0,
    )
    tp_partial_prev_tranche_mfi_high_threshold = _float_or(
        tp_partial_prev_tranche_mfi_raw.get("high_threshold", 80.0),
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
        reentry_cooldown_minutes=int(strat_raw.get("reentry_cooldown_minutes", 0)),
        max_cycles=int(strat_raw.get("max_cycles", 0)),
        output_dir=out_dir,
        png=png,
    )


def _cycle_ref_price(c: MiniCycle) -> float:
    if c.closed_qty > 0 and c.avg_close > 0:
        return float(c.avg_close)
    return float(c.avg_open)


def _cycle_recompute_next_tp_price(
    c: MiniCycle,
    *,
    is_long: bool,
    tp_d_start_pct: float,
    tp_d_step_pct: float,
) -> None:
    if int(c.tp_index) <= 0:
        ref_price = float(c.avg_open)
        tp_frac = float(tp_d_start_pct) / 100.0
    else:
        ref_price = float(c.avg_close) if (c.closed_qty > 0 and c.avg_close > 0) else float(c.avg_open)
        tp_frac = (float(c.tp_index) * float(tp_d_step_pct)) / 100.0

    if ref_price <= 0:
        c.next_tp_price = 0.0
        return

    if is_long:
        c.next_tp_price = ref_price * (1.0 + tp_frac)
    else:
        c.next_tp_price = ref_price * (1.0 - tp_frac)


def _sync_global_position_from_cycles(state: DcaState) -> None:
    size = 0.0
    w_sum = 0.0

    cycles: List[MiniCycle] = []
    if state.tp_active is not None and state.tp_active.size > 0:
        cycles.append(state.tp_active)
    for c in state.tp_bucket:
        if c.size > 0:
            cycles.append(c)

    for c in cycles:
        size += float(c.size)
        w_sum += float(c.size) * float(c.avg_open)

    state.position_size = float(size)
    state.avg_price = (w_sum / size) if size > 0 else 0.0


def compute_p_new(size: float, avg_price: float, q_notional: float, d_frac: float) -> float:
    """Calcule P_new pour une sécurité donnée la distance cible d_frac.

    Formule issue de docs/dca_mean_distance_spec.md :
    P_new = (S * P_avg - Q * d) / (S * (1 + d))
    """
    if size <= 0:
        return 0.0
    numerator = size * avg_price - q_notional * d_frac
    denom = size * (1.0 + d_frac)
    if denom <= 0 or numerator <= 0:
        return 0.0
    return numerator / denom


def compute_p_new_short(size: float, avg_price: float, q_notional: float, d_frac: float) -> float:
    if size <= 0:
        return 0.0
    denom = size * (1.0 - d_frac)
    if denom <= 0:
        return 0.0
    numerator = size * avg_price + q_notional * d_frac
    if numerator <= 0:
        return 0.0
    return numerator / denom


def run_backtest(cfg: DcaConfig) -> Dict[str, Any]:
    logging.info("Loading data for one-side DCA backtest...")

    df = get_crypto_data(
        cfg.symbol,
        cfg.start_date,
        cfg.end_date,
        cfg.timeframe,
        PROJECT_ROOT,
    )

    if df.empty:
        logging.error("No data loaded. Exiting.")
        return {}

    logging.info(f"Loaded {len(df)} rows for {cfg.symbol} {cfg.timeframe}.")

    try:
        period = max(1, int(cfg.cci_period))
        cci_vals = cci_tv(df["high"].tolist(), df["low"].tolist(), df["close"].tolist(), period)
        df = df.copy()
        df["cci12"] = pd.to_numeric(pd.Series(cci_vals), errors="coerce")
    except Exception:
        df = df.copy()
        df["cci12"] = pd.NA

    try:
        if "volume" in df.columns:
            period = max(1, int(cfg.mfi_period))
            mfi_vals = mfi_tv(
                df["high"].tolist(),
                df["low"].tolist(),
                df["close"].tolist(),
                df["volume"].tolist(),
                period,
            )
            df["mfi"] = pd.to_numeric(pd.Series(mfi_vals), errors="coerce")
        else:
            df["mfi"] = pd.NA
    except Exception:
        df["mfi"] = pd.NA

    try:
        fast_p = max(1, int(cfg.macd_fast_period))
        slow_p = max(1, int(cfg.macd_slow_period))
        signal_p = max(1, int(cfg.macd_signal_period))
        _, _, hist = macd_tv(df["close"].tolist(), fast_p, slow_p, signal_p)
        df["macd_hist"] = pd.to_numeric(pd.Series(hist), errors="coerce")
    except Exception:
        df["macd_hist"] = pd.NA

    need_prev_tranche_entry = bool(cfg.macd_hist_flip_enabled) and bool(cfg.macd_prev_opposite_tranche_enabled)
    need_prev_tranche_tp_partial = bool(cfg.tp_partial_macd_hist_flip_enabled) and bool(cfg.tp_partial_prev_opposite_tranche_enabled)
    need_tp_partial_filters = bool(cfg.tp_partial_macd_hist_flip_enabled)

    tp_partial_mode = str(getattr(cfg, "tp_partial_mode", "macd_hist_flip") or "macd_hist_flip").strip().lower()
    if tp_partial_mode in {"stoch", "stoch_cross", "stoch_kd_cross"}:
        tp_partial_mode = "stoch_cross"

    if need_prev_tranche_entry and bool(cfg.macd_prev_tranche_cci_enabled):
        try:
            period = max(1, int(cfg.macd_prev_tranche_cci_period))
            cci_vals = cci_tv(df["high"].tolist(), df["low"].tolist(), df["close"].tolist(), period)
            df["macd_prev_tranche_cci"] = pd.to_numeric(pd.Series(cci_vals), errors="coerce")
        except Exception:
            df["macd_prev_tranche_cci"] = pd.NA
    else:
        df["macd_prev_tranche_cci"] = pd.NA

    if need_tp_partial_filters and bool(cfg.tp_partial_prev_tranche_cci_enabled):
        try:
            period = max(1, int(cfg.tp_partial_prev_tranche_cci_period))
            cci_vals = cci_tv(df["high"].tolist(), df["low"].tolist(), df["close"].tolist(), period)
            df["tp_prev_tranche_cci"] = pd.to_numeric(pd.Series(cci_vals), errors="coerce")
        except Exception:
            df["tp_prev_tranche_cci"] = pd.NA
    else:
        df["tp_prev_tranche_cci"] = pd.NA

    if need_prev_tranche_entry and bool(cfg.macd_prev_tranche_cci_medium_enabled):
        try:
            period = max(1, int(cfg.macd_prev_tranche_cci_medium_period))
            cci_vals = cci_tv(df["high"].tolist(), df["low"].tolist(), df["close"].tolist(), period)
            df["macd_prev_tranche_cci_medium"] = pd.to_numeric(pd.Series(cci_vals), errors="coerce")
        except Exception:
            df["macd_prev_tranche_cci_medium"] = pd.NA
    else:
        df["macd_prev_tranche_cci_medium"] = pd.NA

    if need_tp_partial_filters and bool(cfg.tp_partial_prev_tranche_cci_medium_enabled):
        try:
            period = max(1, int(cfg.tp_partial_prev_tranche_cci_medium_period))
            cci_vals = cci_tv(df["high"].tolist(), df["low"].tolist(), df["close"].tolist(), period)
            df["tp_prev_tranche_cci_medium"] = pd.to_numeric(pd.Series(cci_vals), errors="coerce")
        except Exception:
            df["tp_prev_tranche_cci_medium"] = pd.NA
    else:
        df["tp_prev_tranche_cci_medium"] = pd.NA

    if need_prev_tranche_entry and bool(cfg.macd_prev_tranche_cci_slow_enabled):
        try:
            period = max(1, int(cfg.macd_prev_tranche_cci_slow_period))
            cci_vals = cci_tv(df["high"].tolist(), df["low"].tolist(), df["close"].tolist(), period)
            df["macd_prev_tranche_cci_slow"] = pd.to_numeric(pd.Series(cci_vals), errors="coerce")
        except Exception:
            df["macd_prev_tranche_cci_slow"] = pd.NA
    else:
        df["macd_prev_tranche_cci_slow"] = pd.NA

    if need_tp_partial_filters and bool(cfg.tp_partial_prev_tranche_cci_slow_enabled):
        try:
            period = max(1, int(cfg.tp_partial_prev_tranche_cci_slow_period))
            cci_vals = cci_tv(df["high"].tolist(), df["low"].tolist(), df["close"].tolist(), period)
            df["tp_prev_tranche_cci_slow"] = pd.to_numeric(pd.Series(cci_vals), errors="coerce")
        except Exception:
            df["tp_prev_tranche_cci_slow"] = pd.NA
    else:
        df["tp_prev_tranche_cci_slow"] = pd.NA

    if need_prev_tranche_entry and bool(cfg.macd_prev_tranche_mfi_enabled) and ("volume" in df.columns):
        try:
            period = max(1, int(cfg.macd_prev_tranche_mfi_period))
            mfi_vals = mfi_tv(
                df["high"].tolist(),
                df["low"].tolist(),
                df["close"].tolist(),
                df["volume"].tolist(),
                period,
            )
            df["macd_prev_tranche_mfi"] = pd.to_numeric(pd.Series(mfi_vals), errors="coerce")
        except Exception:
            df["macd_prev_tranche_mfi"] = pd.NA
    else:
        df["macd_prev_tranche_mfi"] = pd.NA

    if need_tp_partial_filters and bool(cfg.tp_partial_prev_tranche_mfi_enabled) and ("volume" in df.columns):
        try:
            period = max(1, int(cfg.tp_partial_prev_tranche_mfi_period))
            mfi_vals = mfi_tv(
                df["high"].tolist(),
                df["low"].tolist(),
                df["close"].tolist(),
                df["volume"].tolist(),
                period,
            )
            df["tp_prev_tranche_mfi"] = pd.to_numeric(pd.Series(mfi_vals), errors="coerce")
        except Exception:
            df["tp_prev_tranche_mfi"] = pd.NA
    else:
        df["tp_prev_tranche_mfi"] = pd.NA

    if need_prev_tranche_entry and bool(cfg.macd_prev_tranche_dmi_enabled):
        try:
            period = max(1, int(cfg.macd_prev_tranche_dmi_period))
            adx_smoothing = max(1, int(cfg.macd_prev_tranche_dmi_adx_smoothing))
            adx, plus_di, minus_di = dmi_tv(
                df["high"].tolist(),
                df["low"].tolist(),
                df["close"].tolist(),
                period,
                adx_smoothing,
            )
            df["dmi_adx"] = pd.to_numeric(pd.Series(adx), errors="coerce")
            df["dmi_plus_di"] = pd.to_numeric(pd.Series(plus_di), errors="coerce")
            df["dmi_minus_di"] = pd.to_numeric(pd.Series(minus_di), errors="coerce")
            plus_s = df["dmi_plus_di"]
            minus_s = df["dmi_minus_di"]
            di_sum = plus_s + minus_s
            dx_s = ((plus_s - minus_s).abs() / di_sum.replace(0.0, pd.NA)) * 100.0
            df["dmi_dx"] = pd.to_numeric(dx_s, errors="coerce")
        except Exception:
            df["dmi_adx"] = pd.NA
            df["dmi_plus_di"] = pd.NA
            df["dmi_minus_di"] = pd.NA
            df["dmi_dx"] = pd.NA
    else:
        df["dmi_adx"] = pd.NA
        df["dmi_plus_di"] = pd.NA
        df["dmi_minus_di"] = pd.NA
        df["dmi_dx"] = pd.NA

    if need_tp_partial_filters and bool(cfg.tp_partial_prev_tranche_dmi_enabled):
        try:
            period = max(1, int(cfg.tp_partial_prev_tranche_dmi_period))
            adx_smoothing = max(1, int(cfg.tp_partial_prev_tranche_dmi_adx_smoothing))
            adx, plus_di, minus_di = dmi_tv(
                df["high"].tolist(),
                df["low"].tolist(),
                df["close"].tolist(),
                period,
                adx_smoothing,
            )
            df["tp_dmi_adx"] = pd.to_numeric(pd.Series(adx), errors="coerce")
            df["tp_dmi_plus_di"] = pd.to_numeric(pd.Series(plus_di), errors="coerce")
            df["tp_dmi_minus_di"] = pd.to_numeric(pd.Series(minus_di), errors="coerce")
            plus_s = df["tp_dmi_plus_di"]
            minus_s = df["tp_dmi_minus_di"]
            di_sum = plus_s + minus_s
            dx_s = ((plus_s - minus_s).abs() / di_sum.replace(0.0, pd.NA)) * 100.0
            df["tp_dmi_dx"] = pd.to_numeric(dx_s, errors="coerce")
        except Exception:
            df["tp_dmi_adx"] = pd.NA
            df["tp_dmi_plus_di"] = pd.NA
            df["tp_dmi_minus_di"] = pd.NA
            df["tp_dmi_dx"] = pd.NA
    else:
        df["tp_dmi_adx"] = pd.NA
        df["tp_dmi_plus_di"] = pd.NA
        df["tp_dmi_minus_di"] = pd.NA
        df["tp_dmi_dx"] = pd.NA

    if tp_partial_mode == "stoch_cross":
        try:
            k_period = max(1, int(getattr(cfg, "tp_partial_stoch_k_period", 14) or 14))
            d_period = max(1, int(getattr(cfg, "tp_partial_stoch_d_period", 3) or 3))

            low_s = pd.to_numeric(df["low"], errors="coerce").astype(float)
            high_s = pd.to_numeric(df["high"], errors="coerce").astype(float)
            close_s = pd.to_numeric(df["close"], errors="coerce").astype(float)
            ll = low_s.rolling(window=k_period, min_periods=k_period).min()
            hh = high_s.rolling(window=k_period, min_periods=k_period).max()
            denom = (hh - ll).astype(float)
            numer = (close_s - ll).astype(float)
            k = 100.0 * (numer / denom.replace(0.0, pd.NA))
            df["stoch_k"] = pd.to_numeric(k, errors="coerce")
            df["stoch_d"] = pd.to_numeric(df["stoch_k"].rolling(window=d_period, min_periods=d_period).mean(), errors="coerce")
        except Exception:
            df["stoch_k"] = pd.NA
            df["stoch_d"] = pd.NA
    else:
        df["stoch_k"] = pd.NA
        df["stoch_d"] = pd.NA

    # Basic timeframe sanity check
    try:
        if "open_time" in df.columns and len(df) >= 3:
            ts_sorted = pd.to_numeric(df["open_time"], errors="coerce").dropna().astype(int).sort_values()
            if len(ts_sorted) >= 3:
                d_ms = ts_sorted.diff().dropna()
                inferred_ms = float(d_ms.median()) if not d_ms.empty else 0.0
                tf = (cfg.timeframe or "").strip().lower()
                expected_ms = 0.0
                if tf.endswith("min") and tf[:-3].isdigit():
                    expected_ms = float(int(tf[:-3]) * 60_000)
                elif tf.endswith("m") and tf[:-1].isdigit():
                    expected_ms = float(int(tf[:-1]) * 60_000)
                elif tf.endswith("h") and tf[:-1].isdigit():
                    expected_ms = float(int(tf[:-1]) * 3_600_000)
                elif tf.endswith("d") and tf[:-1].isdigit():
                    expected_ms = float(int(tf[:-1]) * 86_400_000)
                if expected_ms > 0 and inferred_ms > 0:
                    inferred_min = inferred_ms / 60_000.0
                    expected_min = expected_ms / 60_000.0
                    if abs(inferred_ms - expected_ms) > 1.0:
                        logging.warning(
                            f"Data timeframe mismatch? config={cfg.timeframe!r} (~{expected_min:.2f}m) vs inferred~{inferred_min:.2f}m"
                        )
                    else:
                        logging.info(f"Data timeframe OK: {cfg.timeframe!r} (~{expected_min:.2f}m)")
    except Exception:
        pass

    return run_backtest_one_side_df(cfg=cfg, df=df)

    # Setup DCA state
    portion_margin = cfg.capital_usdt / float(cfg.max_portions)
    portion_notional = portion_margin * cfg.leverage
    q_notional = portion_notional
    q_margin = q_notional / float(cfg.leverage) if float(cfg.leverage) > 0 else 0.0

    state = DcaState(wallet=cfg.capital_usdt)

    equity_rows: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []

    run_stats: Dict[str, Any] = {
        "dca_price_triggers": 0,
        "dca_blocked_indicator": 0,
        "dca_blocked_mfi": 0,
        "dca_blocked_macd": 0,
        "dca_blocked_macd_prev_tranche": 0,
        "dca_blocked_macd_prev_cci": 0,
        "dca_blocked_macd_prev_cci_medium": 0,
        "dca_blocked_macd_prev_cci_slow": 0,
        "dca_blocked_macd_prev_mfi": 0,
        "dca_blocked_macd_prev_dmi": 0,
        "dca_blocked_margin": 0,
        "dca_executed": 0,
    }

    d_start = cfg.d_start_pct / 100.0
    d_step = cfg.d_step_pct / 100.0
    tp_frac = cfg.tp_pct / 100.0

    is_long = str(cfg.side).strip().lower() != "short"
    side_label = "LONG" if is_long else "SHORT"

    prev_hist_sign = 0
    current_tranche_sign = 0
    current_tranche_seen_cci = False
    current_tranche_seen_cci_medium = False
    current_tranche_seen_cci_slow = False
    current_tranche_seen_mfi = False
    current_tranche_seen_dmi = False
    current_tranche_seen_cci_tp = False
    current_tranche_seen_cci_medium_tp = False
    current_tranche_seen_cci_slow_tp = False
    current_tranche_seen_mfi_tp = False
    current_tranche_seen_dmi_tp = False
    last_tranche_sign = 0
    last_tranche_seen_cci = False
    last_tranche_seen_cci_medium = False
    last_tranche_seen_cci_slow = False
    last_tranche_seen_mfi = False
    last_tranche_seen_dmi = False
    last_tranche_seen_cci_tp = False
    last_tranche_seen_cci_medium_tp = False
    last_tranche_seen_cci_slow_tp = False
    last_tranche_seen_mfi_tp = False
    last_tranche_seen_dmi_tp = False

    for i, row in df.iterrows():
        ts = int(row["open_time"])
        price = float(row["close"])
        prev_hist_sign_before = int(prev_hist_sign)
        cci12_val = row.get("cci12", pd.NA)
        mfi_val = row.get("mfi", pd.NA)
        macd_hist_val = row.get("macd_hist", pd.NA)
        macd_prev_tranche_cci_val = row.get("macd_prev_tranche_cci", pd.NA)
        macd_prev_tranche_cci_medium_val = row.get("macd_prev_tranche_cci_medium", pd.NA)
        macd_prev_tranche_cci_slow_val = row.get("macd_prev_tranche_cci_slow", pd.NA)
        macd_prev_tranche_mfi_val = row.get("macd_prev_tranche_mfi", pd.NA)
        tp_prev_tranche_cci_val = row.get("tp_prev_tranche_cci", pd.NA)
        tp_prev_tranche_cci_medium_val = row.get("tp_prev_tranche_cci_medium", pd.NA)
        tp_prev_tranche_cci_slow_val = row.get("tp_prev_tranche_cci_slow", pd.NA)
        tp_prev_tranche_mfi_val = row.get("tp_prev_tranche_mfi", pd.NA)
        dmi_adx_val = row.get("dmi_adx", pd.NA)
        dmi_plus_di_val = row.get("dmi_plus_di", pd.NA)
        dmi_minus_di_val = row.get("dmi_minus_di", pd.NA)
        dmi_dx_val = row.get("dmi_dx", pd.NA)
        tp_dmi_adx_val = row.get("tp_dmi_adx", pd.NA)
        tp_dmi_plus_di_val = row.get("tp_dmi_plus_di", pd.NA)
        tp_dmi_minus_di_val = row.get("tp_dmi_minus_di", pd.NA)
        tp_dmi_dx_val = row.get("tp_dmi_dx", pd.NA)

        hist_sign = 0
        if not pd.isna(macd_hist_val):
            h = float(macd_hist_val)
            if h > 0:
                hist_sign = 1
            elif h < 0:
                hist_sign = -1
            else:
                hist_sign = prev_hist_sign

        macd_flip_to_side = False
        sign_changed = False
        if prev_hist_sign != 0 and hist_sign != 0 and hist_sign != prev_hist_sign:
            sign_changed = True
            if is_long:
                macd_flip_to_side = (prev_hist_sign == -1 and hist_sign == 1)
            else:
                macd_flip_to_side = (prev_hist_sign == 1 and hist_sign == -1)

        if hist_sign != 0 and current_tranche_sign == 0:
            current_tranche_sign = hist_sign
            current_tranche_seen_cci = False
            current_tranche_seen_cci_medium = False
            current_tranche_seen_cci_slow = False
            current_tranche_seen_mfi = False
            current_tranche_seen_dmi = False
            current_tranche_seen_cci_tp = False
            current_tranche_seen_cci_medium_tp = False
            current_tranche_seen_cci_slow_tp = False
            current_tranche_seen_mfi_tp = False
            current_tranche_seen_dmi_tp = False

        if sign_changed and current_tranche_sign != 0:
            last_tranche_sign = current_tranche_sign
            last_tranche_seen_cci = bool(current_tranche_seen_cci)
            last_tranche_seen_cci_medium = bool(current_tranche_seen_cci_medium)
            last_tranche_seen_cci_slow = bool(current_tranche_seen_cci_slow)
            last_tranche_seen_mfi = bool(current_tranche_seen_mfi)
            last_tranche_seen_dmi = bool(current_tranche_seen_dmi)
            last_tranche_seen_cci_tp = bool(current_tranche_seen_cci_tp)
            last_tranche_seen_cci_medium_tp = bool(current_tranche_seen_cci_medium_tp)
            last_tranche_seen_cci_slow_tp = bool(current_tranche_seen_cci_slow_tp)
            last_tranche_seen_mfi_tp = bool(current_tranche_seen_mfi_tp)
            last_tranche_seen_dmi_tp = bool(current_tranche_seen_dmi_tp)
            current_tranche_sign = hist_sign
            current_tranche_seen_cci = False
            current_tranche_seen_cci_medium = False
            current_tranche_seen_cci_slow = False
            current_tranche_seen_mfi = False
            current_tranche_seen_dmi = False
            current_tranche_seen_cci_tp = False
            current_tranche_seen_cci_medium_tp = False
            current_tranche_seen_cci_slow_tp = False
            current_tranche_seen_mfi_tp = False
            current_tranche_seen_dmi_tp = False

        if need_prev_tranche_entry and current_tranche_sign != 0:
            if bool(cfg.macd_prev_tranche_cci_enabled) and (not current_tranche_seen_cci):
                if not pd.isna(macd_prev_tranche_cci_val):
                    cci_f = float(macd_prev_tranche_cci_val)
                    if current_tranche_sign > 0:
                        current_tranche_seen_cci = cci_f >= abs(float(cfg.macd_prev_tranche_cci_bull_threshold))
                    else:
                        current_tranche_seen_cci = cci_f <= (-abs(float(cfg.macd_prev_tranche_cci_bear_threshold)))

            if bool(cfg.macd_prev_tranche_cci_medium_enabled) and (not current_tranche_seen_cci_medium):
                if not pd.isna(macd_prev_tranche_cci_medium_val):
                    cci_f = float(macd_prev_tranche_cci_medium_val)
                    if current_tranche_sign > 0:
                        current_tranche_seen_cci_medium = cci_f >= abs(float(cfg.macd_prev_tranche_cci_medium_bull_threshold))
                    else:
                        current_tranche_seen_cci_medium = cci_f <= (-abs(float(cfg.macd_prev_tranche_cci_medium_bear_threshold)))

            if bool(cfg.macd_prev_tranche_cci_slow_enabled) and (not current_tranche_seen_cci_slow):
                if not pd.isna(macd_prev_tranche_cci_slow_val):
                    cci_f = float(macd_prev_tranche_cci_slow_val)
                    if current_tranche_sign > 0:
                        current_tranche_seen_cci_slow = cci_f >= abs(float(cfg.macd_prev_tranche_cci_slow_bull_threshold))
                    else:
                        current_tranche_seen_cci_slow = cci_f <= (-abs(float(cfg.macd_prev_tranche_cci_slow_bear_threshold)))

            if bool(cfg.macd_prev_tranche_mfi_enabled) and (not current_tranche_seen_mfi):
                if not pd.isna(macd_prev_tranche_mfi_val):
                    mfi_f = float(macd_prev_tranche_mfi_val)
                    if current_tranche_sign > 0:
                        current_tranche_seen_mfi = mfi_f >= float(cfg.macd_prev_tranche_mfi_high_threshold)
                    else:
                        current_tranche_seen_mfi = mfi_f <= float(cfg.macd_prev_tranche_mfi_low_threshold)

            if bool(cfg.macd_prev_tranche_dmi_enabled) and (not current_tranche_seen_dmi):
                if (not pd.isna(dmi_dx_val)) and (not pd.isna(dmi_plus_di_val)) and (not pd.isna(dmi_minus_di_val)):
                    dx_f = float(dmi_dx_val)
                    plus_f = float(dmi_plus_di_val)
                    minus_f = float(dmi_minus_di_val)
                    di_max = max(plus_f, minus_f)
                    di_min = min(plus_f, minus_f)
                    dx_ok = (dx_f > di_max) or (dx_f < di_min)
                    di_align = (plus_f > minus_f) if current_tranche_sign > 0 else (minus_f > plus_f)
                    current_tranche_seen_dmi = bool(dx_ok and di_align)

        if need_prev_tranche_tp_partial and current_tranche_sign != 0:
            if bool(cfg.tp_partial_prev_tranche_cci_enabled) and (not current_tranche_seen_cci_tp):
                if not pd.isna(tp_prev_tranche_cci_val):
                    cci_f = float(tp_prev_tranche_cci_val)
                    if current_tranche_sign > 0:
                        current_tranche_seen_cci_tp = cci_f >= abs(float(cfg.tp_partial_prev_tranche_cci_bull_threshold))
                    else:
                        current_tranche_seen_cci_tp = cci_f <= (-abs(float(cfg.tp_partial_prev_tranche_cci_bear_threshold)))

            if bool(cfg.tp_partial_prev_tranche_cci_medium_enabled) and (not current_tranche_seen_cci_medium_tp):
                if not pd.isna(tp_prev_tranche_cci_medium_val):
                    cci_f = float(tp_prev_tranche_cci_medium_val)
                    if current_tranche_sign > 0:
                        current_tranche_seen_cci_medium_tp = cci_f >= abs(
                            float(cfg.tp_partial_prev_tranche_cci_medium_bull_threshold)
                        )
                    else:
                        current_tranche_seen_cci_medium_tp = cci_f <= (-abs(
                            float(cfg.tp_partial_prev_tranche_cci_medium_bear_threshold)
                        ))

            if bool(cfg.tp_partial_prev_tranche_cci_slow_enabled) and (not current_tranche_seen_cci_slow_tp):
                if not pd.isna(tp_prev_tranche_cci_slow_val):
                    cci_f = float(tp_prev_tranche_cci_slow_val)
                    if current_tranche_sign > 0:
                        current_tranche_seen_cci_slow_tp = cci_f >= abs(
                            float(cfg.tp_partial_prev_tranche_cci_slow_bull_threshold)
                        )
                    else:
                        current_tranche_seen_cci_slow_tp = cci_f <= (-abs(
                            float(cfg.tp_partial_prev_tranche_cci_slow_bear_threshold)
                        ))

            if bool(cfg.tp_partial_prev_tranche_mfi_enabled) and (not current_tranche_seen_mfi_tp):
                if not pd.isna(tp_prev_tranche_mfi_val):
                    mfi_f = float(tp_prev_tranche_mfi_val)
                    if current_tranche_sign > 0:
                        current_tranche_seen_mfi_tp = mfi_f >= float(cfg.tp_partial_prev_tranche_mfi_high_threshold)
                    else:
                        current_tranche_seen_mfi_tp = mfi_f <= float(cfg.tp_partial_prev_tranche_mfi_low_threshold)

            if bool(cfg.tp_partial_prev_tranche_dmi_enabled) and (not current_tranche_seen_dmi_tp):
                if (not pd.isna(tp_dmi_dx_val)) and (not pd.isna(tp_dmi_plus_di_val)) and (not pd.isna(tp_dmi_minus_di_val)):
                    dx_f = float(tp_dmi_dx_val)
                    plus_f = float(tp_dmi_plus_di_val)
                    minus_f = float(tp_dmi_minus_di_val)
                    di_max = max(plus_f, minus_f)
                    di_min = min(plus_f, minus_f)
                    dx_ok = (dx_f > di_max) or (dx_f < di_min)
                    di_align = (plus_f > minus_f) if current_tranche_sign > 0 else (minus_f > plus_f)
                    current_tranche_seen_dmi_tp = bool(dx_ok and di_align)

        prev_tranche_ok_dbg: Optional[bool] = None
        if need_prev_tranche_entry:
            expected_prev_sign = -1 if is_long else 1
            prev_sign_ok = (last_tranche_sign == expected_prev_sign)
            prev_tranche_cci_ok = True
            if bool(cfg.macd_prev_tranche_cci_enabled):
                prev_tranche_cci_ok = prev_tranche_cci_ok and bool(last_tranche_seen_cci)
            if bool(cfg.macd_prev_tranche_cci_medium_enabled):
                prev_tranche_cci_ok = prev_tranche_cci_ok and bool(last_tranche_seen_cci_medium)
            if bool(cfg.macd_prev_tranche_cci_slow_enabled):
                prev_tranche_cci_ok = prev_tranche_cci_ok and bool(last_tranche_seen_cci_slow)
            prev_tranche_mfi_ok = (not bool(cfg.macd_prev_tranche_mfi_enabled)) or bool(last_tranche_seen_mfi)
            prev_tranche_dmi_ok = (not bool(cfg.macd_prev_tranche_dmi_enabled)) or bool(last_tranche_seen_dmi)
            prev_tranche_ok_dbg = bool(prev_sign_ok and prev_tranche_cci_ok and prev_tranche_mfi_ok and prev_tranche_dmi_ok)

        tp_prev_tranche_ok_dbg: Optional[bool] = None
        if need_prev_tranche_tp_partial:
            expected_prev_sign = 1 if is_long else -1
            prev_sign_ok = (last_tranche_sign == expected_prev_sign)
            prev_tranche_cci_ok = True
            if bool(cfg.tp_partial_prev_tranche_cci_enabled):
                prev_tranche_cci_ok = prev_tranche_cci_ok and bool(last_tranche_seen_cci_tp)
            if bool(cfg.tp_partial_prev_tranche_cci_medium_enabled):
                prev_tranche_cci_ok = prev_tranche_cci_ok and bool(last_tranche_seen_cci_medium_tp)
            if bool(cfg.tp_partial_prev_tranche_cci_slow_enabled):
                prev_tranche_cci_ok = prev_tranche_cci_ok and bool(last_tranche_seen_cci_slow_tp)
            prev_tranche_mfi_ok = (not bool(cfg.tp_partial_prev_tranche_mfi_enabled)) or bool(last_tranche_seen_mfi_tp)
            prev_tranche_dmi_ok = (not bool(cfg.tp_partial_prev_tranche_dmi_enabled)) or bool(last_tranche_seen_dmi_tp)
            tp_prev_tranche_ok_dbg = bool(prev_sign_ok and prev_tranche_cci_ok and prev_tranche_mfi_ok and prev_tranche_dmi_ok)

        if hist_sign != 0:
            prev_hist_sign = hist_sign

        # Keep global position in sync for pnl/liquidation checks in tp_cycles mode
        if str(cfg.tp_mode).strip().lower() == "tp_cycles":
            _sync_global_position_from_cycles(state)

        # PnL non réalisé courant
        if state.position_size > 0 and state.avg_price > 0:
            if is_long:
                pnl_unrealized = (price - state.avg_price) * state.position_size
            else:
                pnl_unrealized = (state.avg_price - price) * state.position_size
        else:
            pnl_unrealized = 0.0

        equity = state.wallet + pnl_unrealized

        wallet_free = state.wallet - state.margin_invested
        pnl_total = pnl_unrealized
        margin_total = state.wallet + max(0.0, pnl_total)
        loss_unrealized = max(0.0, -pnl_total)
        liq_limit = (1.0 - float(cfg.liquidation_threshold_pct)) * margin_total

        if (state.position_size > 0.0) and (loss_unrealized >= liq_limit):
            state.is_liquidated = True
            state.liquidation_reason = (
                f"LIQUIDATION price={price:.6f} loss={loss_unrealized:.2f} limit={liq_limit:.2f} "
                f"wallet={state.wallet:.2f} margin_invested={state.margin_invested:.2f}"
            )
            notional = state.position_size * price
            trades.append(
                {
                    "timestamp": ts,
                    "price": price,
                    "macd_hist": (None if pd.isna(macd_hist_val) else float(macd_hist_val)),
                    "macd_hist_sign": int(hist_sign),
                    "macd_tranche_sign": int(current_tranche_sign),
                    "macd_prev_tranche_sign": int(last_tranche_sign),
                    "macd_flip_to_side": bool(macd_flip_to_side),
                    "macd_prev_tranche_ok": (None if prev_tranche_ok_dbg is None else bool(prev_tranche_ok_dbg)),
                    "macd_prev_tranche_seen_cci": bool(last_tranche_seen_cci),
                    "macd_prev_tranche_seen_cci_medium": bool(last_tranche_seen_cci_medium),
                    "macd_prev_tranche_seen_cci_slow": bool(last_tranche_seen_cci_slow),
                    "macd_prev_tranche_seen_mfi": bool(last_tranche_seen_mfi),
                    "macd_prev_tranche_seen_dmi": bool(last_tranche_seen_dmi),
                    "cci12": (None if pd.isna(cci12_val) else float(cci12_val)),
                    "mfi": (None if pd.isna(mfi_val) else float(mfi_val)),
                    "macd_prev_tranche_cci": (None if pd.isna(macd_prev_tranche_cci_val) else float(macd_prev_tranche_cci_val)),
                    "macd_prev_tranche_cci_medium": (None if pd.isna(macd_prev_tranche_cci_medium_val) else float(macd_prev_tranche_cci_medium_val)),
                    "macd_prev_tranche_cci_slow": (None if pd.isna(macd_prev_tranche_cci_slow_val) else float(macd_prev_tranche_cci_slow_val)),
                    "macd_prev_tranche_mfi": (None if pd.isna(macd_prev_tranche_mfi_val) else float(macd_prev_tranche_mfi_val)),
                    "dmi_adx": (None if pd.isna(dmi_adx_val) else float(dmi_adx_val)),
                    "dmi_plus_di": (None if pd.isna(dmi_plus_di_val) else float(dmi_plus_di_val)),
                    "dmi_minus_di": (None if pd.isna(dmi_minus_di_val) else float(dmi_minus_di_val)),
                    "dmi_dx": (None if pd.isna(dmi_dx_val) else float(dmi_dx_val)),
                    "side": side_label,
                    "type": "LIQUIDATION",
                    "qty": state.position_size,
                    "qty_usdt": notional,
                    "pnl_realized": 0.0,
                    "fee": 0.0,
                }
            )
            equity_rows.append(
                {
                    "timestamp": ts,
                    "price": price,
                    "macd_hist": (None if pd.isna(macd_hist_val) else float(macd_hist_val)),
                    "macd_hist_sign": int(hist_sign),
                    "macd_tranche_sign": int(current_tranche_sign),
                    "macd_prev_tranche_sign": int(last_tranche_sign),
                    "macd_flip_to_side": bool(macd_flip_to_side),
                    "macd_prev_tranche_ok": (None if prev_tranche_ok_dbg is None else bool(prev_tranche_ok_dbg)),
                    "macd_prev_tranche_seen_cci": bool(last_tranche_seen_cci),
                    "macd_prev_tranche_seen_cci_medium": bool(last_tranche_seen_cci_medium),
                    "macd_prev_tranche_seen_cci_slow": bool(last_tranche_seen_cci_slow),
                    "macd_prev_tranche_seen_mfi": bool(last_tranche_seen_mfi),
                    "macd_prev_tranche_seen_dmi": bool(last_tranche_seen_dmi),
                    "cci12": (None if pd.isna(cci12_val) else float(cci12_val)),
                    "mfi": (None if pd.isna(mfi_val) else float(mfi_val)),
                    "macd_prev_tranche_cci": (None if pd.isna(macd_prev_tranche_cci_val) else float(macd_prev_tranche_cci_val)),
                    "macd_prev_tranche_cci_medium": (None if pd.isna(macd_prev_tranche_cci_medium_val) else float(macd_prev_tranche_cci_medium_val)),
                    "macd_prev_tranche_cci_slow": (None if pd.isna(macd_prev_tranche_cci_slow_val) else float(macd_prev_tranche_cci_slow_val)),
                    "macd_prev_tranche_mfi": (None if pd.isna(macd_prev_tranche_mfi_val) else float(macd_prev_tranche_mfi_val)),
                    "dmi_adx": (None if pd.isna(dmi_adx_val) else float(dmi_adx_val)),
                    "dmi_plus_di": (None if pd.isna(dmi_plus_di_val) else float(dmi_plus_di_val)),
                    "dmi_minus_di": (None if pd.isna(dmi_minus_di_val) else float(dmi_minus_di_val)),
                    "dmi_dx": (None if pd.isna(dmi_dx_val) else float(dmi_dx_val)),
                    "wallet": state.wallet,
                    "wallet_free": wallet_free,
                    "margin_invested": state.margin_invested,
                    "margin_pct_wallet": (state.margin_invested / state.wallet) if state.wallet > 0 else 0.0,
                    "equity": equity,
                    "margin_total": margin_total,
                    "loss_unrealized": loss_unrealized,
                    "liq_limit": liq_limit,
                    "notional_position": notional,
                    "lev_eff_wallet": (notional / state.wallet) if state.wallet > 0 else 0.0,
                    "lev_eff_capital": (notional / float(cfg.capital_usdt)) if float(cfg.capital_usdt) > 0 else 0.0,
                    "lev_eff_used_margin": (notional / state.margin_invested) if state.margin_invested > 0 else 0.0,
                    "position_size": state.position_size,
                    "avg_price": state.avg_price,
                    "pnl_unrealized": pnl_unrealized,
                    "portions_used": state.portions_used,
                    "cycles_completed": state.cycles_completed,
                    "current_d_index": state.current_d_index,
                    "next_target_price": state.next_target_price,
                    "is_liquidated": bool(state.is_liquidated),
                    "liquidation_reason": state.liquidation_reason if state.is_liquidated else "",
                }
            )
            break

        # --- Gestion TP ---
        tp_mode = str(cfg.tp_mode).strip().lower()
        if tp_mode == "tp_full":
            if state.position_size > 0 and state.avg_price > 0:
                if is_long:
                    tp_price = state.avg_price * (1.0 + tp_frac)
                    should_tp = price >= tp_price
                else:
                    tp_price = state.avg_price * (1.0 - tp_frac)
                    should_tp = price <= tp_price

                if should_tp:
                    notional = state.position_size * price
                    fee = notional * cfg.fee_rate
                    if is_long:
                        pnl_realized = (price - state.avg_price) * state.position_size
                    else:
                        pnl_realized = (state.avg_price - price) * state.position_size

                    state.wallet += pnl_realized
                    state.wallet -= fee

                    trades.append(
                        {
                            "timestamp": ts,
                            "price": price,
                            "macd_hist": (None if pd.isna(macd_hist_val) else float(macd_hist_val)),
                            "macd_hist_sign": int(hist_sign),
                            "macd_tranche_sign": int(current_tranche_sign),
                            "macd_prev_tranche_sign": int(last_tranche_sign),
                            "macd_flip_to_side": bool(macd_flip_to_side),
                            "macd_prev_tranche_ok": (None if prev_tranche_ok_dbg is None else bool(prev_tranche_ok_dbg)),
                            "macd_prev_tranche_seen_cci": bool(last_tranche_seen_cci),
                            "macd_prev_tranche_seen_cci_medium": bool(last_tranche_seen_cci_medium),
                            "macd_prev_tranche_seen_cci_slow": bool(last_tranche_seen_cci_slow),
                            "macd_prev_tranche_seen_mfi": bool(last_tranche_seen_mfi),
                            "macd_prev_tranche_seen_dmi": bool(last_tranche_seen_dmi),
                            "cci12": (None if pd.isna(cci12_val) else float(cci12_val)),
                            "mfi": (None if pd.isna(mfi_val) else float(mfi_val)),
                            "macd_prev_tranche_cci": (
                                None if pd.isna(macd_prev_tranche_cci_val) else float(macd_prev_tranche_cci_val)
                            ),
                            "macd_prev_tranche_cci_medium": (
                                None
                                if pd.isna(macd_prev_tranche_cci_medium_val)
                                else float(macd_prev_tranche_cci_medium_val)
                            ),
                            "macd_prev_tranche_cci_slow": (
                                None if pd.isna(macd_prev_tranche_cci_slow_val) else float(macd_prev_tranche_cci_slow_val)
                            ),
                            "macd_prev_tranche_mfi": (
                                None if pd.isna(macd_prev_tranche_mfi_val) else float(macd_prev_tranche_mfi_val)
                            ),
                            "dmi_adx": (None if pd.isna(dmi_adx_val) else float(dmi_adx_val)),
                            "dmi_plus_di": (None if pd.isna(dmi_plus_di_val) else float(dmi_plus_di_val)),
                            "dmi_minus_di": (None if pd.isna(dmi_minus_di_val) else float(dmi_minus_di_val)),
                            "dmi_dx": (None if pd.isna(dmi_dx_val) else float(dmi_dx_val)),
                            "side": side_label,
                            "type": "TP_FULL",
                            "qty": state.position_size,
                            "qty_usdt": notional,
                            "pnl_realized": pnl_realized,
                            "fee": fee,
                        }
                    )

                    state.position_size = 0.0
                    state.avg_price = 0.0
                    state.current_d_index = 0
                    state.next_target_price = 0.0
                    state.margin_invested = 0.0

                    state.cycles_completed += 1
                    state.last_exit_ts = ts
                    state.portions_used = 0

        elif tp_mode == "tp_cycles":
            entry_signal_raw = True
            if bool(cfg.macd_hist_flip_enabled):
                prev_ok = True
                if need_prev_tranche_entry:
                    prev_ok = bool(prev_tranche_ok_dbg) if prev_tranche_ok_dbg is not None else False
                entry_signal_raw = bool(macd_flip_to_side and prev_ok)

            macd_flip_away_from_side = False
            if prev_hist_sign_before != 0 and hist_sign != 0 and hist_sign != prev_hist_sign_before:
                if is_long:
                    macd_flip_away_from_side = (prev_hist_sign_before == 1 and hist_sign == -1)
                else:
                    macd_flip_away_from_side = (prev_hist_sign_before == -1 and hist_sign == 1)

            tp_partial_signal_raw = bool(cfg.tp_partial_macd_hist_flip_enabled and macd_flip_away_from_side)
            if need_prev_tranche_tp_partial:
                tp_ok = bool(tp_prev_tranche_ok_dbg) if tp_prev_tranche_ok_dbg is not None else False
                tp_partial_signal_raw = bool(tp_partial_signal_raw and tp_ok)

            entry_signal = bool(entry_signal_raw and (not state.prev_entry_signal_raw))
            tp_partial_signal = bool(tp_partial_signal_raw and (not state.prev_tp_partial_signal_raw))
            state.prev_entry_signal_raw = bool(entry_signal_raw)
            state.prev_tp_partial_signal_raw = bool(tp_partial_signal_raw)

            # Track tp_reached for all bucket cycles (and active too)
            for c in state.tp_bucket:
                if c.next_tp_price > 0:
                    if is_long:
                        if price >= c.next_tp_price:
                            c.tp_reached = True
                    else:
                        if price <= c.next_tp_price:
                            c.tp_reached = True
            if state.tp_active is not None and state.tp_active.next_tp_price > 0:
                c = state.tp_active
                if is_long:
                    if price >= c.next_tp_price:
                        c.tp_reached = True
                else:
                    if price <= c.next_tp_price:
                        c.tp_reached = True

            # Enter TP phase on TP partial signal (and freeze openings)
            if tp_partial_signal:
                state.tp_phase = "TP"
                if state.tp_active is not None and state.tp_active.size > 0:
                    state.tp_bucket.append(state.tp_active)
                    state.tp_active = None

            # Execute partial TP on signal (bucket cycles, LIFO)
            if state.tp_phase == "TP" and tp_partial_signal and state.tp_bucket:
                close_ratio = float(cfg.tp_close_ratio)
                if close_ratio <= 0:
                    close_ratio = 0.0
                if close_ratio > 1.0:
                    close_ratio = 1.0

                new_bucket: List[MiniCycle] = []
                for c in reversed(state.tp_bucket):
                    eligible = bool(c.next_tp_price > 0 and (c.tp_reached or ((price >= c.next_tp_price) if is_long else (price <= c.next_tp_price))))
                    if (not eligible) or c.size <= 0 or close_ratio <= 0:
                        new_bucket.append(c)
                        continue

                    close_qty = float(c.size) * close_ratio
                    if close_qty <= 0:
                        new_bucket.append(c)
                        continue

                    notional = close_qty * price
                    fee = notional * cfg.fee_rate
                    if is_long:
                        pnl_realized = (price - c.avg_open) * close_qty
                    else:
                        pnl_realized = (c.avg_open - price) * close_qty

                    state.wallet += pnl_realized
                    state.wallet -= fee

                    margin_release = notional / float(cfg.leverage) if float(cfg.leverage) > 0 else 0.0
                    state.margin_invested = max(0.0, float(state.margin_invested) - float(margin_release))

                    if c.closed_qty <= 0:
                        c.avg_close = float(price)
                        c.closed_qty = float(close_qty)
                    else:
                        new_closed_qty = float(c.closed_qty) + float(close_qty)
                        c.avg_close = (float(c.avg_close) * float(c.closed_qty) + float(price) * float(close_qty)) / new_closed_qty
                        c.closed_qty = new_closed_qty

                    c.size = max(0.0, float(c.size) - float(close_qty))
                    c.tp_index += 1
                    c.tp_reached = False
                    _cycle_recompute_next_tp_price(
                        c,
                        is_long=is_long,
                        tp_d_start_pct=float(cfg.tp_d_start_pct),
                        tp_d_step_pct=float(cfg.tp_d_step_pct),
                    )

                    trades.append(
                        {
                            "timestamp": ts,
                            "price": price,
                            "macd_hist": (None if pd.isna(macd_hist_val) else float(macd_hist_val)),
                            "macd_hist_sign": int(hist_sign),
                            "macd_tranche_sign": int(current_tranche_sign),
                            "macd_prev_tranche_sign": int(last_tranche_sign),
                            "macd_flip_to_side": bool(macd_flip_to_side),
                            "macd_prev_tranche_ok": (None if prev_tranche_ok_dbg is None else bool(prev_tranche_ok_dbg)),
                            "side": side_label,
                            "type": "TP_PARTIAL",
                            "cycle_id": int(c.cycle_id),
                            "qty": float(close_qty),
                            "qty_usdt": float(notional),
                            "pnl_realized": float(pnl_realized),
                            "fee": float(fee),
                        }
                    )

                    if c.size <= 0:
                        state.cycles_completed += 1
                        state.last_exit_ts = ts
                    else:
                        new_bucket.append(c)

                state.tp_bucket = list(reversed(new_bucket))

            # keep portions_used roughly in sync for reporting
            used_portions_f = (state.margin_invested / portion_margin) if portion_margin > 0 else 0.0
            state.portions_used = int(max(0, round(used_portions_f)))

        # --- Base order si aucune position et aucune portion utilisée ---
        can_start_new_cycle = True
        if cfg.max_cycles and state.cycles_completed >= int(cfg.max_cycles):
            can_start_new_cycle = False
        if cfg.reentry_cooldown_minutes and state.last_exit_ts:
            cooldown_ms = int(cfg.reentry_cooldown_minutes) * 60_000
            if (ts - int(state.last_exit_ts)) < cooldown_ms:
                can_start_new_cycle = False

        if tp_mode == "tp_full":
            if state.position_size <= 0 and state.portions_used == 0 and can_start_new_cycle:
                # On ouvre une position long de taille q_notional au prix courant
                qty = q_notional / price if price > 0 else 0.0
                if qty > 0:
                    fee = q_notional * cfg.fee_rate

                    wallet_free = state.wallet - state.margin_invested
                    required_free = q_margin + fee
                    if wallet_free < required_free:
                        qty = 0.0

                if qty > 0:
                    new_size = state.position_size + qty
                    new_avg = (state.position_size * state.avg_price + q_notional) / new_size

                    state.position_size = new_size
                    state.avg_price = new_avg
                    state.wallet -= fee
                    state.margin_invested += q_margin
                    state.portions_used = 1

                    trades.append(
                        {
                            "timestamp": ts,
                            "price": price,
                            "macd_hist": (None if pd.isna(macd_hist_val) else float(macd_hist_val)),
                            "macd_hist_sign": int(hist_sign),
                            "macd_tranche_sign": int(current_tranche_sign),
                            "macd_prev_tranche_sign": int(last_tranche_sign),
                            "macd_flip_to_side": bool(macd_flip_to_side),
                            "macd_prev_tranche_ok": (
                                None if prev_tranche_ok_dbg is None else bool(prev_tranche_ok_dbg)
                            ),
                            "macd_prev_tranche_seen_cci": bool(last_tranche_seen_cci),
                            "macd_prev_tranche_seen_cci_medium": bool(last_tranche_seen_cci_medium),
                            "macd_prev_tranche_seen_cci_slow": bool(last_tranche_seen_cci_slow),
                            "macd_prev_tranche_seen_mfi": bool(last_tranche_seen_mfi),
                            "macd_prev_tranche_seen_dmi": bool(last_tranche_seen_dmi),
                            "cci12": (None if pd.isna(cci12_val) else float(cci12_val)),
                            "mfi": (None if pd.isna(mfi_val) else float(mfi_val)),
                            "macd_prev_tranche_cci": (
                                None if pd.isna(macd_prev_tranche_cci_val) else float(macd_prev_tranche_cci_val)
                            ),
                            "macd_prev_tranche_cci_medium": (
                                None
                                if pd.isna(macd_prev_tranche_cci_medium_val)
                                else float(macd_prev_tranche_cci_medium_val)
                            ),
                            "macd_prev_tranche_cci_slow": (
                                None
                                if pd.isna(macd_prev_tranche_cci_slow_val)
                                else float(macd_prev_tranche_cci_slow_val)
                            ),
                            "macd_prev_tranche_mfi": (
                                None if pd.isna(macd_prev_tranche_mfi_val) else float(macd_prev_tranche_mfi_val)
                            ),
                            "dmi_adx": (None if pd.isna(dmi_adx_val) else float(dmi_adx_val)),
                            "dmi_plus_di": (None if pd.isna(dmi_plus_di_val) else float(dmi_plus_di_val)),
                            "dmi_minus_di": (None if pd.isna(dmi_minus_di_val) else float(dmi_minus_di_val)),
                            "dmi_dx": (None if pd.isna(dmi_dx_val) else float(dmi_dx_val)),
                            "side": side_label,
                            "type": "BASE",
                            "qty": qty,
                            "qty_usdt": q_notional,
                            "pnl_realized": 0.0,
                            "fee": fee,
                            "margin_cost": q_margin,
                        }
                    )

                    state.current_d_index = 1
                    d = d_start
                    if is_long:
                        p_new = compute_p_new(state.position_size, state.avg_price, q_notional, d)
                    else:
                        p_new = compute_p_new_short(state.position_size, state.avg_price, q_notional, d)
                    state.next_target_price = p_new

        if tp_mode == "tp_cycles":
            used_portions_f = (state.margin_invested / portion_margin) if portion_margin > 0 else 0.0
            has_active = bool(state.tp_active is not None and state.tp_active.size > 0)

            # Exit TP phase only via entry signal + BASE
            if state.tp_phase == "TP":
                if entry_signal and can_start_new_cycle and (used_portions_f + 1.0) <= float(cfg.max_portions):
                    allow_base = True
                    if state.tp_bucket:
                        n = float(cfg.tp_new_cycle_min_distance_pct)
                        if n > 0:
                            if is_long:
                                ref = min(float(c.avg_open) for c in state.tp_bucket if c.avg_open > 0)
                                allow_base = price <= ref * (1.0 - n / 100.0)
                            else:
                                ref = max(float(c.avg_open) for c in state.tp_bucket if c.avg_open > 0)
                                allow_base = price >= ref * (1.0 + n / 100.0)
                    if allow_base and price > 0:
                        qty = q_notional / price
                        fee = q_notional * cfg.fee_rate
                        wallet_free = state.wallet - state.margin_invested
                        required_free = q_margin + fee
                        if wallet_free >= required_free and qty > 0:
                            cycle_id = int(state.tp_next_cycle_id)
                            state.tp_next_cycle_id += 1
                            c = MiniCycle(cycle_id=cycle_id, created_ts=int(ts), size=float(qty), avg_open=float(price))
                            c.current_d_index = 1
                            d = d_start
                            if is_long:
                                c.next_target_price = compute_p_new(c.size, c.avg_open, q_notional, d)
                            else:
                                c.next_target_price = compute_p_new_short(c.size, c.avg_open, q_notional, d)
                            _cycle_recompute_next_tp_price(
                                c,
                                is_long=is_long,
                                tp_d_start_pct=float(cfg.tp_d_start_pct),
                                tp_d_step_pct=float(cfg.tp_d_step_pct),
                            )

                            state.tp_active = c
                            state.wallet -= fee
                            state.margin_invested += q_margin
                            state.tp_phase = "OPEN"

                            trades.append(
                                {
                                    "timestamp": ts,
                                    "price": price,
                                    "macd_hist": (None if pd.isna(macd_hist_val) else float(macd_hist_val)),
                                    "macd_hist_sign": int(hist_sign),
                                    "macd_tranche_sign": int(current_tranche_sign),
                                    "macd_prev_tranche_sign": int(last_tranche_sign),
                                    "macd_flip_to_side": bool(macd_flip_to_side),
                                    "macd_prev_tranche_ok": (
                                        None if prev_tranche_ok_dbg is None else bool(prev_tranche_ok_dbg)
                                    ),
                                    "side": side_label,
                                    "type": "BASE",
                                    "cycle_id": cycle_id,
                                    "qty": float(qty),
                                    "qty_usdt": float(q_notional),
                                    "pnl_realized": 0.0,
                                    "fee": float(fee),
                                    "margin_cost": float(q_margin),
                                }
                            )

            # Start/open BASE if no active cycle and OPEN phase
            if (state.tp_phase == "OPEN") and (not has_active) and can_start_new_cycle and (used_portions_f + 1.0) <= float(cfg.max_portions):
                allow_base = True
                if state.tp_bucket:
                    allow_base = bool(entry_signal)
                    n = float(cfg.tp_new_cycle_min_distance_pct)
                    if allow_base and n > 0:
                        if is_long:
                            ref = min(float(c.avg_open) for c in state.tp_bucket if c.avg_open > 0)
                            allow_base = price <= ref * (1.0 - n / 100.0)
                        else:
                            ref = max(float(c.avg_open) for c in state.tp_bucket if c.avg_open > 0)
                            allow_base = price >= ref * (1.0 + n / 100.0)

                if allow_base and price > 0:
                    qty = q_notional / price
                    fee = q_notional * cfg.fee_rate
                    wallet_free = state.wallet - state.margin_invested
                    required_free = q_margin + fee
                    if wallet_free >= required_free and qty > 0:
                        cycle_id = int(state.tp_next_cycle_id)
                        state.tp_next_cycle_id += 1
                        c = MiniCycle(cycle_id=cycle_id, created_ts=int(ts), size=float(qty), avg_open=float(price))
                        c.current_d_index = 1
                        d = d_start
                        if is_long:
                            c.next_target_price = compute_p_new(c.size, c.avg_open, q_notional, d)
                        else:
                            c.next_target_price = compute_p_new_short(c.size, c.avg_open, q_notional, d)
                        _cycle_recompute_next_tp_price(
                            c,
                            is_long=is_long,
                            tp_d_start_pct=float(cfg.tp_d_start_pct),
                            tp_d_step_pct=float(cfg.tp_d_step_pct),
                        )

                        state.tp_active = c
                        state.wallet -= fee
                        state.margin_invested += q_margin

                        trades.append(
                            {
                                "timestamp": ts,
                                "price": price,
                                "macd_hist": (None if pd.isna(macd_hist_val) else float(macd_hist_val)),
                                "macd_hist_sign": int(hist_sign),
                                "macd_tranche_sign": int(current_tranche_sign),
                                "macd_prev_tranche_sign": int(last_tranche_sign),
                                "macd_flip_to_side": bool(macd_flip_to_side),
                                "macd_prev_tranche_ok": (None if prev_tranche_ok_dbg is None else bool(prev_tranche_ok_dbg)),
                                "side": side_label,
                                "type": "BASE",
                                "cycle_id": cycle_id,
                                "qty": float(qty),
                                "qty_usdt": float(q_notional),
                                "pnl_realized": 0.0,
                                "fee": float(fee),
                                "margin_cost": float(q_margin),
                            }
                        )

        # --- Sécurité DCA ---
        if tp_mode == "tp_full":
            if state.position_size > 0 and state.portions_used < cfg.max_portions:
                if state.current_d_index > 0 and state.next_target_price > 0.0:
                    # Long: sécurité plus bas. Short: sécurité plus haut.
                    should_dca = (price <= state.next_target_price) if is_long else (price >= state.next_target_price)
                    if should_dca:
                        run_stats["dca_price_triggers"] += 1
                    cci_ok = True
                    if bool(cfg.cci12_enabled):
                        if pd.isna(cci12_val):
                            cci_ok = False
                        else:
                            cci_f = float(cci12_val)
                            if is_long:
                                cci_ok = cci_f <= (-abs(float(cfg.cci_long_threshold)))
                            else:
                                cci_ok = cci_f >= (abs(float(cfg.cci_short_threshold)))

                    mfi_ok = True
                    if bool(cfg.mfi_enabled):
                        if pd.isna(mfi_val):
                            mfi_ok = False
                        else:
                            mfi_f = float(mfi_val)
                            if is_long:
                                mfi_ok = mfi_f <= float(cfg.mfi_long_threshold)
                            else:
                                mfi_ok = mfi_f >= float(cfg.mfi_short_threshold)

                    if should_dca and (not cci_ok):
                        run_stats["dca_blocked_indicator"] += 1
                    if should_dca and (not mfi_ok):
                        run_stats["dca_blocked_mfi"] += 1

                    macd_ok = True
                    if bool(cfg.macd_hist_flip_enabled):
                        prev_tranche_ok = True
                        prev_tranche_cci_ok = True
                        prev_tranche_cci_fast_ok = True
                        prev_tranche_cci_medium_ok = True
                        prev_tranche_cci_slow_ok = True
                        prev_tranche_mfi_ok = True
                        prev_tranche_dmi_ok = True
                        if need_prev_tranche_entry:
                            expected_prev_sign = -1 if is_long else 1
                            prev_sign_ok = (last_tranche_sign == expected_prev_sign)
                            if bool(cfg.macd_prev_tranche_cci_enabled):
                                prev_tranche_cci_fast_ok = bool(last_tranche_seen_cci)
                            if bool(cfg.macd_prev_tranche_cci_medium_enabled):
                                prev_tranche_cci_medium_ok = bool(last_tranche_seen_cci_medium)
                            if bool(cfg.macd_prev_tranche_cci_slow_enabled):
                                prev_tranche_cci_slow_ok = bool(last_tranche_seen_cci_slow)
                            prev_tranche_cci_ok = bool(
                                prev_tranche_cci_fast_ok and prev_tranche_cci_medium_ok and prev_tranche_cci_slow_ok
                            )
                            prev_tranche_mfi_ok = (not bool(cfg.macd_prev_tranche_mfi_enabled)) or bool(last_tranche_seen_mfi)
                            prev_tranche_dmi_ok = (not bool(cfg.macd_prev_tranche_dmi_enabled)) or bool(last_tranche_seen_dmi)
                            prev_tranche_ok = bool(
                                prev_sign_ok and prev_tranche_cci_ok and prev_tranche_mfi_ok and prev_tranche_dmi_ok
                            )

                        macd_ok = bool(macd_flip_to_side and prev_tranche_ok)
                    if should_dca and cci_ok and mfi_ok and (not macd_ok):
                        run_stats["dca_blocked_macd"] += 1
                        if need_prev_tranche_entry and bool(macd_flip_to_side):
                            if not prev_tranche_ok:
                                run_stats["dca_blocked_macd_prev_tranche"] += 1
                                if bool(cfg.macd_prev_tranche_cci_enabled) and (not prev_tranche_cci_fast_ok):
                                    run_stats["dca_blocked_macd_prev_cci"] += 1
                                if bool(cfg.macd_prev_tranche_cci_medium_enabled) and (not prev_tranche_cci_medium_ok):
                                    run_stats["dca_blocked_macd_prev_cci_medium"] += 1
                                if bool(cfg.macd_prev_tranche_cci_slow_enabled) and (not prev_tranche_cci_slow_ok):
                                    run_stats["dca_blocked_macd_prev_cci_slow"] += 1
                                if bool(cfg.macd_prev_tranche_mfi_enabled) and (not prev_tranche_mfi_ok):
                                    run_stats["dca_blocked_macd_prev_mfi"] += 1
                                if bool(cfg.macd_prev_tranche_dmi_enabled) and (not prev_tranche_dmi_ok):
                                    run_stats["dca_blocked_macd_prev_dmi"] += 1

                    if should_dca and cci_ok and mfi_ok and macd_ok:
                        # Exécuter une sécurité à notional fixe q_notional
                        qty = q_notional / price if price > 0 else 0.0
                        if qty > 0:
                            fee = q_notional * cfg.fee_rate

                            wallet_free = state.wallet - state.margin_invested
                            required_free = q_margin + fee
                            if wallet_free < required_free:
                                run_stats["dca_blocked_margin"] += 1
                                qty = 0.0

                        if qty > 0:
                            new_size = state.position_size + qty
                            new_avg = (state.position_size * state.avg_price + q_notional) / new_size

                            state.position_size = new_size
                            state.avg_price = new_avg
                            state.wallet -= fee
                            state.margin_invested += q_margin
                            state.portions_used += 1

                            trades.append(
                                {
                                    "timestamp": ts,
                                    "price": price,
                                    "macd_hist": (None if pd.isna(macd_hist_val) else float(macd_hist_val)),
                                    "macd_hist_sign": int(hist_sign),
                                    "macd_tranche_sign": int(current_tranche_sign),
                                    "macd_prev_tranche_sign": int(last_tranche_sign),
                                    "macd_flip_to_side": bool(macd_flip_to_side),
                                    "macd_prev_tranche_ok": (
                                        None if prev_tranche_ok_dbg is None else bool(prev_tranche_ok_dbg)
                                    ),
                                    "macd_prev_tranche_seen_cci": bool(last_tranche_seen_cci),
                                    "macd_prev_tranche_seen_cci_medium": bool(last_tranche_seen_cci_medium),
                                    "macd_prev_tranche_seen_cci_slow": bool(last_tranche_seen_cci_slow),
                                    "macd_prev_tranche_seen_mfi": bool(last_tranche_seen_mfi),
                                    "macd_prev_tranche_seen_dmi": bool(last_tranche_seen_dmi),
                                    "cci12": (None if pd.isna(cci12_val) else float(cci12_val)),
                                    "mfi": (None if pd.isna(mfi_val) else float(mfi_val)),
                                    "macd_prev_tranche_cci": (
                                        None
                                        if pd.isna(macd_prev_tranche_cci_val)
                                        else float(macd_prev_tranche_cci_val)
                                    ),
                                    "macd_prev_tranche_cci_medium": (
                                        None
                                        if pd.isna(macd_prev_tranche_cci_medium_val)
                                        else float(macd_prev_tranche_cci_medium_val)
                                    ),
                                    "macd_prev_tranche_cci_slow": (
                                        None
                                        if pd.isna(macd_prev_tranche_cci_slow_val)
                                        else float(macd_prev_tranche_cci_slow_val)
                                    ),
                                    "macd_prev_tranche_mfi": (
                                        None
                                        if pd.isna(macd_prev_tranche_mfi_val)
                                        else float(macd_prev_tranche_mfi_val)
                                    ),
                                    "dmi_adx": (None if pd.isna(dmi_adx_val) else float(dmi_adx_val)),
                                    "dmi_plus_di": (None if pd.isna(dmi_plus_di_val) else float(dmi_plus_di_val)),
                                    "dmi_minus_di": (None if pd.isna(dmi_minus_di_val) else float(dmi_minus_di_val)),
                                    "dmi_dx": (None if pd.isna(dmi_dx_val) else float(dmi_dx_val)),
                                    "side": side_label,
                                    "type": "DCA",
                                    "qty": qty,
                                    "qty_usdt": q_notional,
                                    "pnl_realized": 0.0,
                                    "fee": fee,
                                    "margin_cost": q_margin,
                                }
                            )
                            run_stats["dca_executed"] += 1

                            state.current_d_index += 1
                            d = d_start + (state.current_d_index - 1) * d_step
                            if is_long:
                                p_new = compute_p_new(state.position_size, state.avg_price, q_notional, d)
                            else:
                                p_new = compute_p_new_short(state.position_size, state.avg_price, q_notional, d)
                            state.next_target_price = p_new

        if tp_mode == "tp_cycles":
            used_portions_f = (state.margin_invested / portion_margin) if portion_margin > 0 else 0.0
            if (
                state.tp_phase == "OPEN"
                and state.tp_active is not None
                and state.tp_active.size > 0
                and used_portions_f < float(cfg.max_portions)
            ):
                c = state.tp_active
                if c.current_d_index > 0 and c.next_target_price > 0.0:
                    should_dca = (price <= c.next_target_price) if is_long else (price >= c.next_target_price)
                    if should_dca:
                        run_stats["dca_price_triggers"] += 1

                    cci_ok = True
                    if bool(cfg.cci12_enabled):
                        if pd.isna(cci12_val):
                            cci_ok = False
                        else:
                            cci_f = float(cci12_val)
                            if is_long:
                                cci_ok = cci_f <= (-abs(float(cfg.cci_long_threshold)))
                            else:
                                cci_ok = cci_f >= (abs(float(cfg.cci_short_threshold)))

                    mfi_ok = True
                    if bool(cfg.mfi_enabled):
                        if pd.isna(mfi_val):
                            mfi_ok = False
                        else:
                            mfi_f = float(mfi_val)
                            if is_long:
                                mfi_ok = mfi_f <= float(cfg.mfi_long_threshold)
                            else:
                                mfi_ok = mfi_f >= float(cfg.mfi_short_threshold)

                    if should_dca and (not cci_ok):
                        run_stats["dca_blocked_indicator"] += 1
                    if should_dca and (not mfi_ok):
                        run_stats["dca_blocked_mfi"] += 1

                    macd_ok = True
                    if bool(cfg.macd_hist_flip_enabled):
                        prev_ok = True
                        if need_prev_tranche_entry:
                            prev_ok = bool(prev_tranche_ok_dbg) if prev_tranche_ok_dbg is not None else False
                        macd_ok = bool(macd_flip_to_side and prev_ok)

                    if should_dca and cci_ok and mfi_ok and (not macd_ok):
                        run_stats["dca_blocked_macd"] += 1

                    if should_dca and cci_ok and mfi_ok and macd_ok:
                        qty = q_notional / price if price > 0 else 0.0
                        if qty > 0:
                            fee = q_notional * cfg.fee_rate
                            wallet_free = state.wallet - state.margin_invested
                            required_free = q_margin + fee
                            if wallet_free < required_free:
                                run_stats["dca_blocked_margin"] += 1
                                qty = 0.0

                        if qty > 0:
                            new_size = c.size + qty
                            new_avg = (c.size * c.avg_open + q_notional) / new_size
                            c.size = new_size
                            c.avg_open = new_avg
                            state.wallet -= fee
                            state.margin_invested += q_margin

                            trades.append(
                                {
                                    "timestamp": ts,
                                    "price": price,
                                    "macd_hist": (None if pd.isna(macd_hist_val) else float(macd_hist_val)),
                                    "macd_hist_sign": int(hist_sign),
                                    "macd_tranche_sign": int(current_tranche_sign),
                                    "macd_prev_tranche_sign": int(last_tranche_sign),
                                    "macd_flip_to_side": bool(macd_flip_to_side),
                                    "macd_prev_tranche_ok": (None if prev_tranche_ok_dbg is None else bool(prev_tranche_ok_dbg)),
                                    "side": side_label,
                                    "type": "DCA",
                                    "cycle_id": int(c.cycle_id),
                                    "qty": float(qty),
                                    "qty_usdt": float(q_notional),
                                    "pnl_realized": 0.0,
                                    "fee": float(fee),
                                    "margin_cost": float(q_margin),
                                }
                            )
                            run_stats["dca_executed"] += 1

                            c.current_d_index += 1
                            d = d_start + (c.current_d_index - 1) * d_step
                            if is_long:
                                c.next_target_price = compute_p_new(c.size, c.avg_open, q_notional, d)
                            else:
                                c.next_target_price = compute_p_new_short(c.size, c.avg_open, q_notional, d)

            _sync_global_position_from_cycles(state)

        # Recompute unrealized and equity after possible actions
        if state.position_size > 0 and state.avg_price > 0:
            if is_long:
                pnl_unrealized = (price - state.avg_price) * state.position_size
            else:
                pnl_unrealized = (state.avg_price - price) * state.position_size
        else:
            pnl_unrealized = 0.0
        equity = state.wallet + pnl_unrealized

        wallet_free = state.wallet - state.margin_invested
        pnl_total = pnl_unrealized
        margin_total = state.wallet + max(0.0, pnl_total)
        loss_unrealized = max(0.0, -pnl_total)
        liq_limit = (1.0 - float(cfg.liquidation_threshold_pct)) * margin_total

        notional_position = state.position_size * price
        lev_eff_wallet = (notional_position / state.wallet) if state.wallet > 0 else 0.0
        lev_eff_capital = (notional_position / float(cfg.capital_usdt)) if float(cfg.capital_usdt) > 0 else 0.0
        margin_pct_wallet = (state.margin_invested / state.wallet) if state.wallet > 0 else 0.0
        lev_eff_used_margin = (notional_position / state.margin_invested) if state.margin_invested > 0 else 0.0

        equity_rows.append(
            {
                "timestamp": ts,
                "price": price,
                "macd_hist": (None if pd.isna(macd_hist_val) else float(macd_hist_val)),
                "macd_hist_sign": int(hist_sign),
                "macd_tranche_sign": int(current_tranche_sign),
                "macd_prev_tranche_sign": int(last_tranche_sign),
                "macd_flip_to_side": bool(macd_flip_to_side),
                "macd_prev_tranche_ok": (None if prev_tranche_ok_dbg is None else bool(prev_tranche_ok_dbg)),
                "macd_prev_tranche_seen_cci": bool(last_tranche_seen_cci),
                "macd_prev_tranche_seen_cci_medium": bool(last_tranche_seen_cci_medium),
                "macd_prev_tranche_seen_cci_slow": bool(last_tranche_seen_cci_slow),
                "macd_prev_tranche_seen_mfi": bool(last_tranche_seen_mfi),
                "macd_prev_tranche_seen_dmi": bool(last_tranche_seen_dmi),
                "cci12": (None if pd.isna(cci12_val) else float(cci12_val)),
                "mfi": (None if pd.isna(mfi_val) else float(mfi_val)),
                "macd_prev_tranche_cci": (None if pd.isna(macd_prev_tranche_cci_val) else float(macd_prev_tranche_cci_val)),
                "macd_prev_tranche_cci_medium": (None if pd.isna(macd_prev_tranche_cci_medium_val) else float(macd_prev_tranche_cci_medium_val)),
                "macd_prev_tranche_cci_slow": (None if pd.isna(macd_prev_tranche_cci_slow_val) else float(macd_prev_tranche_cci_slow_val)),
                "macd_prev_tranche_mfi": (None if pd.isna(macd_prev_tranche_mfi_val) else float(macd_prev_tranche_mfi_val)),
                "dmi_adx": (None if pd.isna(dmi_adx_val) else float(dmi_adx_val)),
                "dmi_plus_di": (None if pd.isna(dmi_plus_di_val) else float(dmi_plus_di_val)),
                "dmi_minus_di": (None if pd.isna(dmi_minus_di_val) else float(dmi_minus_di_val)),
                "dmi_dx": (None if pd.isna(dmi_dx_val) else float(dmi_dx_val)),
                "wallet": state.wallet,
                "wallet_free": wallet_free,
                "margin_invested": state.margin_invested,
                "margin_pct_wallet": margin_pct_wallet,
                "equity": equity,
                "margin_total": margin_total,
                "loss_unrealized": loss_unrealized,
                "liq_limit": liq_limit,
                "notional_position": notional_position,
                "lev_eff_wallet": lev_eff_wallet,
                "lev_eff_capital": lev_eff_capital,
                "lev_eff_used_margin": lev_eff_used_margin,
                "position_size": state.position_size,
                "avg_price": state.avg_price,
                "pnl_unrealized": pnl_unrealized,
                "portions_used": state.portions_used,
                "cycles_completed": state.cycles_completed,
                "current_d_index": state.current_d_index,
                "next_target_price": state.next_target_price,
                "is_liquidated": bool(state.is_liquidated),
                "liquidation_reason": state.liquidation_reason if state.is_liquidated else "",
            }
        )

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trades)

    return {"equity": equity_df, "trades": trades_df, "stats": run_stats}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "backtest_one_side_dca_mean_distance.yaml"),
        help="Path to YAML config for one-side DCA mean-distance backtest",
    )
    args = ap.parse_args()

    cfg = load_core_config(str(args.config))
    res = run_backtest(cfg)
    if not res:
        return

    equity_df: pd.DataFrame = res["equity"]
    trades_df: pd.DataFrame = res["trades"]
    stats: Dict[str, Any] = res.get("stats", {}) if isinstance(res, dict) else {}

    output_dir = (PROJECT_ROOT / str(cfg.output_dir)).resolve() if not str(cfg.output_dir).startswith("/") else Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = output_dir / f"one_side_dca_{cfg.symbol}_{cfg.timeframe}_{cfg.start_date}_{cfg.end_date}"

    eq_path = f"{prefix}_equity.csv"
    tr_path = f"{prefix}_trades.csv"

    equity_df.to_csv(eq_path, index=False)
    logging.info(f"Saved equity to {eq_path}")

    if not trades_df.empty:
        trades_df.to_csv(tr_path, index=False)
        logging.info(f"Saved trades to {tr_path}")

    # Plot equity evolution
    if bool(cfg.png):
        try:
            if not equity_df.empty:
                ts = pd.to_datetime(equity_df["timestamp"], unit="ms")
                plt.figure(figsize=(12, 6))
                plt.plot(ts, equity_df["equity"], label="Equity", color="blue")
                plt.plot(ts, equity_df["wallet"], label="Wallet", color="orange", linestyle="--")
                plt.title(f"One-Side DCA Equity - {cfg.symbol} {cfg.timeframe} {cfg.start_date} to {cfg.end_date}")
                plt.xlabel("Time")
                plt.ylabel("USDT")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                png_path = f"{prefix}_equity.png"
                plt.savefig(png_path)
                plt.close()
                logging.info(f"Saved equity plot to {png_path}")
        except Exception as e:
            logging.error(f"Failed to plot equity: {e}")

    # Petit résumé
    if not equity_df.empty:
        initial_cap = cfg.capital_usdt
        final_equity = float(equity_df.iloc[-1]["equity"])
        pnl_net = final_equity - initial_cap
        roi_pct = (pnl_net / initial_cap) * 100.0 if initial_cap > 0 else 0.0

        equity_series = equity_df["equity"]
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max
        max_dd_pct = drawdown.min() * 100.0

        idx_min_eq = int(equity_series.idxmin()) if not equity_series.empty else 0
        idx_max_dd = int(drawdown.idxmin()) if not drawdown.empty else 0
        ts_min_eq = None
        ts_max_dd = None
        try:
            ts_min_eq = int(equity_df.iloc[idx_min_eq]["timestamp"])
        except Exception:
            ts_min_eq = None
        try:
            ts_max_dd = int(equity_df.iloc[idx_max_dd]["timestamp"])
        except Exception:
            ts_max_dd = None

        print("\n" + "=" * 40)
        print("   ONE-SIDE DCA MEAN-DIST BACKTEST   ")
        print("=" * 40)
        print(f"Symbol          : {cfg.symbol}")
        print(f"Period          : {cfg.start_date} to {cfg.end_date}")
        print(f"Timeframe       : {cfg.timeframe}")
        print(f"Capital         : {initial_cap:.2f} USDT")
        print(f"Final Equity    : {final_equity:.2f} USDT")
        print(f"Net PnL         : {pnl_net:+.2f} USDT ({roi_pct:+.2f}%)")
        print(f"Max Drawdown    : {max_dd_pct:.2f}%")
        if ts_max_dd is not None:
            try:
                dt = pd.to_datetime(int(ts_max_dd), unit="ms")
                print(f"Max DD At       : {dt}")
            except Exception:
                pass
        if ts_min_eq is not None:
            try:
                dt = pd.to_datetime(int(ts_min_eq), unit="ms")
                print(f"Min Equity At   : {dt}")
            except Exception:
                pass
        try:
            row = equity_df.iloc[idx_max_dd]
            dd_price = float(row.get("price", 0.0))
            dd_avg = float(row.get("avg_price", 0.0))
            dd_portions = int(row.get("portions_used", 0))
            print(f"DD Context      : price={dd_price:.4f} avg={dd_avg:.4f} portions={dd_portions}")
        except Exception:
            pass
        print(f"Portions Used   : {int(equity_df['portions_used'].max())}")
        if 'cycles_completed' in equity_df.columns:
            print(f"Cycles Closed   : {int(equity_df['cycles_completed'].max())}")

        try:
            max_notional = float(equity_df.get('notional_position', pd.Series([0.0])).fillna(0.0).max())
            max_lev_wallet = float(equity_df.get('lev_eff_wallet', pd.Series([0.0])).fillna(0.0).max())
            max_lev_cap = float(equity_df.get('lev_eff_capital', pd.Series([0.0])).fillna(0.0).max())
            max_lev_used_margin = float(equity_df.get('lev_eff_used_margin', pd.Series([0.0])).fillna(0.0).max())
            max_margin_pct_wallet = float(equity_df.get('margin_pct_wallet', pd.Series([0.0])).fillna(0.0).max()) * 100.0
            print(f"Max Notional    : {max_notional:.2f} USDT")
            print(f"Max Lev/Wallet  : {max_lev_wallet:.2f}x")
            print(f"Max Lev/Capital : {max_lev_cap:.2f}x")
            print(f"Max Lev/UsedMgn : {max_lev_used_margin:.2f}x")
            print(f"Max Used Margin : {max_margin_pct_wallet:.2f}% of wallet")
        except Exception:
            pass

        try:
            loss_series = pd.to_numeric(equity_df.get('loss_unrealized', pd.Series([0.0])), errors='coerce').fillna(0.0)
            liq_series = pd.to_numeric(equity_df.get('liq_limit', pd.Series([0.0])), errors='coerce').fillna(0.0)
            eq_series = pd.to_numeric(equity_df.get('equity', pd.Series([0.0])), errors='coerce').fillna(0.0)

            max_loss = float(loss_series.max())
            min_liq_limit = float(liq_series.min())
            min_equity = float(eq_series.min())

            idx_max_loss = int(loss_series.idxmax()) if not loss_series.empty else 0
            liq_at_max_loss = float(liq_series.iloc[idx_max_loss]) if idx_max_loss < len(liq_series) else 0.0
            ts_at_max_loss = equity_df.iloc[idx_max_loss].get('timestamp', None) if idx_max_loss < len(equity_df) else None

            idx_min_liq = int(liq_series.idxmin()) if not liq_series.empty else 0
            loss_at_min_liq = float(loss_series.iloc[idx_min_liq]) if idx_min_liq < len(loss_series) else 0.0
            ts_at_min_liq = equity_df.iloc[idx_min_liq].get('timestamp', None) if idx_min_liq < len(equity_df) else None

            max_excess = float((loss_series - liq_series).max()) if (not loss_series.empty and not liq_series.empty) else 0.0

            print(f"Worst Loss      : {max_loss:.2f} USDT")
            if ts_at_max_loss is not None:
                try:
                    dt = pd.to_datetime(int(ts_at_max_loss), unit='ms')
                    print(f"  at            : {dt}")
                except Exception:
                    pass
            print(f"Liq Limit @Loss : {liq_at_max_loss:.2f} USDT")
            print(f"Min Liq Limit   : {min_liq_limit:.2f} USDT")
            if ts_at_min_liq is not None:
                try:
                    dt = pd.to_datetime(int(ts_at_min_liq), unit='ms')
                    print(f"  at            : {dt}")
                except Exception:
                    pass
            print(f"Loss @MinLimit  : {loss_at_min_liq:.2f} USDT")
            print(f"Max (Loss-Limit): {max_excess:.2f} USDT")
            print(f"Min Equity      : {min_equity:.2f} USDT")
        except Exception:
            pass

        try:
            is_liq = bool(equity_df.get('is_liquidated', pd.Series([False])).fillna(False).any())
            if is_liq:
                print("Liquidated      : True")
                if 'liquidation_reason' in equity_df.columns:
                    reason_val = equity_df.loc[equity_df['is_liquidated'].fillna(False)].tail(1)['liquidation_reason'].values[0]
                    if reason_val is not None and not (pd.isna(reason_val)):
                        reason = str(reason_val).strip()
                        if reason and reason.lower() != 'nan':
                            print(f"Liq Reason      : {reason}")
            else:
                print("Liquidated      : False")
        except Exception:
            pass

        try:
            dca_price_triggers = int(stats.get("dca_price_triggers", 0))
            dca_executed = int(stats.get("dca_executed", 0))
            dca_blocked_indicator = int(stats.get("dca_blocked_indicator", 0))
            dca_blocked_mfi = int(stats.get("dca_blocked_mfi", 0))
            dca_blocked_macd = int(stats.get("dca_blocked_macd", 0))
            dca_blocked_margin = int(stats.get("dca_blocked_margin", 0))
            print(f"DCA Price Hits  : {dca_price_triggers}")
            print(f"DCA Executed    : {dca_executed}")
            if bool(cfg.cci12_enabled):
                print(f"DCA Blocked CCI : {dca_blocked_indicator}")
            if bool(cfg.mfi_enabled):
                print(f"DCA Blocked MFI : {dca_blocked_mfi}")
            if bool(cfg.macd_hist_flip_enabled):
                print(f"DCA Blocked MACD: {dca_blocked_macd}")
                if bool(cfg.macd_prev_opposite_tranche_enabled):
                    print(f"DCA Blocked Prev: {int(stats.get('dca_blocked_macd_prev_tranche', 0))}")
                    if bool(cfg.macd_prev_tranche_cci_enabled):
                        print(f"Prev Blocked CCI: {int(stats.get('dca_blocked_macd_prev_cci', 0))}")
                    if bool(cfg.macd_prev_tranche_cci_medium_enabled):
                        print(f"Prev Blocked CCI Medium: {int(stats.get('dca_blocked_macd_prev_cci_medium', 0))}")
                    if bool(cfg.macd_prev_tranche_cci_slow_enabled):
                        print(f"Prev Blocked CCI Slow  : {int(stats.get('dca_blocked_macd_prev_cci_slow', 0))}")
                    if bool(cfg.macd_prev_tranche_mfi_enabled):
                        print(f"Prev Blocked MFI: {int(stats.get('dca_blocked_macd_prev_mfi', 0))}")
                    if bool(cfg.macd_prev_tranche_dmi_enabled):
                        print(f"Prev Blocked DMI: {int(stats.get('dca_blocked_macd_prev_dmi', 0))}")
            print(f"DCA Blocked Mgn : {dca_blocked_margin}")
        except Exception:
            pass
        print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
