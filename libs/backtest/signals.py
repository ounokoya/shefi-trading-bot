from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from libs.backtest.config import FullConfig, SignalPresetConfig
from libs.blocks.get_current_tranche_extreme_zone_confluence_signal import (
    get_current_tranche_extreme_zone_confluence_signal,
    get_current_tranche_extreme_zone_confluence_tranche_last_signal,
)
from libs.presets.extreme_confluence_presets import get_extreme_confluence_preset


@dataclass(frozen=True)
class SignalDecision:
    side: str | None  # LONG|SHORT
    meta: dict[str, object]


def _target_close_extreme_kind(mode: str) -> str | None:
    m = str(mode).lower()
    if m == "long":
        return "LOW"
    if m == "short":
        return "HIGH"
    if m == "both":
        return None
    raise ValueError(f"Unexpected mode: {mode}")


def _opposite_side(side: str) -> str:
    s = str(side).upper()
    if s == "LONG":
        return "SHORT"
    if s == "SHORT":
        return "LONG"
    raise ValueError(f"Unexpected side: {side}")


def _confluence_signal(
    window: pd.DataFrame,
    *,
    cfg: FullConfig,
    sig: SignalPresetConfig,
    trend_filter: str,
) -> dict[str, object]:
    preset = get_extreme_confluence_preset(sig.name)

    cci_fast_col = f"cci_{int(cfg.indicators.cci_fast)}"
    cci_medium_col = f"cci_{int(cfg.indicators.cci_medium)}"
    cci_slow_col = f"cci_{int(cfg.indicators.cci_slow)}"
    vwma_fast_col = f"vwma_{int(cfg.indicators.vwma_fast)}"
    vwma_medium_col = f"vwma_{int(cfg.indicators.vwma_medium)}"

    aliases = {
        "cci_fast": cci_fast_col,
        "cci_medium": cci_medium_col,
        "cci_slow": cci_slow_col,
        "vwma_fast": vwma_fast_col,
        "vwma_medium": vwma_medium_col,
        # legacy preset names (map to current configured periods)
        "cci_30": cci_fast_col,
        "cci_120": cci_medium_col,
        "cci_300": cci_slow_col,
        "vwma_4": vwma_fast_col,
        "vwma_12": vwma_medium_col,
    }

    def _resolve_series_name(name: object) -> str:
        k = str(name).strip()
        if not k:
            return ""
        return str(aliases.get(k, k))

    series_cols_raw = list(preset.series_cols)
    for c in sig.series_add:
        cc = str(c).strip()
        if cc and cc not in series_cols_raw:
            series_cols_raw.append(cc)

    series_cols: list[str] = []
    for c in series_cols_raw:
        cc = _resolve_series_name(c)
        if cc and cc not in series_cols:
            series_cols.append(cc)

    excludes_raw = [str(c).strip() for c in sig.series_exclude if str(c).strip()]
    excludes = {_resolve_series_name(c) for c in excludes_raw if _resolve_series_name(c)}
    if excludes:
        series_cols = [c for c in series_cols if c not in excludes]

    if not series_cols:
        raise ValueError("signals.*.params.series leads to empty series_cols")

    if sig.min_confirmed is None:
        required_min = len(series_cols)
    else:
        required_min = int(sig.min_confirmed)
        if required_min < 1:
            raise ValueError("min_confirmed must be >= 1")
        if required_min > len(series_cols):
            raise ValueError(f"min_confirmed={required_min} is greater than number of series ({len(series_cols)})")

    cci_fast_thr = sig.cci_fast_threshold if sig.cci_fast_threshold is not None else preset.cci_fast_threshold
    cci_medium_thr = sig.cci_medium_threshold if sig.cci_medium_threshold is not None else preset.cci_medium_threshold
    cci_slow_thr = sig.cci_slow_threshold if sig.cci_slow_threshold is not None else preset.cci_slow_threshold

    target = _target_close_extreme_kind(sig.mode)

    missing = [c for c in series_cols if c not in set(window.columns)]
    if missing:
        raise ValueError(f"Missing series columns in window: {missing}")

    if str(sig.confluence_type).lower() == "tranche_last":
        return get_current_tranche_extreme_zone_confluence_tranche_last_signal(
            window,
            ts_col=cfg.data.ts_col,
            hist_col="macd_hist",
            close_col=cfg.data.ohlc.close,
            series_cols=series_cols,
            target_close_extreme_kind=target,
            cci_fast_threshold=cci_fast_thr,
            cci_medium_threshold=cci_medium_thr,
            cci_slow_threshold=cci_slow_thr,
            cci_fast_col=cci_fast_col,
            cci_medium_col=cci_medium_col,
            cci_slow_col=cci_slow_col,
            min_confirmed=required_min,
            trend_filter=trend_filter,
        )

    return get_current_tranche_extreme_zone_confluence_signal(
        window,
        ts_col=cfg.data.ts_col,
        hist_col="macd_hist",
        close_col=cfg.data.ohlc.close,
        series_cols=series_cols,
        target_close_extreme_kind=target,
        cci_fast_threshold=cci_fast_thr,
        cci_medium_threshold=cci_medium_thr,
        cci_slow_threshold=cci_slow_thr,
        cci_fast_col=cci_fast_col,
        cci_medium_col=cci_medium_col,
        cci_slow_col=cci_slow_col,
        min_confirmed=required_min,
        trend_filter=trend_filter,
    )


def entry_signal(window: pd.DataFrame, *, cfg: FullConfig) -> SignalDecision:
    tf = "none"
    if bool(cfg.trend_filter.enabled):
        tf = str(cfg.trend_filter.mode)

    z = _confluence_signal(window, cfg=cfg, sig=cfg.signals_entry, trend_filter=tf)
    if not bool(z.get("is_zone")):
        return SignalDecision(side=None, meta={"is_zone": False})

    side = z.get("open_side")
    if side not in ("LONG", "SHORT"):
        return SignalDecision(side=None, meta={"is_zone": True, "open_side": side})
    return SignalDecision(side=str(side), meta=dict(z))


def exit_signal(window: pd.DataFrame, *, cfg: FullConfig, position_side: str) -> SignalDecision:
    # IMPORTANT: trend_filter is NOT applied on exit.
    z = _confluence_signal(window, cfg=cfg, sig=cfg.signals_exit, trend_filter="none")
    if not bool(z.get("is_zone")):
        return SignalDecision(side=None, meta={"is_zone": False})

    side = z.get("open_side")
    if side not in ("LONG", "SHORT"):
        return SignalDecision(side=None, meta={"is_zone": True, "open_side": side})

    rule = str(cfg.signals_exit.direction_rule).lower()
    if rule == "opposite_to_position":
        if str(side).upper() != _opposite_side(position_side):
            return SignalDecision(side=None, meta=dict(z))
        return SignalDecision(side=str(side), meta=dict(z))

    if rule == "any":
        return SignalDecision(side=str(side), meta=dict(z))

    raise ValueError(f"Unexpected exit direction_rule: {cfg.signals_exit.direction_rule}")
