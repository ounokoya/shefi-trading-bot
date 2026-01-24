from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class OhlcColumns:
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: str = "volume"


@dataclass(frozen=True)
class DataConfig:
    csv: str
    ts_col: str = "ts"
    ohlc: OhlcColumns = OhlcColumns()


@dataclass(frozen=True)
class IndicatorsConfig:
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    cci_fast: int = 30
    cci_medium: int = 120
    cci_slow: int = 300

    vwma_fast: int = 4
    vwma_medium: int = 12

    vortex_period: int = 300
    dmi_period: int = 300
    dmi_adx_smoothing: int = 14

    atr_len: int = 14


@dataclass(frozen=True)
class TrendFilterConfig:
    enabled: bool = False
    mode: str = "none"  # none|vortex|dmi|both


@dataclass(frozen=True)
class SignalPresetConfig:
    name: str
    confluence_type: str = "instant"  # instant|tranche_last
    mode: str = "both"  # long|short|both
    min_confirmed: int | None = None
    direction_rule: str = "any"  # any|opposite_to_position

    series_add: tuple[str, ...] = ()
    series_exclude: tuple[str, ...] = ()

    cci_fast_threshold: float | None = None
    cci_medium_threshold: float | None = None
    cci_slow_threshold: float | None = None


@dataclass(frozen=True)
class TpConfig:
    mode: str = "none"  # none|fixed_pct
    tp_pct: float | None = None


@dataclass(frozen=True)
class SlConfig:
    mode: str = "none"  # none|fixed_pct|trailing_pct|atr
    sl_pct: float | None = None
    trail_pct: float | None = None
    atr_len: int | None = None
    atr_mult: float | None = None


@dataclass(frozen=True)
class ExitPolicyConfig:
    allow_exit_signal: bool = True


@dataclass(frozen=True)
class BacktestConfig:
    window_size: int = 600
    horizon_days: int = 7
    fee_rate: float = 0.0015


@dataclass(frozen=True)
class OutputConfig:
    out_dir: str = "data/processed/backtests"
    png: bool = True


@dataclass(frozen=True)
class FullConfig:
    data: DataConfig
    indicators: IndicatorsConfig
    backtest: BacktestConfig
    signals_entry: SignalPresetConfig
    signals_exit: SignalPresetConfig
    trend_filter: TrendFilterConfig
    exit_policy: ExitPolicyConfig
    tp: TpConfig
    sl: SlConfig
    output: OutputConfig


def _get(d: dict[str, Any], path: list[str], default: Any) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_config_dict(raw: dict[str, Any]) -> FullConfig:
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping")

    ohlc_raw = _get(raw, ["data", "ohlc"], {})
    ohlc = OhlcColumns(
        open=str(ohlc_raw.get("open", "open")),
        high=str(ohlc_raw.get("high", "high")),
        low=str(ohlc_raw.get("low", "low")),
        close=str(ohlc_raw.get("close", "close")),
        volume=str(ohlc_raw.get("volume", "volume")),
    )

    data_csv = _get(raw, ["data", "csv"], None)
    if not data_csv:
        raise ValueError("Missing config: data.csv")
    data = DataConfig(
        csv=str(data_csv),
        ts_col=str(_get(raw, ["data", "ts_col"], "ts")),
        ohlc=ohlc,
    )

    ind_raw = _get(raw, ["indicators"], {})
    indicators = IndicatorsConfig(
        macd_fast=int(_get(ind_raw, ["macd", "fast"], 12)),
        macd_slow=int(_get(ind_raw, ["macd", "slow"], 26)),
        macd_signal=int(_get(ind_raw, ["macd", "signal"], 9)),
        cci_fast=int(_get(ind_raw, ["cci", "fast"], 30)),
        cci_medium=int(_get(ind_raw, ["cci", "medium"], 120)),
        cci_slow=int(_get(ind_raw, ["cci", "slow"], 300)),
        vwma_fast=int(_get(ind_raw, ["vwma", "fast"], 4)),
        vwma_medium=int(_get(ind_raw, ["vwma", "medium"], 12)),
        vortex_period=int(_get(ind_raw, ["vortex", "period"], 300)),
        dmi_period=int(_get(ind_raw, ["dmi", "period"], 300)),
        dmi_adx_smoothing=int(_get(ind_raw, ["dmi", "adx_smoothing"], 14)),
        atr_len=int(_get(ind_raw, ["atr", "len"], 14)),
    )

    bt_raw = _get(raw, ["backtest"], {})
    backtest = BacktestConfig(
        window_size=int(bt_raw.get("window_size", 600)),
        horizon_days=int(bt_raw.get("horizon_days", 7)),
        fee_rate=float(bt_raw.get("fee_rate", 0.0015)),
    )

    tf_raw = _get(raw, ["trend_filter"], {})
    trend_filter = TrendFilterConfig(
        enabled=bool(tf_raw.get("enabled", False)),
        mode=str(tf_raw.get("mode", "none")),
    )

    def _signal_from(section: str, *, default_direction_rule: str) -> SignalPresetConfig:
        sraw = _get(raw, ["signals", section], {})
        name = sraw.get("name")
        if not name:
            raise ValueError(f"Missing config: signals.{section}.name")
        params = sraw.get("params", {})
        if not isinstance(params, dict):
            params = {}

        series_cfg = params.get("series", {})
        if not isinstance(series_cfg, dict):
            series_cfg = {}
        series_add_raw = series_cfg.get("add", [])
        if series_add_raw is None:
            series_add_raw = []
        if not isinstance(series_add_raw, list):
            series_add_raw = [series_add_raw]
        series_add = tuple(str(x) for x in series_add_raw)

        series_exclude_raw = series_cfg.get("exclude", [])
        if series_exclude_raw is None:
            series_exclude_raw = []
        if not isinstance(series_exclude_raw, list):
            series_exclude_raw = [series_exclude_raw]
        series_exclude = tuple(str(x) for x in series_exclude_raw)

        cci_thr = params.get("cci_thresholds", {})
        if not isinstance(cci_thr, dict):
            cci_thr = {}

        return SignalPresetConfig(
            name=str(name),
            confluence_type=str(params.get("confluence_type", "instant")),
            mode=str(params.get("mode", "both")),
            min_confirmed=(None if params.get("min_confirmed") is None else int(params.get("min_confirmed"))),
            direction_rule=str(sraw.get("direction_rule", default_direction_rule)),
            series_add=series_add,
            series_exclude=series_exclude,
            cci_fast_threshold=(None if cci_thr.get("fast") is None else float(cci_thr.get("fast"))),
            cci_medium_threshold=(None if cci_thr.get("medium") is None else float(cci_thr.get("medium"))),
            cci_slow_threshold=(None if cci_thr.get("slow") is None else float(cci_thr.get("slow"))),
        )

    signals_entry = _signal_from("entry", default_direction_rule="any")
    signals_exit = _signal_from("exit", default_direction_rule="opposite_to_position")

    ep_raw = _get(raw, ["exit_policy"], {})
    exit_policy = ExitPolicyConfig(
        allow_exit_signal=bool(ep_raw.get("allow_exit_signal", True)),
    )

    tp_raw = _get(raw, ["tp"], {})
    tp = TpConfig(
        mode=str(tp_raw.get("mode", "none")),
        tp_pct=(None if tp_raw.get("tp_pct") is None else float(tp_raw.get("tp_pct"))),
    )

    sl_raw = _get(raw, ["sl"], {})
    sl_mode = str(sl_raw.get("mode", "none"))
    sl_pct = None if sl_raw.get("sl_pct") is None else float(sl_raw.get("sl_pct"))
    trail_pct = None if sl_raw.get("trail_pct") is None else float(sl_raw.get("trail_pct"))
    if str(sl_mode).lower() == "trailing_pct" and trail_pct is None and sl_pct is not None:
        trail_pct = float(sl_pct)
    sl = SlConfig(
        mode=sl_mode,
        sl_pct=sl_pct,
        trail_pct=trail_pct,
        atr_len=(None if sl_raw.get("atr_len") is None else int(sl_raw.get("atr_len"))),
        atr_mult=(None if sl_raw.get("atr_mult") is None else float(sl_raw.get("atr_mult"))),
    )

    out_raw = _get(raw, ["output"], {})
    output = OutputConfig(
        out_dir=str(out_raw.get("out_dir", "data/processed/backtests")),
        png=bool(out_raw.get("png", True)),
    )

    cfg = FullConfig(
        data=data,
        indicators=indicators,
        backtest=backtest,
        signals_entry=signals_entry,
        signals_exit=signals_exit,
        trend_filter=trend_filter,
        exit_policy=exit_policy,
        tp=tp,
        sl=sl,
        output=output,
    )

    # Validation (fail-fast)
    tp_mode = str(cfg.tp.mode).lower()
    if tp_mode == "fixed_pct" and cfg.tp.tp_pct is None:
        raise ValueError("tp.mode=fixed_pct requires tp.tp_pct")
    if tp_mode not in {"", "none", "off", "0", "fixed_pct"}:
        raise ValueError(f"Unexpected tp.mode: {cfg.tp.mode}")

    sl_mode2 = str(cfg.sl.mode).lower()
    if sl_mode2 == "fixed_pct" and cfg.sl.sl_pct is None:
        raise ValueError("sl.mode=fixed_pct requires sl.sl_pct")
    if sl_mode2 == "trailing_pct" and cfg.sl.trail_pct is None:
        raise ValueError("sl.mode=trailing_pct requires sl.trail_pct (or sl.sl_pct as alias)")
    if sl_mode2 == "atr" and cfg.sl.atr_mult is None:
        raise ValueError("sl.mode=atr requires sl.atr_mult")
    if sl_mode2 not in {"", "none", "off", "0", "fixed_pct", "trailing_pct", "atr"}:
        raise ValueError(f"Unexpected sl.mode: {cfg.sl.mode}")

    return cfg


def load_config_yaml(path: str | Path) -> FullConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping")
    return load_config_dict(raw)
