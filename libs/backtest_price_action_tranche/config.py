from __future__ import annotations

from dataclasses import dataclass, field
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

    stoch_k: int = 14
    stoch_d: int = 3

    vortex_period: int = 300
    dmi_period: int = 300
    dmi_adx_smoothing: int = 14

    atr_len: int = 14


@dataclass(frozen=True)
class TrendFilterConfig:
    enabled: bool = False
    mode: str = "none"


@dataclass(frozen=True)
class TriggerSignalConfig:
    name: str
    confluence_type: str = "instant"
    mode: str = "both"
    min_confirmed: int | None = None

    series_add: tuple[str, ...] = ()
    series_exclude: tuple[str, ...] = ()

    cci_fast_threshold: float | None = None
    cci_medium_threshold: float | None = None
    cci_slow_threshold: float | None = None


@dataclass(frozen=True)
class PriceActionSignalConfig:
    filters_add: tuple[str, ...] = ()
    filters_exclude: tuple[str, ...] = ()

    entry_mode: str = "simple"

    vwma_break_max_bars: int = 6
    tranche_hist_trend_mode: str = "none"
    macd_hist_accel_mode: str = "diff2"

    vwma_confirm_bars: int = 1
    macd_hist_slope_mode: str = "delta"
    exit_hist_sign_change_mode: str = "cross"


@dataclass(frozen=True)
class ExitPolicyConfig:
    allow_exit_signal: bool = True


@dataclass(frozen=True)
class TpConfig:
    mode: str = "none"
    tp_pct: float | None = None


@dataclass(frozen=True)
class SlConfig:
    mode: str = "none"
    sl_pct: float | None = None
    trail_pct: float | None = None
    atr_len: int | None = None
    atr_mult: float | None = None


@dataclass(frozen=True)
class PivotGridConfig:
    enabled: bool = False
    symbol: str | None = None

    registries: dict[str, str] = field(default_factory=dict)

    grid_pct: float = 0.05
    mode: str = "grid"
    keep_top2_5m: bool = True
    min_supports: int = 2
    min_resistances: int = 2
    zones_cfg: dict[str, dict[str, Any]] | None = None


@dataclass(frozen=True)
class PivotTemporalMemoryConfig:
    enabled: bool = False
    radius_pct: float = 0.01
    min_fast: int = 4
    min_medium: int = 2
    min_slow: int = 1
    max_events: int = 50


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

    signals_trigger: TriggerSignalConfig
    signals_entry: PriceActionSignalConfig
    signals_exit: PriceActionSignalConfig

    trend_filter: TrendFilterConfig
    exit_policy: ExitPolicyConfig
    tp: TpConfig
    sl: SlConfig
    pivot_grid: PivotGridConfig
    pivot_temporal_memory: PivotTemporalMemoryConfig
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
    stoch_raw = _get(ind_raw, ["stoch"], {})
    indicators = IndicatorsConfig(
        macd_fast=int(_get(ind_raw, ["macd", "fast"], 12)),
        macd_slow=int(_get(ind_raw, ["macd", "slow"], 26)),
        macd_signal=int(_get(ind_raw, ["macd", "signal"], 9)),
        cci_fast=int(_get(ind_raw, ["cci", "fast"], 30)),
        cci_medium=int(_get(ind_raw, ["cci", "medium"], 120)),
        cci_slow=int(_get(ind_raw, ["cci", "slow"], 300)),
        vwma_fast=int(_get(ind_raw, ["vwma", "fast"], 4)),
        vwma_medium=int(_get(ind_raw, ["vwma", "medium"], 12)),
        stoch_k=int(_get(stoch_raw, ["k"], 14)),
        stoch_d=int(_get(stoch_raw, ["d"], 3)),
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

    sig_raw = _get(raw, ["signals"], {})

    trig_raw = _get(sig_raw, ["trigger"], {})
    trig_name = trig_raw.get("name")
    if not trig_name:
        raise ValueError("Missing config: signals.trigger.name")
    trig_params = trig_raw.get("params", {})
    if not isinstance(trig_params, dict):
        trig_params = {}

    series_cfg = trig_params.get("series", {})
    if not isinstance(series_cfg, dict):
        series_cfg = {}
    add_raw = series_cfg.get("add", [])
    if add_raw is None:
        add_raw = []
    if not isinstance(add_raw, list):
        add_raw = [add_raw]
    series_add = tuple(str(x) for x in add_raw)

    exc_raw = series_cfg.get("exclude", [])
    if exc_raw is None:
        exc_raw = []
    if not isinstance(exc_raw, list):
        exc_raw = [exc_raw]
    series_exclude = tuple(str(x) for x in exc_raw)

    cci_thr = trig_params.get("cci_thresholds", {})
    if not isinstance(cci_thr, dict):
        cci_thr = {}

    signals_trigger = TriggerSignalConfig(
        name=str(trig_name),
        confluence_type=str(trig_params.get("confluence_type", "instant")),
        mode=str(trig_params.get("mode", "both")),
        min_confirmed=(None if trig_params.get("min_confirmed") is None else int(trig_params.get("min_confirmed"))),
        series_add=series_add,
        series_exclude=series_exclude,
        cci_fast_threshold=(None if cci_thr.get("fast") is None else float(cci_thr.get("fast"))),
        cci_medium_threshold=(None if cci_thr.get("medium") is None else float(cci_thr.get("medium"))),
        cci_slow_threshold=(None if cci_thr.get("slow") is None else float(cci_thr.get("slow"))),
    )

    def _pa_from(section: str) -> PriceActionSignalConfig:
        sraw = _get(sig_raw, [section], {})
        params = sraw.get("params", {})
        if not isinstance(params, dict):
            params = {}
        pa_raw = params.get("price_action", {})
        if not isinstance(pa_raw, dict):
            pa_raw = {}
        filters_raw = pa_raw.get("filters", {})
        if not isinstance(filters_raw, dict):
            filters_raw = {}

        add2 = filters_raw.get("add", [])
        if add2 is None:
            add2 = []
        if not isinstance(add2, list):
            add2 = [add2]
        exclude2 = filters_raw.get("exclude", [])
        if exclude2 is None:
            exclude2 = []
        if not isinstance(exclude2, list):
            exclude2 = [exclude2]

        return PriceActionSignalConfig(
            filters_add=tuple(str(x) for x in add2),
            filters_exclude=tuple(str(x) for x in exclude2),
            entry_mode=str(pa_raw.get("entry_mode", "simple")),
            vwma_break_max_bars=int(pa_raw.get("vwma_break_max_bars", 6) or 6),
            tranche_hist_trend_mode=str(pa_raw.get("tranche_hist_trend_mode", "none")),
            macd_hist_accel_mode=str(pa_raw.get("macd_hist_accel_mode", "diff2")),
            vwma_confirm_bars=int(pa_raw.get("vwma_confirm_bars", 1) or 1),
            macd_hist_slope_mode=str(pa_raw.get("macd_hist_slope_mode", "delta")),
            exit_hist_sign_change_mode=str(pa_raw.get("exit_hist_sign_change_mode", "cross")),
        )

    signals_entry = _pa_from("entry")
    signals_exit = _pa_from("exit")

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

    pg_raw = _get(raw, ["pivot_grid"], {})
    if not isinstance(pg_raw, dict):
        pg_raw = {}
    pg_regs_raw = pg_raw.get("registries", {})
    if not isinstance(pg_regs_raw, dict):
        pg_regs_raw = {}
    pg_regs = {str(k): str(v) for k, v in pg_regs_raw.items() if str(k).strip() and str(v).strip()}
    pg_zones_cfg_raw = pg_raw.get("zones_cfg")
    pg_zones_cfg = None
    if isinstance(pg_zones_cfg_raw, dict):
        pg_zones_cfg = {str(k): dict(v) for k, v in pg_zones_cfg_raw.items() if isinstance(v, dict)}
    pivot_grid = PivotGridConfig(
        enabled=bool(pg_raw.get("enabled", False)),
        symbol=(None if pg_raw.get("symbol") is None else str(pg_raw.get("symbol"))),
        registries=pg_regs,
        grid_pct=float(pg_raw.get("grid_pct", 0.05)),
        mode=str(pg_raw.get("mode", "grid")),
        keep_top2_5m=bool(pg_raw.get("keep_top2_5m", True)),
        min_supports=int(pg_raw.get("min_supports", 2)),
        min_resistances=int(pg_raw.get("min_resistances", 2)),
        zones_cfg=pg_zones_cfg,
    )

    ptm_raw = _get(raw, ["pivot_temporal_memory"], {})
    if not isinstance(ptm_raw, dict):
        ptm_raw = {}
    pivot_temporal_memory = PivotTemporalMemoryConfig(
        enabled=bool(ptm_raw.get("enabled", False)),
        radius_pct=float(ptm_raw.get("radius_pct", 0.01)),
        min_fast=int(ptm_raw.get("min_fast", 4)),
        min_medium=int(ptm_raw.get("min_medium", 2)),
        min_slow=int(ptm_raw.get("min_slow", 1)),
        max_events=int(ptm_raw.get("max_events", 50)),
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
        signals_trigger=signals_trigger,
        signals_entry=signals_entry,
        signals_exit=signals_exit,
        trend_filter=trend_filter,
        exit_policy=exit_policy,
        tp=tp,
        sl=sl,
        pivot_grid=pivot_grid,
        pivot_temporal_memory=pivot_temporal_memory,
        output=output,
    )

    tp_mode = str(cfg.tp.mode).lower()
    if tp_mode == "fixed_pct" and cfg.tp.tp_pct is None:
        raise ValueError("tp.mode=fixed_pct requires tp.tp_pct")
    if tp_mode not in {"", "none", "off", "0", "fixed_pct", "pivot_grid"}:
        raise ValueError(f"Unexpected tp.mode: {cfg.tp.mode}")

    if tp_mode == "pivot_grid":
        if not bool(cfg.pivot_grid.enabled):
            raise ValueError("tp.mode=pivot_grid requires pivot_grid.enabled=true")
        if not str(cfg.pivot_grid.symbol or "").strip():
            raise ValueError("tp.mode=pivot_grid requires pivot_grid.symbol")
        regs = dict(cfg.pivot_grid.registries or {})
        for tf in ("5m", "1h", "4h"):
            if tf not in regs or not str(regs.get(tf) or "").strip():
                raise ValueError(f"tp.mode=pivot_grid requires pivot_grid.registries.{tf}")
        mm = str(cfg.pivot_grid.mode).strip().lower()
        if mm not in {"grid", "zones"}:
            raise ValueError(f"Unexpected pivot_grid.mode: {cfg.pivot_grid.mode}")
        if int(cfg.pivot_grid.min_supports) < 0 or int(cfg.pivot_grid.min_resistances) < 0:
            raise ValueError("pivot_grid.min_supports/min_resistances must be >= 0")
        if float(cfg.pivot_grid.grid_pct) <= 0 and mm == "grid":
            raise ValueError("pivot_grid.grid_pct must be > 0 when pivot_grid.mode=grid")

    if float(cfg.pivot_temporal_memory.radius_pct) < 0:
        raise ValueError("pivot_temporal_memory.radius_pct must be >= 0")
    if (
        int(cfg.pivot_temporal_memory.min_fast) < 0
        or int(cfg.pivot_temporal_memory.min_medium) < 0
        or int(cfg.pivot_temporal_memory.min_slow) < 0
    ):
        raise ValueError("pivot_temporal_memory.min_fast/min_medium/min_slow must be >= 0")
    if int(cfg.pivot_temporal_memory.max_events) < 1:
        raise ValueError("pivot_temporal_memory.max_events must be >= 1")

    sl_mode2 = str(cfg.sl.mode).lower()
    if sl_mode2 == "fixed_pct" and cfg.sl.sl_pct is None:
        raise ValueError("sl.mode=fixed_pct requires sl.sl_pct")
    if sl_mode2 == "trailing_pct" and cfg.sl.trail_pct is None:
        raise ValueError("sl.mode=trailing_pct requires sl.trail_pct (or sl.sl_pct as alias)")
    if sl_mode2 == "atr" and cfg.sl.atr_mult is None:
        raise ValueError("sl.mode=atr requires sl.atr_mult")
    if sl_mode2 not in {"", "none", "off", "0", "fixed_pct", "trailing_pct", "atr"}:
        raise ValueError(f"Unexpected sl.mode: {cfg.sl.mode}")

    for section_name, pa_cfg in (("entry", cfg.signals_entry), ("exit", cfg.signals_exit)):
        m = str(pa_cfg.entry_mode).strip().lower()
        if m not in {"", "simple", "vwma_break"}:
            raise ValueError(f"Unexpected signals.{section_name}.params.price_action.entry_mode: {pa_cfg.entry_mode}")
        if int(pa_cfg.vwma_break_max_bars) < 1:
            raise ValueError(f"signals.{section_name}.params.price_action.vwma_break_max_bars must be >= 1")
        t = str(pa_cfg.tranche_hist_trend_mode).strip().lower()
        if t not in {"", "none", "trend", "healthy"}:
            raise ValueError(
                f"Unexpected signals.{section_name}.params.price_action.tranche_hist_trend_mode: {pa_cfg.tranche_hist_trend_mode}"
            )
        a = str(pa_cfg.macd_hist_accel_mode).strip().lower()
        if a not in {"", "diff2", "mono"}:
            raise ValueError(
                f"Unexpected signals.{section_name}.params.price_action.macd_hist_accel_mode: {pa_cfg.macd_hist_accel_mode}"
            )

    return cfg


def load_config_yaml(path: str | Path) -> FullConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping")
    return load_config_dict(raw)
