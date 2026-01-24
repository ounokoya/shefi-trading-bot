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
class VwmaConfig:
    fast: int = 12
    mid: int = 72
    slow: int = 168


@dataclass(frozen=True)
class ZoneConfig:
    fast_radius_pct: float = 0.001
    mid_radius_pct: float = 0.001
    slow_radius_pct: float = 0.001
    zone_large_mult: float = 2.0
    break_confirm_bars: int = 1


@dataclass(frozen=True)
class BacktestConfig:
    fee_rate: float = 0.0015
    no_lookahead: bool = True


@dataclass(frozen=True)
class ScalpStrategyConfig:
    tp_pct: float = 0.003
    sl_buffer_pct: float = 0.001


@dataclass(frozen=True)
class SwingStrategyConfig:
    sl_buffer_pct: float = 0.001


@dataclass(frozen=True)
class StrategyConfig:
    mode: str = "scalp"  # scalp|swing
    max_sl_pct: float | None = None
    scalp: ScalpStrategyConfig = ScalpStrategyConfig()
    swing: SwingStrategyConfig = SwingStrategyConfig()


@dataclass(frozen=True)
class OutputConfig:
    out_dir: str = "data/processed/backtests_triple_vwma_trend"


@dataclass(frozen=True)
class FullConfig:
    data: DataConfig
    vwma: VwmaConfig
    zones: ZoneConfig
    backtest: BacktestConfig
    strategy: StrategyConfig
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

    data_csv = _get(raw, ["data", "csv"], None)
    if not data_csv:
        raise ValueError("Missing config: data.csv")

    ohlc_raw = _get(raw, ["data", "ohlc"], {})
    ohlc = OhlcColumns(
        open=str(ohlc_raw.get("open", "open")),
        high=str(ohlc_raw.get("high", "high")),
        low=str(ohlc_raw.get("low", "low")),
        close=str(ohlc_raw.get("close", "close")),
        volume=str(ohlc_raw.get("volume", "volume")),
    )

    data = DataConfig(
        csv=str(data_csv),
        ts_col=str(_get(raw, ["data", "ts_col"], "ts")),
        ohlc=ohlc,
    )

    vw_raw = _get(raw, ["vwma"], {})
    vwma = VwmaConfig(
        fast=int(vw_raw.get("fast", 12)),
        mid=int(vw_raw.get("mid", 72)),
        slow=int(vw_raw.get("slow", 168)),
    )

    z_raw = _get(raw, ["zones"], {})
    zones = ZoneConfig(
        fast_radius_pct=float(z_raw.get("fast_radius_pct", 0.001)),
        mid_radius_pct=float(z_raw.get("mid_radius_pct", 0.001)),
        slow_radius_pct=float(z_raw.get("slow_radius_pct", 0.001)),
        zone_large_mult=float(z_raw.get("zone_large_mult", 2.0)),
        break_confirm_bars=int(z_raw.get("break_confirm_bars", 1)),
    )

    bt_raw = _get(raw, ["backtest"], {})
    backtest = BacktestConfig(
        fee_rate=float(bt_raw.get("fee_rate", 0.0015)),
        no_lookahead=bool(bt_raw.get("no_lookahead", True)),
    )

    strat_raw = _get(raw, ["strategy"], {})
    scalp_raw = _get(strat_raw, ["scalp"], {})
    swing_raw = _get(strat_raw, ["swing"], {})

    max_sl_pct_raw = strat_raw.get("max_sl_pct", None)
    max_sl_pct: float | None = None
    if max_sl_pct_raw is not None:
        max_sl_pct = float(max_sl_pct_raw)
        if float(max_sl_pct) <= 0.0:
            raise ValueError("strategy.max_sl_pct must be > 0 when provided")

    strategy = StrategyConfig(
        mode=str(strat_raw.get("mode", "scalp")),
        max_sl_pct=max_sl_pct,
        scalp=ScalpStrategyConfig(
            tp_pct=float(scalp_raw.get("tp_pct", 0.003)),
            sl_buffer_pct=float(scalp_raw.get("sl_buffer_pct", 0.001)),
        ),
        swing=SwingStrategyConfig(
            sl_buffer_pct=float(swing_raw.get("sl_buffer_pct", 0.001)),
        ),
    )

    out_raw = _get(raw, ["output"], {})
    output = OutputConfig(
        out_dir=str(out_raw.get("out_dir", "data/processed/backtests_triple_vwma_trend")),
    )

    mode = str(strategy.mode).strip().lower()
    if mode not in {"scalp", "swing"}:
        raise ValueError(f"Unexpected strategy.mode: {strategy.mode}")

    if float(strategy.scalp.tp_pct) <= 0.0 and mode == "scalp":
        raise ValueError("strategy.scalp.tp_pct must be > 0")
    if float(strategy.scalp.sl_buffer_pct) < 0.0:
        raise ValueError("strategy.scalp.sl_buffer_pct must be >= 0")
    if float(strategy.swing.sl_buffer_pct) < 0.0:
        raise ValueError("strategy.swing.sl_buffer_pct must be >= 0")
    if strategy.max_sl_pct is not None and float(strategy.max_sl_pct) <= 0.0:
        raise ValueError("strategy.max_sl_pct must be > 0")

    if int(vwma.fast) < 1 or int(vwma.mid) < 1 or int(vwma.slow) < 1:
        raise ValueError("vwma.fast/mid/slow must be >= 1")

    return FullConfig(
        data=data,
        vwma=vwma,
        zones=zones,
        backtest=backtest,
        strategy=strategy,
        output=output,
    )


def load_config_yaml(path: str | Path) -> FullConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping")
    return load_config_dict(raw)
