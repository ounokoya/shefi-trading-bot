from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BybitConfig:
    symbol: str = "LINKUSDT"
    category: str = "linear"
    base_url: str = "https://api.bybit.com"
    interval: str = "6h"
    limit: int = 1000
    start: str = "2025-01-01"
    end: str = "2025-12-31"


@dataclass(frozen=True)
class WindowConfig:
    window_days: int = 90


@dataclass(frozen=True)
class IndicatorsConfig:
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    dmi_period: int = 14
    dmi_adx_smoothing: int = 6

    stoch_k: int = 12
    stoch_k_smooth: int = 2
    stoch_d: int = 3

    cci_period: int = 20


@dataclass(frozen=True)
class PivotsConfig:
    zone_pct: float = 0.01
    merge_pct: float = 0.01
    max_pivots: int = 10


@dataclass(frozen=True)
class SignalsConfig:
    condition_window_bars: int = 30
    enable_premature: bool = True


@dataclass(frozen=True)
class BacktestFlipConfig:
    sl_pct: float = 0.05


@dataclass(frozen=True)
class OutputConfig:
    out_dir: str = "data/processed/backtests/new_strategie_flip"
    png: bool = True
    save_csv: bool = True


@dataclass(frozen=True)
class FullNewStrategieBacktestConfig:
    bybit: BybitConfig = BybitConfig()
    window: WindowConfig = WindowConfig()
    indicators: IndicatorsConfig = IndicatorsConfig()
    pivots: PivotsConfig = PivotsConfig()
    signals: SignalsConfig = SignalsConfig()
    backtest: BacktestFlipConfig = BacktestFlipConfig()
    output: OutputConfig = OutputConfig()


def _get(d: dict[str, Any], path: list[str], default: Any) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_config_dict(raw: dict[str, Any]) -> FullNewStrategieBacktestConfig:
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping")

    by = _get(raw, ["bybit"], {})
    bybit = BybitConfig(
        symbol=str(by.get("symbol", "LINKUSDT")),
        category=str(by.get("category", "linear")),
        base_url=str(by.get("base_url", "https://api.bybit.com")),
        interval=str(by.get("interval", "6h")),
        limit=int(by.get("limit", 1000)),
        start=str(by.get("start", "2025-01-01")),
        end=str(by.get("end", "2025-12-31")),
    )

    win = _get(raw, ["window"], {})
    window = WindowConfig(window_days=int(win.get("window_days", 90)))

    ind = _get(raw, ["indicators"], {})
    macd = _get(ind, ["macd"], {})
    dmi = _get(ind, ["dmi"], {})
    stoch = _get(ind, ["stoch"], {})
    cci = _get(ind, ["cci"], {})
    indicators = IndicatorsConfig(
        macd_fast=int(macd.get("fast", 12)),
        macd_slow=int(macd.get("slow", 26)),
        macd_signal=int(macd.get("signal", 9)),
        dmi_period=int(dmi.get("period", 14)),
        dmi_adx_smoothing=int(dmi.get("adx_smoothing", 6)),
        stoch_k=int(stoch.get("k", 12)),
        stoch_k_smooth=int(stoch.get("k_smooth", 2)),
        stoch_d=int(stoch.get("d", 3)),
        cci_period=int(cci.get("period", 20)),
    )

    piv = _get(raw, ["pivots"], {})
    pivots = PivotsConfig(
        zone_pct=float(piv.get("zone_pct", 0.01)),
        merge_pct=float(piv.get("merge_pct", 0.01)),
        max_pivots=int(piv.get("max_pivots", 10)),
    )

    sig = _get(raw, ["signals"], {})
    signals = SignalsConfig(
        condition_window_bars=int(sig.get("condition_window_bars", 30)),
        enable_premature=bool(sig.get("enable_premature", True)),
    )

    bt = _get(raw, ["backtest"], {})
    backtest = BacktestFlipConfig(sl_pct=float(bt.get("sl_pct", 0.05)))

    out = _get(raw, ["output"], {})
    output = OutputConfig(
        out_dir=str(out.get("out_dir", "data/processed/backtests/new_strategie_flip")),
        png=bool(out.get("png", True)),
        save_csv=bool(out.get("save_csv", True)),
    )

    return FullNewStrategieBacktestConfig(
        bybit=bybit,
        window=window,
        indicators=indicators,
        pivots=pivots,
        signals=signals,
        backtest=backtest,
        output=output,
    )


def load_config_yaml(path: str | Path) -> FullNewStrategieBacktestConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping")
    return load_config_dict(raw)
