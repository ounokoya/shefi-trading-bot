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
    exec_interval: str = "5m"
    ctx_interval: str = "15m"
    exec_limit: int = 1000
    ctx_limit: int = 1000
    start: str = "2026-01-01"
    end: str = "2026-01-12"
    warmup_bars: int = 0


@dataclass(frozen=True)
class MacdConfig:
    fast: int = 12
    slow: int = 26
    signal: int = 9


@dataclass(frozen=True)
class CciConfig:
    fast: int = 30
    medium: int = 120
    slow: int = 300


@dataclass(frozen=True)
class StochConfig:
    k: int = 14
    d: int = 3


@dataclass(frozen=True)
class IndicatorsConfig:
    exec_macd: MacdConfig = MacdConfig()
    ctx_macd: MacdConfig = MacdConfig()
    exec_cci: CciConfig = CciConfig()
    ctx_cci: CciConfig = CciConfig()
    stoch: StochConfig = StochConfig()


@dataclass(frozen=True)
class AgentConfig:
    exec_cci_extreme: float = 100.0
    ctx_cci_extreme: float = 100.0
    min_abs_force_exec: float = 0.0
    min_abs_force_ctx: float = 0.0
    take_exec_cci_extreme_if_ctx_not_extreme: bool = False
    take_exec_and_ctx_cci_extreme: bool = False
    signal_on_ctx_flip_if_exec_aligned: bool = False


@dataclass(frozen=True)
class BacktestConfig:
    fee_rate: float = 0.0015
    exit_mode: str = "exec_tranche_end"
    tp_pct: float = 0.0
    trailing_stop_pct: float = 0.0
    sl_pct: float = 0.0
    stoch_high: float = 80.0
    stoch_low: float = 20.0
    stoch_wait_extreme: bool = True
    max_signals: int = 0


@dataclass(frozen=True)
class OutputConfig:
    out_dir: str = "data/processed/backtests/macd_momentum_two_tf"
    save_csv: bool = True
    print_top_reasons: int = 10


@dataclass(frozen=True)
class FullConfig:
    bybit: BybitConfig = BybitConfig()
    indicators: IndicatorsConfig = IndicatorsConfig()
    agent: AgentConfig = AgentConfig()
    backtest: BacktestConfig = BacktestConfig()
    output: OutputConfig = OutputConfig()


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

    bybit_raw = _get(raw, ["bybit"], {})
    bybit = BybitConfig(
        symbol=str(bybit_raw.get("symbol", "LINKUSDT")),
        category=str(bybit_raw.get("category", "linear")),
        base_url=str(bybit_raw.get("base_url", "https://api.bybit.com")),
        exec_interval=str(bybit_raw.get("exec_interval", "5m")),
        ctx_interval=str(bybit_raw.get("ctx_interval", "15m")),
        exec_limit=int(bybit_raw.get("exec_limit", 1000)),
        ctx_limit=int(bybit_raw.get("ctx_limit", 1000)),
        start=str(bybit_raw.get("start", "2026-01-01")),
        end=str(bybit_raw.get("end", "2026-01-12")),
        warmup_bars=int(bybit_raw.get("warmup_bars", 0)),
    )

    ind_raw = _get(raw, ["indicators"], {})

    exec_macd_raw = _get(ind_raw, ["exec_macd"], {})
    ctx_macd_raw = _get(ind_raw, ["ctx_macd"], {})
    exec_cci_raw = _get(ind_raw, ["exec_cci"], {})
    ctx_cci_raw = _get(ind_raw, ["ctx_cci"], {})
    stoch_raw = _get(ind_raw, ["stoch"], {})

    indicators = IndicatorsConfig(
        exec_macd=MacdConfig(
            fast=int(exec_macd_raw.get("fast", 12)),
            slow=int(exec_macd_raw.get("slow", 26)),
            signal=int(exec_macd_raw.get("signal", 9)),
        ),
        ctx_macd=MacdConfig(
            fast=int(ctx_macd_raw.get("fast", 12)),
            slow=int(ctx_macd_raw.get("slow", 26)),
            signal=int(ctx_macd_raw.get("signal", 9)),
        ),
        exec_cci=CciConfig(
            fast=int(exec_cci_raw.get("fast", 30)),
            medium=int(exec_cci_raw.get("medium", 120)),
            slow=int(exec_cci_raw.get("slow", 300)),
        ),
        ctx_cci=CciConfig(
            fast=int(ctx_cci_raw.get("fast", 30)),
            medium=int(ctx_cci_raw.get("medium", 120)),
            slow=int(ctx_cci_raw.get("slow", 300)),
        ),
        stoch=StochConfig(
            k=int(stoch_raw.get("k", 14)),
            d=int(stoch_raw.get("d", 3)),
        ),
    )

    agent_raw = _get(raw, ["agent"], {})
    agent = AgentConfig(
        exec_cci_extreme=float(agent_raw.get("exec_cci_extreme", 100.0)),
        ctx_cci_extreme=float(agent_raw.get("ctx_cci_extreme", 100.0)),
        min_abs_force_exec=float(agent_raw.get("min_abs_force_exec", 0.0)),
        min_abs_force_ctx=float(agent_raw.get("min_abs_force_ctx", 0.0)),
        take_exec_cci_extreme_if_ctx_not_extreme=bool(agent_raw.get("take_exec_cci_extreme_if_ctx_not_extreme", False)),
        take_exec_and_ctx_cci_extreme=bool(agent_raw.get("take_exec_and_ctx_cci_extreme", False)),
        signal_on_ctx_flip_if_exec_aligned=bool(agent_raw.get("signal_on_ctx_flip_if_exec_aligned", False)),
    )

    bt_raw = _get(raw, ["backtest"], {})
    backtest = BacktestConfig(
        fee_rate=float(bt_raw.get("fee_rate", 0.0015)),
        exit_mode=str(bt_raw.get("exit_mode", "exec_tranche_end")),
        tp_pct=float(bt_raw.get("tp_pct", 0.0)),
        trailing_stop_pct=float(bt_raw.get("trailing_stop_pct", 0.0)),
        sl_pct=float(bt_raw.get("sl_pct", 0.0)),
        stoch_high=float(bt_raw.get("stoch_high", 80.0)),
        stoch_low=float(bt_raw.get("stoch_low", 20.0)),
        stoch_wait_extreme=bool(bt_raw.get("stoch_wait_extreme", True)),
        max_signals=int(bt_raw.get("max_signals", 0)),
    )

    out_raw = _get(raw, ["output"], {})
    output = OutputConfig(
        out_dir=str(out_raw.get("out_dir", "data/processed/backtests/macd_momentum_two_tf")),
        save_csv=bool(out_raw.get("save_csv", True)),
        print_top_reasons=int(out_raw.get("print_top_reasons", 10)),
    )

    cfg = FullConfig(bybit=bybit, indicators=indicators, agent=agent, backtest=backtest, output=output)

    exit_mode = str(cfg.backtest.exit_mode).strip().lower()
    allowed_exit_modes = {"exec_tranche_end", "opposite_signal", "eod", "signal", "tp_pct", "trailing_stop"}
    if exit_mode not in allowed_exit_modes:
        raise ValueError(f"Unexpected backtest.exit_mode: {cfg.backtest.exit_mode}")
    if exit_mode == "tp_pct" and float(cfg.backtest.tp_pct) <= 0.0:
        raise ValueError("backtest.exit_mode=tp_pct requires backtest.tp_pct > 0")
    if exit_mode == "trailing_stop" and float(cfg.backtest.trailing_stop_pct) <= 0.0:
        raise ValueError("backtest.exit_mode=trailing_stop requires backtest.trailing_stop_pct > 0")

    if float(cfg.backtest.sl_pct) < 0.0:
        raise ValueError("backtest.sl_pct must be >= 0")

    return cfg


def load_config_yaml(path: str | Path) -> FullConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping")
    return load_config_dict(raw)
