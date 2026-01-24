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
    interval: str = "15m"
    limit: int = 1000
    start: str = "2026-01-01"
    end: str = "2026-01-12"
    warmup_bars: int = 0


@dataclass(frozen=True)
class MacdConfig:
    fast: int = 12
    slow: int = 26
    signal: int = 9


@dataclass(frozen=True)
class IndicatorsConfig:
    macd_fast: MacdConfig = MacdConfig()
    macd_medium: MacdConfig = MacdConfig(fast=24, slow=52, signal=18)
    macd_slow: MacdConfig = MacdConfig(fast=48, slow=104, signal=36)


@dataclass(frozen=True)
class AgentRoleConfig:
    enabled: bool = True
    reject_zone_transition: bool = False
    min_abs_force: float = 0.0
    require_force_rising: bool = False
    force_rising_bars: int = 2
    allow_trade_when_respiration: bool = True
    require_align_zone_to_macro: bool = False
    require_align_hist_to_macro: bool = False


@dataclass(frozen=True)
class AgentConfig:
    hist_zero_policy: str = "carry_prev_sign"
    require_hists_rising_on_entry: bool = False

    slow: AgentRoleConfig = AgentRoleConfig()
    medium: AgentRoleConfig = AgentRoleConfig()
    fast: AgentRoleConfig = AgentRoleConfig()


@dataclass(frozen=True)
class BacktestConfig:
    fee_rate: float = 0.0015
    use_net: bool = True

    exit_mode: str = "opposite_signal"

    tp_pct: float = 0.0
    trailing_stop_pct: float = 0.0
    sl_pct: float = 0.0

    entry_on_next_bar: bool = True
    max_signals: int = 0


@dataclass(frozen=True)
class OutputConfig:
    out_dir: str = "data/processed/backtests/triple_macd_simple_alignment_yaml"
    save_csv: bool = True


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
        interval=str(bybit_raw.get("interval", "15m")),
        limit=int(bybit_raw.get("limit", 1000)),
        start=str(bybit_raw.get("start", "2026-01-01")),
        end=str(bybit_raw.get("end", "2026-01-12")),
        warmup_bars=int(bybit_raw.get("warmup_bars", 0)),
    )

    ind_raw = _get(raw, ["indicators"], {})

    def _macd(path: str, *, default_fast: int, default_slow: int, default_signal: int) -> MacdConfig:
        m = _get(ind_raw, [path], {})
        return MacdConfig(
            fast=int(m.get("fast", default_fast)),
            slow=int(m.get("slow", default_slow)),
            signal=int(m.get("signal", default_signal)),
        )

    indicators = IndicatorsConfig(
        macd_fast=_macd("macd_fast", default_fast=12, default_slow=26, default_signal=9),
        macd_medium=_macd("macd_medium", default_fast=24, default_slow=52, default_signal=18),
        macd_slow=_macd("macd_slow", default_fast=48, default_slow=104, default_signal=36),
    )

    agent_raw = _get(raw, ["agent"], {})

    def _role(path: str) -> AgentRoleConfig:
        d = _get(agent_raw, [path], {})
        if not isinstance(d, dict):
            d = {}
        return AgentRoleConfig(
            enabled=bool(d.get("enabled", True)),
            reject_zone_transition=bool(d.get("reject_zone_transition", False)),
            min_abs_force=float(d.get("min_abs_force", 0.0)),
            require_force_rising=bool(d.get("require_force_rising", False)),
            force_rising_bars=int(d.get("force_rising_bars", 2)),
            allow_trade_when_respiration=bool(d.get("allow_trade_when_respiration", True)),
            require_align_zone_to_macro=bool(d.get("require_align_zone_to_macro", False)),
            require_align_hist_to_macro=bool(d.get("require_align_hist_to_macro", False)),
        )

    agent = AgentConfig(
        hist_zero_policy=str(agent_raw.get("hist_zero_policy", "carry_prev_sign")),
        require_hists_rising_on_entry=bool(agent_raw.get("require_hists_rising_on_entry", False)),
        slow=_role("slow"),
        medium=_role("medium"),
        fast=_role("fast"),
    )

    bt_raw = _get(raw, ["backtest"], {})
    backtest = BacktestConfig(
        fee_rate=float(bt_raw.get("fee_rate", 0.0015)),
        use_net=bool(bt_raw.get("use_net", True)),
        exit_mode=str(bt_raw.get("exit_mode", "opposite_signal")),
        tp_pct=float(bt_raw.get("tp_pct", 0.0)),
        trailing_stop_pct=float(bt_raw.get("trailing_stop_pct", 0.0)),
        sl_pct=float(bt_raw.get("sl_pct", 0.0)),
        entry_on_next_bar=bool(bt_raw.get("entry_on_next_bar", True)),
        max_signals=int(bt_raw.get("max_signals", 0)),
    )

    out_raw = _get(raw, ["output"], {})
    output = OutputConfig(
        out_dir=str(out_raw.get("out_dir", "data/processed/backtests/triple_macd_simple_alignment_yaml")),
        save_csv=bool(out_raw.get("save_csv", True)),
    )

    cfg = FullConfig(bybit=bybit, indicators=indicators, agent=agent, backtest=backtest, output=output)

    exit_mode = str(cfg.backtest.exit_mode).strip().lower()
    allowed_exit_modes = {"opposite_signal", "eod", "tp_pct", "trailing_stop"}
    if exit_mode not in allowed_exit_modes:
        raise ValueError(f"Unexpected backtest.exit_mode: {cfg.backtest.exit_mode}")
    if exit_mode == "tp_pct" and float(cfg.backtest.tp_pct) <= 0.0:
        raise ValueError("backtest.exit_mode=tp_pct requires backtest.tp_pct > 0")
    if exit_mode == "trailing_stop" and float(cfg.backtest.trailing_stop_pct) <= 0.0:
        raise ValueError("backtest.exit_mode=trailing_stop requires backtest.trailing_stop_pct > 0")

    if float(cfg.backtest.sl_pct) < 0.0:
        raise ValueError("backtest.sl_pct must be >= 0")

    for name, m in (
        ("macd_fast", cfg.indicators.macd_fast),
        ("macd_medium", cfg.indicators.macd_medium),
        ("macd_slow", cfg.indicators.macd_slow),
    ):
        if int(m.fast) < 1 or int(m.slow) < 1 or int(m.signal) < 1:
            raise ValueError(f"indicators.{name}.fast/slow/signal must be >= 1")

    zp = str(cfg.agent.hist_zero_policy).strip().lower()
    if zp not in {"carry_prev_sign", "neutral_zero"}:
        raise ValueError(f"Unexpected agent.hist_zero_policy: {cfg.agent.hist_zero_policy}")

    for name, lv in (("slow", cfg.agent.slow), ("medium", cfg.agent.medium), ("fast", cfg.agent.fast)):
        if float(lv.min_abs_force) < 0.0:
            raise ValueError(f"agent.{name}.min_abs_force must be >= 0")
        if int(lv.force_rising_bars) < 2:
            raise ValueError(f"agent.{name}.force_rising_bars must be >= 2")

    return cfg


def load_config_yaml(path: str | Path) -> FullConfig:
    p = Path(path)
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping")
    return load_config_dict(raw)
