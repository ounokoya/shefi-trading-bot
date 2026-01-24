from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.agents.macd_momentum_two_tf_cycle_agent import MacdMomentumTwoTFConfig
from libs.backtest_macd_momentum_two_tf.config import load_config_dict, load_config_yaml
from libs.backtest_macd_momentum_two_tf.engine import BacktestMacdMomentumTwoTFConfig, run_backtest_macd_momentum_two_tf
from libs.blocks.add_macd_tv_columns_df import add_macd_tv_columns_df
from libs.indicators.momentum.cci_tv import cci_tv


def _deep_set(d: dict[str, Any], path: list[str], value: object) -> None:
    cur: Any = d
    for k in path[:-1]:
        if not isinstance(cur, dict):
            raise ValueError(f"Cannot set path {'.'.join(path)} on non-dict")
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    if not isinstance(cur, dict):
        raise ValueError(f"Cannot set path {'.'.join(path)} on non-dict")
    cur[path[-1]] = value


def _deep_get(d: dict[str, Any], path: list[str], default: object = None) -> object:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _apply_search_space(
    trial,
    *,
    base_cfg_dict: dict[str, Any],
    search_space: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, object]]:
    cfg_dict = copy.deepcopy(base_cfg_dict)
    chosen: dict[str, object] = {}

    def rec(space: object, path: list[str]) -> None:
        if not isinstance(space, dict):
            return

        if "suggest" in space:
            suggest = str(space.get("suggest"))
            name = ".".join(path)
            if suggest == "categorical":
                choices = list(space.get("choices") or [])
                val = trial.suggest_categorical(name, choices)
            elif suggest == "int":
                low = int(space.get("low"))
                high = int(space.get("high"))
                step_raw = space.get("step", 1)
                step = int(1 if step_raw is None else step_raw)
                val = int(trial.suggest_int(name, low, high, step=int(step)))
            elif suggest == "float":
                low = float(space.get("low"))
                high = float(space.get("high"))
                step = space.get("step")
                if step is None:
                    val = float(trial.suggest_float(name, low, high))
                else:
                    val = float(trial.suggest_float(name, low, high, step=float(step)))
            else:
                raise ValueError(f"Unsupported suggest type: {suggest} for {name}")

            chosen[name] = val
            _deep_set(cfg_dict, path, val)
            return

        for k, v in space.items():
            rec(v, path + [str(k)])

    rec(search_space, [])
    return cfg_dict, chosen


def _interval_to_bybit(interval: str) -> str:
    s = str(interval).strip()
    if not s:
        raise ValueError("interval cannot be empty")

    if s.isdigit():
        return s

    s_lower = s.lower()
    if s_lower in {"d", "1d"}:
        return "D"
    if s_lower in {"w", "1w"}:
        return "W"
    if s == "M" or s_lower in {"1mo", "mo", "1month"}:
        return "M"

    import re

    m = re.fullmatch(r"(\d+)([mhd])", s_lower)
    if not m:
        raise ValueError(f"unsupported interval format: {interval}")

    n = int(m.group(1))
    unit = m.group(2)
    if unit == "m":
        minutes = n
    elif unit == "h":
        minutes = n * 60
    elif unit == "d":
        if n != 1:
            raise ValueError(f"Bybit only supports daily interval as 1d/D, got: {interval}")
        return "D"
    else:
        raise ValueError(f"unsupported interval unit: {unit}")

    allowed_minutes = {1, 3, 5, 15, 30, 60, 120, 240, 360, 720}
    if minutes not in allowed_minutes:
        allowed_str = ", ".join(str(x) for x in sorted(allowed_minutes))
        raise ValueError(
            f"unsupported minute interval for Bybit: {minutes} (from {interval}). Allowed: {allowed_str}"
        )
    return str(minutes)


def _interval_to_minutes(interval: str) -> int:
    s = str(interval).strip()
    if not s:
        return 0
    if s.isdigit():
        return int(s)
    s_lower = s.lower()
    if s_lower.endswith("m") and s_lower[:-1].isdigit():
        return int(s_lower[:-1])
    if s_lower.endswith("h") and s_lower[:-1].isdigit():
        return int(s_lower[:-1]) * 60
    if s_lower in {"d", "1d"}:
        return 24 * 60
    return 0


def _fetch_bybit_klines_range(
    *,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    page_limit: int,
    category: str,
    base_url: str,
    timeout_s: float,
    max_pages: int = 250,
) -> pd.DataFrame:
    import requests

    url = f"{base_url.rstrip('/')}/v5/market/kline"

    if int(page_limit) <= 0:
        page_limit = 200
    if int(page_limit) > 1000:
        page_limit = 1000

    start = int(start_ms)
    cur_end = int(end_ms)
    if int(cur_end) < int(start):
        return pd.DataFrame([])

    seen: set[int] = set()
    out_rows: list[dict[str, object]] = []

    for _ in range(int(max_pages)):
        params = {
            "category": str(category),
            "symbol": str(symbol),
            "interval": str(interval),
            "limit": str(int(page_limit)),
            "start": str(int(start)),
            "end": str(int(cur_end)),
        }
        r = requests.get(url, params=params, timeout=float(timeout_s))
        r.raise_for_status()
        payload = r.json()
        if str(payload.get("retCode")) != "0":
            raise RuntimeError(f"Bybit retCode={payload.get('retCode')} retMsg={payload.get('retMsg')}")

        result = payload.get("result") or {}
        rows = result.get("list") or []
        if not rows:
            break

        ts_list: list[int] = []
        for row in rows:
            if not isinstance(row, list) or len(row) < 1:
                continue
            try:
                ts_list.append(int(row[0]))
            except Exception:
                continue
        if not ts_list:
            break

        oldest = int(min(ts_list))

        for row in rows:
            if not isinstance(row, list) or len(row) < 6:
                continue
            ts = int(row[0])
            if ts in seen:
                continue
            seen.add(ts)
            if int(start_ms) <= ts <= int(end_ms):
                out_rows.append(
                    {
                        "ts": ts,
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4]),
                        "volume": float(row[5]),
                    }
                )

        if int(oldest) <= int(start_ms):
            break
        nxt_end = int(oldest) - 1
        if int(nxt_end) >= int(cur_end):
            break
        cur_end = int(nxt_end)

    df = pd.DataFrame(out_rows)
    if len(df):
        df = df.sort_values("ts").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return df


def _add_ccis(df: pd.DataFrame, *, cci_fast: int, cci_medium: int, cci_slow: int) -> tuple[pd.DataFrame, str, str, str]:
    out = df.copy()
    high = pd.to_numeric(out["high"], errors="coerce").astype(float).tolist()
    low = pd.to_numeric(out["low"], errors="coerce").astype(float).tolist()
    close = pd.to_numeric(out["close"], errors="coerce").astype(float).tolist()

    cci_fast_col = f"cci_{int(cci_fast)}"
    cci_medium_col = f"cci_{int(cci_medium)}"
    cci_slow_col = f"cci_{int(cci_slow)}"

    out[cci_fast_col] = cci_tv(high, low, close, int(cci_fast))
    out[cci_medium_col] = cci_tv(high, low, close, int(cci_medium))
    out[cci_slow_col] = cci_tv(high, low, close, int(cci_slow))

    return out, cci_fast_col, cci_medium_col, cci_slow_col


def _add_stoch(df: pd.DataFrame, *, k_period: int, d_period: int) -> pd.DataFrame:
    import numpy as np

    out = df.copy()
    k_period2 = int(k_period)
    d_period2 = int(d_period)
    if int(k_period2) < 1:
        raise ValueError("stoch.k must be >= 1")
    if int(d_period2) < 1:
        raise ValueError("stoch.d must be >= 1")

    low_s = pd.to_numeric(out["low"], errors="coerce").astype(float)
    high_s = pd.to_numeric(out["high"], errors="coerce").astype(float)
    close_s = pd.to_numeric(out["close"], errors="coerce").astype(float)

    ll = low_s.rolling(window=k_period2, min_periods=k_period2).min()
    hh = high_s.rolling(window=k_period2, min_periods=k_period2).max()
    denom = (hh - ll).astype(float)
    numer = (close_s - ll).astype(float)

    k = 100.0 * (numer / denom.replace(0.0, np.nan))
    out["stoch_k"] = k
    out["stoch_d"] = out["stoch_k"].rolling(window=d_period2, min_periods=d_period2).mean()
    return out


def _space_max(space: dict[str, Any], path: list[str]) -> object | None:
    cur: Any = space
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    if not isinstance(cur, dict):
        return None
    if "suggest" not in cur:
        return None
    if "high" not in cur:
        return None
    return cur.get("high")


def _compute_warmup_bars(*, base_cfg_dict: dict[str, Any], search_space: dict[str, Any]) -> int:
    warmup0 = int(_deep_get(base_cfg_dict, ["bybit", "warmup_bars"], 0) or 0)
    if int(warmup0) > 0:
        return int(warmup0)

    def _max_int(base_path: list[str], space_path: list[str], default: int) -> int:
        base_v = int(_deep_get(base_cfg_dict, base_path, default) or default)
        space_v = _space_max(search_space, space_path)
        if space_v is None:
            return int(base_v)
        try:
            return int(max(int(base_v), int(space_v)))
        except Exception:
            return int(base_v)

    exec_cci_slow = _max_int(["indicators", "exec_cci", "slow"], ["indicators", "exec_cci", "slow"], 300)
    ctx_cci_slow = _max_int(["indicators", "ctx_cci", "slow"], ["indicators", "ctx_cci", "slow"], 300)

    exec_macd_slow = _max_int(["indicators", "exec_macd", "slow"], ["indicators", "exec_macd", "slow"], 26)
    ctx_macd_slow = _max_int(["indicators", "ctx_macd", "slow"], ["indicators", "ctx_macd", "slow"], 26)

    stoch_k = _max_int(["indicators", "stoch", "k"], ["indicators", "stoch", "k"], 14)

    return int(max(exec_cci_slow, ctx_cci_slow, int(exec_macd_slow) * 3, int(ctx_macd_slow) * 3, stoch_k, 50))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to Optuna YAML config")
    args = ap.parse_args()

    try:
        import optuna
    except Exception as e:
        raise RuntimeError(f"Optuna is required but could not be imported: {e}")

    opt_cfg_path = Path(str(args.config))
    raw = yaml.safe_load(opt_cfg_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Optuna config must be a mapping")

    base_path = raw.get("base_backtest_config")
    if not base_path:
        raise ValueError("Missing base_backtest_config")

    base_path_s = str(base_path)
    if base_path_s.startswith("/"):
        base_cfg_path = Path(base_path_s)
    else:
        cand_paths = [
            (opt_cfg_path.parent / base_path_s).resolve(),
            (PROJECT_ROOT / base_path_s).resolve(),
        ]
        base_cfg_path = None
        for p in cand_paths:
            if p.exists():
                base_cfg_path = p
                break
        if base_cfg_path is None:
            tried = "\n".join([f"- {p}" for p in cand_paths])
            raise FileNotFoundError(f"base_backtest_config not found: {base_path_s}\nTried:\n{tried}")

    base_cfg_dict = yaml.safe_load(base_cfg_path.read_text())
    if not isinstance(base_cfg_dict, dict):
        raise ValueError("Base backtest config must be a mapping")

    base_cfg = load_config_yaml(str(base_cfg_path))

    out_cfg = raw.get("output") or {}
    if not isinstance(out_cfg, dict):
        out_cfg = {}
    out_dir = Path(str(out_cfg.get("out_dir") or "data/processed/backtests/optuna_macd_momentum_two_tf_multi"))
    out_dir.mkdir(parents=True, exist_ok=True)

    optuna_cfg = raw.get("optuna") or {}
    if not isinstance(optuna_cfg, dict):
        optuna_cfg = {}

    study_name = str(optuna_cfg.get("study_name") or "macd_momentum_two_tf_multi")
    storage_raw = optuna_cfg.get("storage") or optuna_cfg.get("storage_url") or optuna_cfg.get("storage_uri")
    if storage_raw is None or str(storage_raw).strip() == "":
        db_path = (out_dir / f"{study_name}.db").resolve()
        storage = f"sqlite:///{db_path.as_posix()}"
    else:
        storage_s = str(storage_raw).strip()
        if "://" in storage_s:
            storage = storage_s
        else:
            db_path = Path(storage_s)
            if not db_path.is_absolute():
                db_path = (out_dir / db_path).resolve()
            storage = f"sqlite:///{db_path.as_posix()}"

    load_if_exists_raw = optuna_cfg.get("load_if_exists")
    load_if_exists = True if load_if_exists_raw is None else bool(load_if_exists_raw)
    n_trials = int(optuna_cfg.get("n_trials") or 100)
    sampler = str(optuna_cfg.get("sampler") or "tpe").strip().lower()
    seed = optuna_cfg.get("seed")
    seed_i = None if seed is None else int(seed)

    if sampler == "random":
        sam = optuna.samplers.RandomSampler(seed=seed_i)
    elif sampler == "tpe":
        sam = optuna.samplers.TPESampler(seed=seed_i)
    else:
        raise ValueError(f"Unsupported sampler: {sampler}")

    optuna_log_level = str(optuna_cfg.get("log_level") or optuna_cfg.get("verbosity") or "").strip().lower()
    if not optuna_log_level:
        optuna_log_level = "warning"
    _lvl_map = {
        "critical": optuna.logging.CRITICAL,
        "error": optuna.logging.ERROR,
        "warning": optuna.logging.WARNING,
        "info": optuna.logging.INFO,
        "debug": optuna.logging.DEBUG,
    }
    if optuna_log_level not in _lvl_map:
        raise ValueError(f"Unsupported optuna.log_level: {optuna_log_level}")
    optuna.logging.set_verbosity(_lvl_map[optuna_log_level])

    show_progress_bar_raw = optuna_cfg.get("show_progress_bar")
    show_progress_bar = True if show_progress_bar_raw is None else bool(show_progress_bar_raw)

    search_space = raw.get("search_space") or {}
    if not isinstance(search_space, dict):
        raise ValueError("search_space must be a mapping")

    constraints = raw.get("constraints") or {}
    if not isinstance(constraints, dict):
        constraints = {}
    min_trades = int(constraints.get("min_trades") or 1)

    warmup_bars = _compute_warmup_bars(base_cfg_dict=base_cfg_dict, search_space=search_space)

    start_ts = int(pd.Timestamp(base_cfg.bybit.start, tz="UTC").value // 1_000_000)
    end_ts = int(
        (pd.Timestamp(base_cfg.bybit.end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)).value
        // 1_000_000
    )

    exec_min = _interval_to_minutes(str(base_cfg.bybit.exec_interval))
    ctx_min = _interval_to_minutes(str(base_cfg.bybit.ctx_interval))
    exec_warmup_ms = int(warmup_bars) * int(exec_min) * 60_000 if int(exec_min) > 0 else 0
    ctx_warmup_ms = int(warmup_bars) * int(ctx_min) * 60_000 if int(ctx_min) > 0 else 0
    exec_fetch_start_ms = int(max(0, int(start_ts) - int(exec_warmup_ms)))
    ctx_fetch_start_ms = int(max(0, int(start_ts) - int(ctx_warmup_ms)))

    df_exec_base = _fetch_bybit_klines_range(
        symbol=str(base_cfg.bybit.symbol),
        interval=_interval_to_bybit(str(base_cfg.bybit.exec_interval)),
        start_ms=int(exec_fetch_start_ms),
        end_ms=int(end_ts),
        page_limit=int(base_cfg.bybit.exec_limit),
        category=str(base_cfg.bybit.category),
        base_url=str(base_cfg.bybit.base_url),
        timeout_s=30.0,
    )
    df_ctx_base = _fetch_bybit_klines_range(
        symbol=str(base_cfg.bybit.symbol),
        interval=_interval_to_bybit(str(base_cfg.bybit.ctx_interval)),
        start_ms=int(ctx_fetch_start_ms),
        end_ms=int(end_ts),
        page_limit=int(base_cfg.bybit.ctx_limit),
        category=str(base_cfg.bybit.category),
        base_url=str(base_cfg.bybit.base_url),
        timeout_s=30.0,
    )

    if not len(df_exec_base):
        raise RuntimeError("no exec klines fetched")
    if not len(df_ctx_base):
        raise RuntimeError("no ctx klines fetched")

    def _evaluate_cfg(cfg) -> dict[str, float]:
        df_exec = df_exec_base.copy()
        df_ctx = df_ctx_base.copy()

        df_exec = add_macd_tv_columns_df(
            df_exec,
            close_col="close",
            fast_period=int(cfg.indicators.exec_macd.fast),
            slow_period=int(cfg.indicators.exec_macd.slow),
            signal_period=int(cfg.indicators.exec_macd.signal),
        )
        df_ctx = add_macd_tv_columns_df(
            df_ctx,
            close_col="close",
            fast_period=int(cfg.indicators.ctx_macd.fast),
            slow_period=int(cfg.indicators.ctx_macd.slow),
            signal_period=int(cfg.indicators.ctx_macd.signal),
        )

        df_exec, cci_exec_fast_col, cci_exec_medium_col, cci_exec_slow_col = _add_ccis(
            df_exec,
            cci_fast=int(cfg.indicators.exec_cci.fast),
            cci_medium=int(cfg.indicators.exec_cci.medium),
            cci_slow=int(cfg.indicators.exec_cci.slow),
        )
        df_ctx, cci_ctx_fast_col, cci_ctx_medium_col, cci_ctx_slow_col = _add_ccis(
            df_ctx,
            cci_fast=int(cfg.indicators.ctx_cci.fast),
            cci_medium=int(cfg.indicators.ctx_cci.medium),
            cci_slow=int(cfg.indicators.ctx_cci.slow),
        )

        df_exec = _add_stoch(df_exec, k_period=int(cfg.indicators.stoch.k), d_period=int(cfg.indicators.stoch.d))

        agent_cfg = MacdMomentumTwoTFConfig(
            ts_col="ts",
            dt_col="dt",
            open_col="open",
            high_col="high",
            low_col="low",
            close_col="close",
            hist_col="macd_hist",
            cci_exec_fast_col=str(cci_exec_fast_col),
            cci_exec_medium_col=str(cci_exec_medium_col),
            cci_exec_slow_col=str(cci_exec_slow_col),
            cci_exec_fast_period=int(cfg.indicators.exec_cci.fast),
            cci_exec_medium_period=int(cfg.indicators.exec_cci.medium),
            cci_exec_slow_period=int(cfg.indicators.exec_cci.slow),
            cci_ctx_fast_col=str(cci_ctx_fast_col),
            cci_ctx_medium_col=str(cci_ctx_medium_col),
            cci_ctx_slow_col=str(cci_ctx_slow_col),
            cci_ctx_fast_period=int(cfg.indicators.ctx_cci.fast),
            cci_ctx_medium_period=int(cfg.indicators.ctx_cci.medium),
            cci_ctx_slow_period=int(cfg.indicators.ctx_cci.slow),
            min_abs_force_exec=float(cfg.agent.min_abs_force_exec),
            min_abs_force_ctx=float(cfg.agent.min_abs_force_ctx),
            cci_global_extreme_level_exec=float(cfg.agent.exec_cci_extreme),
            cci_global_extreme_level_ctx=float(cfg.agent.ctx_cci_extreme),
            take_exec_cci_extreme_if_ctx_not_extreme=bool(cfg.agent.take_exec_cci_extreme_if_ctx_not_extreme),
            take_exec_and_ctx_cci_extreme=bool(cfg.agent.take_exec_and_ctx_cci_extreme),
            signal_on_ctx_flip_if_exec_aligned=bool(cfg.agent.signal_on_ctx_flip_if_exec_aligned),
        )

        bt_cfg = BacktestMacdMomentumTwoTFConfig(
            fee_rate=float(cfg.backtest.fee_rate),
            exit_mode=str(cfg.backtest.exit_mode),
            tp_pct=float(cfg.backtest.tp_pct),
            trailing_stop_pct=float(cfg.backtest.trailing_stop_pct),
            sl_pct=float(cfg.backtest.sl_pct),
            stoch_high=float(cfg.backtest.stoch_high),
            stoch_low=float(cfg.backtest.stoch_low),
            stoch_wait_extreme=bool(cfg.backtest.stoch_wait_extreme),
        )

        res = run_backtest_macd_momentum_two_tf(
            df_exec=df_exec,
            df_ctx=df_ctx,
            agent_cfg=agent_cfg,
            bt_cfg=bt_cfg,
            start_ts=int(start_ts),
            end_ts=int(end_ts),
            max_signals=int(cfg.backtest.max_signals),
        )

        summary = dict(res.get("summary") or {})
        n_trades = int(summary.get("n_trades") or 0)
        pnl = float(summary.get("equity_end") or 0.0)
        eq_dd = abs(float(summary.get("max_dd") or 0.0))
        trade_dd = float(summary.get("dd_max_trade") or 0.0)
        dur_max = float(summary.get("duration_s_max") or 0.0)
        winrate = float(summary.get("winrate") or 0.0)

        if int(n_trades) < int(min_trades):
            pnl = float(pnl) - 1e6
            eq_dd = float(eq_dd) + 1e6
            trade_dd = float(trade_dd) + 1e6
            dur_max = float(dur_max) + 1e12
            winrate = 0.0

        return {
            "n_trades": float(n_trades),
            "pnl": float(pnl),
            "equity_dd": float(eq_dd),
            "trade_dd": float(trade_dd),
            "duration_s_max": float(dur_max),
            "winrate": float(winrate),
        }

    def objective(trial):
        cfg_dict, chosen = _apply_search_space(trial, base_cfg_dict=base_cfg_dict, search_space=search_space)

        cfg_dict.setdefault("output", {})
        if isinstance(cfg_dict.get("output"), dict):
            cfg_dict["output"]["save_csv"] = False
            cfg_dict["output"]["print_top_reasons"] = 0

        cfg = load_config_dict(cfg_dict)

        if int(cfg.indicators.exec_macd.fast) >= int(cfg.indicators.exec_macd.slow):
            raise optuna.TrialPruned()
        if int(cfg.indicators.ctx_macd.fast) >= int(cfg.indicators.ctx_macd.slow):
            raise optuna.TrialPruned()
        if not (int(cfg.indicators.exec_cci.fast) < int(cfg.indicators.exec_cci.medium) < int(cfg.indicators.exec_cci.slow)):
            raise optuna.TrialPruned()
        if not (int(cfg.indicators.ctx_cci.fast) < int(cfg.indicators.ctx_cci.medium) < int(cfg.indicators.ctx_cci.slow)):
            raise optuna.TrialPruned()

        exit_mode = str(cfg.backtest.exit_mode).strip().lower()
        if exit_mode == "tp_pct" and float(cfg.backtest.tp_pct) <= 0.0:
            raise optuna.TrialPruned()
        if exit_mode == "trailing_stop" and float(cfg.backtest.trailing_stop_pct) <= 0.0:
            raise optuna.TrialPruned()

        m = _evaluate_cfg(cfg)

        for k, v in chosen.items():
            trial.set_user_attr(f"param.{k}", v)

        trial.set_user_attr("n_trades", int(m["n_trades"]))
        trial.set_user_attr("pnl", float(m["pnl"]))
        trial.set_user_attr("equity_dd", float(m["equity_dd"]))
        trial.set_user_attr("trade_dd", float(m["trade_dd"]))
        trial.set_user_attr("duration_s_max", float(m["duration_s_max"]))
        trial.set_user_attr("winrate", float(m["winrate"]))

        return (
            float(m["pnl"]),
            float(m["equity_dd"]),
            float(m["trade_dd"]),
            float(m["duration_s_max"]),
            float(m["winrate"]),
        )

    directions = ["maximize", "minimize", "minimize", "minimize", "maximize"]
    study = optuna.create_study(
        study_name=study_name,
        directions=directions,
        sampler=sam,
        storage=storage,
        load_if_exists=bool(load_if_exists),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=bool(show_progress_bar))

    trials_rows: list[dict[str, object]] = []
    for t in study.trials:
        r: dict[str, object] = {
            "number": int(t.number),
            "state": str(t.state),
        }
        vals = list(t.values) if t.values is not None else []
        for i, v in enumerate(vals):
            r[f"value.{i}"] = None if v is None else float(v)
        for k, v in (t.params or {}).items():
            r[f"param.{k}"] = v
        for k, v in (t.user_attrs or {}).items():
            r[f"attr.{k}"] = v
        trials_rows.append(r)

    trials_df = pd.DataFrame(trials_rows).sort_values(["value.0"], ascending=False)
    trials_csv = out_dir / "optuna_trials.csv"
    trials_df.to_csv(trials_csv, index=False, float_format="%.10f")

    pareto = study.best_trials
    pareto_rows: list[dict[str, object]] = []
    for t in pareto:
        r = {
            "number": int(t.number),
            "pnl": float(t.values[0]),
            "equity_dd": float(t.values[1]),
            "trade_dd": float(t.values[2]),
            "duration_s_max": float(t.values[3]),
            "winrate": float(t.values[4]),
        }
        for k, v in (t.params or {}).items():
            r[f"param.{k}"] = v
        pareto_rows.append(r)
    pareto_df = pd.DataFrame(pareto_rows)
    pareto_csv = out_dir / "pareto_trials.csv"
    pareto_df.to_csv(pareto_csv, index=False, float_format="%.10f")

    best_by_pnl = None
    if len(pareto):
        best_by_pnl = max(pareto, key=lambda t: float(t.values[0]))

    best_cfg_path = None
    if best_by_pnl is not None:
        best_cfg_dict = copy.deepcopy(base_cfg_dict)
        for k, v in best_by_pnl.params.items():
            path = [p for p in str(k).split(".") if p]
            _deep_set(best_cfg_dict, path, v)
        best_cfg_path = out_dir / "best_by_pnl_backtest_config.yaml"
        best_cfg_path.write_text(yaml.safe_dump(best_cfg_dict, sort_keys=False))

    summary = {
        "study_name": str(study.study_name),
        "storage": str(storage),
        "load_if_exists": bool(load_if_exists),
        "n_trials": int(len(study.trials)),
        "trials_csv": str(trials_csv),
        "pareto_csv": str(pareto_csv),
        "best_by_pnl_config": (None if best_cfg_path is None else str(best_cfg_path)),
    }
    summary_path = out_dir / "optuna_summary.yaml"
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))

    print("Optuna MacdMomentumTwoTF multi-objective:")
    print(f"- study_name: {study.study_name}")
    print(f"- storage: {storage}")
    print(f"- trials: {len(study.trials)}")
    print(f"- pareto: {len(pareto)}")
    print(f"Wrote: {trials_csv}")
    print(f"Wrote: {pareto_csv}")
    if best_cfg_path is not None:
        print(f"Wrote: {best_cfg_path}")
    print(f"Wrote: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
