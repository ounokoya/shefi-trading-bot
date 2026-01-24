from __future__ import annotations

import argparse
import copy
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


def _load_dca_module() -> Any:
    mod_path = (PROJECT_ROOT / "scripts" / "40_backtest_one_side_dca_mean_distance.py").resolve()
    if not mod_path.exists():
        raise FileNotFoundError(f"DCA backtest script not found: {mod_path}")

    spec = importlib.util.spec_from_file_location("one_side_dca_mean_distance", str(mod_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to create module spec for DCA backtest")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _compute_metrics(*, equity_df: pd.DataFrame, trades_df: pd.DataFrame, initial_cap: float) -> dict[str, float]:
    if not isinstance(equity_df, pd.DataFrame) or equity_df.empty:
        equity_df = pd.DataFrame({"equity": [float(initial_cap)]})
    if not isinstance(trades_df, pd.DataFrame):
        trades_df = pd.DataFrame([])

    equity_s = pd.to_numeric(equity_df.get("equity", pd.Series([float(initial_cap)])), errors="coerce").fillna(
        float(initial_cap)
    )
    final_equity = float(equity_s.iloc[-1]) if len(equity_s) else float(initial_cap)

    running_max = equity_s.cummax()
    denom = running_max.replace(0.0, pd.NA)
    dd = ((equity_s - running_max) / denom).fillna(0.0)
    max_dd = float(abs(dd.min())) if len(dd) else 0.0

    is_liq = bool(equity_df.get("is_liquidated", pd.Series([False])).fillna(False).any())
    liq_flag = 1.0 if is_liq else 0.0

    lev_cap_s = pd.to_numeric(
        equity_df.get("lev_eff_capital", pd.Series([0.0])),
        errors="coerce",
    ).fillna(0.0)
    exposure = float(lev_cap_s.max()) if not lev_cap_s.empty else 0.0

    net_pnl = float(final_equity) - float(initial_cap)

    n_trades = float(len(trades_df))

    return {
        "liquidated": float(liq_flag),
        "max_dd": float(max_dd),
        "exposure": float(exposure),
        "net_pnl": float(net_pnl),
        "n_trades": float(n_trades),
    }


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

    out_cfg = raw.get("output") or {}
    if not isinstance(out_cfg, dict):
        out_cfg = {}
    out_dir = Path(str(out_cfg.get("out_dir") or "data/processed/backtests/optuna_one_side_dca_mean_distance_multi"))
    out_dir.mkdir(parents=True, exist_ok=True)

    optuna_cfg = raw.get("optuna") or {}
    if not isinstance(optuna_cfg, dict):
        optuna_cfg = {}

    study_name = str(optuna_cfg.get("study_name") or "one_side_dca_mean_distance_multi")

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
    sampler = str(optuna_cfg.get("sampler") or "nsgaii").strip().lower()
    seed = optuna_cfg.get("seed")
    seed_i = None if seed is None else int(seed)

    if sampler == "random":
        sam = optuna.samplers.RandomSampler(seed=seed_i)
    elif sampler == "tpe":
        sam = optuna.samplers.TPESampler(seed=seed_i)
    elif sampler in {"nsga", "nsgaii", "nsga2"}:
        sam = optuna.samplers.NSGAIISampler(seed=seed_i)
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

    # Reduce log noise from repeated backtests
    logging.getLogger().setLevel(logging.WARNING)

    dca_mod = _load_dca_module()

    tmp_cfg_path = (out_dir / "_trial_backtest_config.yaml").resolve()

    def objective(trial):
        cfg_dict, chosen = _apply_search_space(trial, base_cfg_dict=base_cfg_dict, search_space=search_space)

        # Avoid side effects in backtest main (we call run_backtest directly anyway)
        cfg_dict.setdefault("output", {})
        if isinstance(cfg_dict.get("output"), dict):
            cfg_dict["output"]["png"] = False

        tmp_cfg_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False))

        try:
            cfg = dca_mod.load_one_side_dca_config_yaml(str(tmp_cfg_path))
            res = dca_mod.run_backtest(cfg)
            equity_df: pd.DataFrame = res.get("equity")
            trades_df: pd.DataFrame = res.get("trades")

            m = _compute_metrics(equity_df=equity_df, trades_df=trades_df, initial_cap=float(cfg.capital_usdt))
        except Exception:
            # Hard-fail trials as very bad solutions
            m = {
                "liquidated": 1.0,
                "max_dd": 1e6,
                "exposure": 1e6,
                "net_pnl": -1e6,
                "n_trades": 0.0,
            }

        if int(m["n_trades"]) < int(min_trades):
            m["liquidated"] = 1.0
            m["max_dd"] = float(m["max_dd"]) + 1e3
            m["exposure"] = float(m["exposure"]) + 1e3
            m["net_pnl"] = float(m["net_pnl"]) - 1e3

        for k, v in chosen.items():
            trial.set_user_attr(f"param.{k}", v)

        trial.set_user_attr("liquidated", float(m["liquidated"]))
        trial.set_user_attr("max_dd", float(m["max_dd"]))
        trial.set_user_attr("exposure", float(m["exposure"]))
        trial.set_user_attr("net_pnl", float(m["net_pnl"]))
        trial.set_user_attr("n_trades", int(m["n_trades"]))

        # Objectives:
        # 0) minimize liquidation flag (0 best)
        # 1) minimize max drawdown ratio
        # 2) minimize exposure (max lev / capital)
        # 3) maximize net pnl (USDT)
        return (float(m["liquidated"]), float(m["max_dd"]), float(m["exposure"]), float(m["net_pnl"]))

    directions = ["minimize", "minimize", "minimize", "maximize"]
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

    trials_df = pd.DataFrame(trials_rows)
    trials_csv = out_dir / "optuna_trials.csv"
    trials_df.to_csv(trials_csv, index=False, float_format="%.10f")

    pareto = study.best_trials
    pareto_rows: list[dict[str, object]] = []
    for t in pareto:
        r = {
            "number": int(t.number),
            "liquidated": float(t.values[0]),
            "max_dd": float(t.values[1]),
            "exposure": float(t.values[2]),
            "net_pnl": float(t.values[3]),
        }
        for k, v in (t.params or {}).items():
            r[f"param.{k}"] = v
        pareto_rows.append(r)
    pareto_df = pd.DataFrame(pareto_rows)
    pareto_csv = out_dir / "pareto_trials.csv"
    pareto_df.to_csv(pareto_csv, index=False, float_format="%.10f")

    best_trial = None
    if len(pareto):
        zero_liq = [t for t in pareto if float(t.values[0]) <= 0.0]
        pool = zero_liq if zero_liq else pareto
        best_trial = max(pool, key=lambda t: float(t.values[3]))

    best_cfg_path = None
    if best_trial is not None:
        best_cfg_dict = copy.deepcopy(base_cfg_dict)
        for k, v in best_trial.params.items():
            path = [p for p in str(k).split(".") if p]
            _deep_set(best_cfg_dict, path, v)
        best_cfg_path = out_dir / "best_backtest_config.yaml"
        best_cfg_path.write_text(yaml.safe_dump(best_cfg_dict, sort_keys=False))

    summary = {
        "study_name": str(study.study_name),
        "storage": str(storage),
        "load_if_exists": bool(load_if_exists),
        "n_trials": int(len(study.trials)),
        "trials_csv": str(trials_csv),
        "pareto_csv": str(pareto_csv),
        "best_backtest_config": (None if best_cfg_path is None else str(best_cfg_path)),
    }
    summary_path = out_dir / "optuna_summary.yaml"
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))

    print("Optuna one-side DCA mean-distance multi-objective:")
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
