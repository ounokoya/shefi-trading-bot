from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.backtest.config import load_config_dict, load_config_yaml
from libs.backtest.engine import run_backtest_from_config
from libs.backtest.indicators import ensure_indicators_df


def _deep_set(d: dict, path: list[str], value: object) -> None:
    cur: object = d
    for k in path[:-1]:
        if not isinstance(cur, dict):
            raise ValueError(f"Cannot set path {'.'.join(path)} on non-dict")
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    if not isinstance(cur, dict):
        raise ValueError(f"Cannot set path {'.'.join(path)} on non-dict")
    cur[path[-1]] = value


def _deep_get(d: dict, path: list[str], default: object = None) -> object:
    cur: object = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _apply_series_profiles(cfg_dict: dict[str, object]) -> None:
    profiles: dict[str, dict[str, list[str]]] = {
        "full": {
            "add": [
                "close",
                "macd_line",
                "macd_hist",
                "cci_30",
                "cci_120",
                "cci_300",
                "vwma_4",
                "vwma_12",
            ],
            "exclude": [],
        },
        "no_vwma_no_cci_slow": {
            "add": [
                "close",
                "macd_line",
                "macd_hist",
                "cci_30",
                "cci_120",
            ],
            "exclude": [
                "vwma_4",
                "vwma_12",
                "cci_300",
            ],
        },
    }

    for section in ("entry", "exit"):
        prof = _deep_get(cfg_dict, ["signals", section, "params", "series_profile"], None)
        if prof is None:
            continue
        prof_s = str(prof).strip()
        if not prof_s:
            continue
        if prof_s not in profiles:
            raise ValueError(f"Unknown signals.{section}.params.series_profile: {prof_s}")

        _deep_set(cfg_dict, ["signals", section, "params", "series", "add"], list(profiles[prof_s]["add"]))
        _deep_set(cfg_dict, ["signals", section, "params", "series", "exclude"], list(profiles[prof_s]["exclude"]))


def _apply_series_flags(cfg_dict: dict[str, object]) -> None:
    candidates = [
        "close",
        "macd_line",
        "macd_hist",
        "cci_fast",
        "cci_medium",
        "cci_slow",
        "vwma_fast",
        "vwma_medium",
    ]

    for section in ("entry", "exit"):
        flags = _deep_get(cfg_dict, ["signals", section, "params", "series_flags"], None)
        if flags is None:
            continue
        if not isinstance(flags, dict):
            raise ValueError(f"signals.{section}.params.series_flags must be a mapping")

        enabled: dict[str, bool] = {}
        for c in candidates:
            if c in flags:
                enabled[c] = bool(flags.get(c))
            else:
                enabled[c] = True

        # keep at least close to avoid empty series
        enabled["close"] = True

        add = [c for c in candidates if enabled.get(c, False)]
        exclude = [c for c in candidates if not enabled.get(c, False)]

        _deep_set(cfg_dict, ["signals", section, "params", "series", "add"], add)
        _deep_set(cfg_dict, ["signals", section, "params", "series", "exclude"], exclude)

        # Ensure min_confirmed is feasible when the series set is optimized.
        mc = _deep_get(cfg_dict, ["signals", section, "params", "min_confirmed"], None)
        if mc is not None:
            mc_i = int(mc)
            if mc_i < 1:
                mc_i = 1
            if mc_i > len(add):
                mc_i = len(add)
            _deep_set(cfg_dict, ["signals", section, "params", "min_confirmed"], mc_i)


def _score_with_penalties(
    *,
    summary: dict[str, object],
    target_max_dd: float,
    target_min_trades: int,
    dd_penalty_weight: float,
    trades_penalty_weight: float,
    neg_equity_penalty_weight: float,
) -> dict[str, float]:
    equity_end = float(summary.get("equity_end") or 0.0)
    max_dd = float(summary.get("max_dd") or 0.0)
    n_trades = int(summary.get("n_trades") or 0)

    dd_excess = max(0.0, abs(float(max_dd)) - float(target_max_dd))
    dd_penalty = float(dd_penalty_weight) * float(dd_excess)

    trades_shortfall = max(0, int(target_min_trades) - int(n_trades))
    trades_penalty = float(trades_penalty_weight) * (
        float(trades_shortfall) / float(max(1, int(target_min_trades)))
    )

    neg_equity_penalty = float(neg_equity_penalty_weight) * max(0.0, -float(equity_end))

    score = float(equity_end) - float(dd_penalty) - float(trades_penalty) - float(neg_equity_penalty)

    return {
        "score": float(score),
        "dd_penalty": float(dd_penalty),
        "trades_penalty": float(trades_penalty),
        "neg_equity_penalty": float(neg_equity_penalty),
    }


def _walk_forward_score_rows(
    *,
    cfg,
    df: pd.DataFrame,
    start_ts: int | None,
    end_ts: int | None,
    fold_count: int,
    train_pct: float,
    test_pct: float,
    step: str,
    score_cfg: dict[str, object],
) -> tuple[float, pd.DataFrame]:
    ts_col = cfg.data.ts_col
    ts = pd.to_numeric(df[ts_col], errors="coerce").astype("Int64")

    start_ts_eff = int(ts.dropna().iloc[0]) if start_ts is None else int(start_ts)
    end_ts_eff = int(ts.dropna().iloc[-1]) if end_ts is None else int(end_ts)

    mask = (ts >= start_ts_eff) & (ts <= end_ts_eff)
    df_r = df.loc[mask].reset_index(drop=True)
    if len(df_r) < 10:
        return 0.0, pd.DataFrame([])

    n = int(len(df_r))
    fc = int(max(1, int(fold_count)))
    fold_size = int(n // fc)
    if fold_size < 2:
        fold_size = 2

    tp = float(train_pct)
    tep = float(test_pct)
    tot = float(tp + tep)
    tp /= tot
    tep /= tot

    train_size = int(max(1, min(fold_size - 1, int(round(float(fold_size) * float(tp))))))
    test_size = int(fold_size - train_size)

    step_eff = str(step).strip().lower() if step is not None else "test"
    if step_eff in {"test", "test_size"}:
        step_size = int(test_size)
    elif step_eff in {"fold", "fold_size"}:
        step_size = int(fold_size)
    else:
        raise ValueError(f"Unexpected walk_forward.step: {step}")

    target_min_trades = int(score_cfg.get("target_min_trades") or 50)
    target_max_dd = float(score_cfg.get("target_max_dd") or 0.15)
    dd_w = float(score_cfg.get("dd_penalty_weight") or 1.0)
    trades_w = float(score_cfg.get("trades_penalty_weight") or 0.02)
    neg_w = float(score_cfg.get("neg_equity_penalty_weight") or 1.0)

    rows: list[dict[str, object]] = []
    score_total = 0.0

    fold_i = 0
    start_i = 0
    while True:
        train_start_i = int(start_i)
        train_end_i = int(train_start_i + train_size - 1)
        test_start_i = int(train_end_i + 1)
        test_end_i = int(test_start_i + test_size - 1)

        if test_end_i >= n:
            break

        train_start_ts = int(df_r[ts_col].iloc[train_start_i])
        train_end_ts = int(df_r[ts_col].iloc[train_end_i])
        test_start_ts = int(df_r[ts_col].iloc[test_start_i])
        test_end_ts = int(df_r[ts_col].iloc[test_end_i])

        train_res = run_backtest_from_config(
            cfg=cfg,
            df=df_r,
            start_ts=train_start_ts,
            end_ts=train_end_ts,
            ensure_indicators=False,
        )
        test_res = run_backtest_from_config(
            cfg=cfg,
            df=df_r,
            start_ts=test_start_ts,
            end_ts=test_end_ts,
            ensure_indicators=False,
        )

        train_sum = dict(train_res.get("summary") or {})
        test_sum = dict(test_res.get("summary") or {})

        test_score = _score_with_penalties(
            summary=test_sum,
            target_max_dd=target_max_dd,
            target_min_trades=target_min_trades,
            dd_penalty_weight=dd_w,
            trades_penalty_weight=trades_w,
            neg_equity_penalty_weight=neg_w,
        )
        score_total += float(test_score["score"])

        rows.append(
            {
                "fold": int(fold_i),
                "train_start_ts": int(train_start_ts),
                "train_end_ts": int(train_end_ts),
                "test_start_ts": int(test_start_ts),
                "test_end_ts": int(test_end_ts),
                "train_n_trades": int(train_sum.get("n_trades") or 0),
                "train_n_wins": int(train_sum.get("n_wins") or 0),
                "train_n_losses": int(train_sum.get("n_losses") or 0),
                "train_winrate": float(train_sum.get("winrate") or 0.0),
                "train_avg_win": float(train_sum.get("avg_win") or 0.0),
                "train_avg_loss": float(train_sum.get("avg_loss") or 0.0),
                "train_equity_end": float(train_sum.get("equity_end") or 0.0),
                "train_max_dd": float(train_sum.get("max_dd") or 0.0),
                "test_n_trades": int(test_sum.get("n_trades") or 0),
                "test_n_wins": int(test_sum.get("n_wins") or 0),
                "test_n_losses": int(test_sum.get("n_losses") or 0),
                "test_winrate": float(test_sum.get("winrate") or 0.0),
                "test_avg_win": float(test_sum.get("avg_win") or 0.0),
                "test_avg_loss": float(test_sum.get("avg_loss") or 0.0),
                "test_equity_end": float(test_sum.get("equity_end") or 0.0),
                "test_max_dd": float(test_sum.get("max_dd") or 0.0),
                "test_score": float(test_score["score"]),
                "test_dd_penalty": float(test_score["dd_penalty"]),
                "test_trades_penalty": float(test_score["trades_penalty"]),
                "test_neg_equity_penalty": float(test_score["neg_equity_penalty"]),
            }
        )

        fold_i += 1
        start_i = int(start_i + step_size)

    return float(score_total), pd.DataFrame(rows)


def _apply_search_space(trial, *, base_cfg_dict: dict[str, object], search_space: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to optuna YAML config")
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
    out_dir = Path(str(out_cfg.get("out_dir") or "data/processed/backtests/optuna_walk_forward"))
    out_dir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(Path(base_cfg.data.csv))

    wf = raw.get("walk_forward") or {}
    if not isinstance(wf, dict):
        wf = {}

    fold_count = int(wf.get("fold_count") or 20)
    train_pct = float(wf.get("train_pct") or 0.70)
    test_pct = float(wf.get("test_pct") or 0.30)
    step = str(wf.get("step") or "test")
    start_ts = wf.get("start_ts")
    end_ts = wf.get("end_ts")

    score_cfg = wf.get("score") or {}
    if not isinstance(score_cfg, dict):
        score_cfg = {}

    optuna_cfg = raw.get("optuna") or {}
    if not isinstance(optuna_cfg, dict):
        optuna_cfg = {}

    study_name = str(optuna_cfg.get("study_name") or "walk_forward_opt")
    direction = str(optuna_cfg.get("direction") or "maximize")
    n_trials = int(optuna_cfg.get("n_trials") or 50)
    sampler = str(optuna_cfg.get("sampler") or "tpe").strip().lower()
    seed = optuna_cfg.get("seed")
    seed_i = None if seed is None else int(seed)

    if sampler == "random":
        sam = optuna.samplers.RandomSampler(seed=seed_i)
    elif sampler == "tpe":
        sam = optuna.samplers.TPESampler(seed=seed_i)
    else:
        raise ValueError(f"Unsupported sampler: {sampler}")

    search_space = raw.get("search_space") or {}
    if not isinstance(search_space, dict):
        raise ValueError("search_space must be a mapping")

    print_trial_log = bool(out_cfg.get("print_trial_log", True))

    optuna_log_level = str(optuna_cfg.get("log_level") or optuna_cfg.get("verbosity") or "").strip().lower()
    if not optuna_log_level:
        optuna_log_level = "warning" if print_trial_log else "info"
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

    def objective(trial) -> float:
        cfg_dict, chosen = _apply_search_space(trial, base_cfg_dict=base_cfg_dict, search_space=search_space)

        cfg_dict.setdefault("output", {})
        if isinstance(cfg_dict.get("output"), dict):
            cfg_dict["output"]["png"] = False

        _apply_series_profiles(cfg_dict)
        _apply_series_flags(cfg_dict)
        cfg = load_config_dict(cfg_dict)

        df_trial = ensure_indicators_df(df_raw, cfg=cfg, force=True)

        score, folds_df = _walk_forward_score_rows(
            cfg=cfg,
            df=df_trial,
            start_ts=(None if start_ts is None else int(start_ts)),
            end_ts=(None if end_ts is None else int(end_ts)),
            fold_count=fold_count,
            train_pct=train_pct,
            test_pct=test_pct,
            step=step,
            score_cfg=score_cfg,
        )

        trial.set_user_attr("folds", int(len(folds_df)))
        if len(folds_df):
            test_equity_end_sum = float(folds_df.get("test_equity_end", pd.Series([0.0])).sum())
            test_max_dd_min = float(folds_df.get("test_max_dd", pd.Series([0.0])).min())

            test_n_trades_sum = int(folds_df.get("test_n_trades", pd.Series([0])).sum())
            test_n_wins_sum = int(folds_df.get("test_n_wins", pd.Series([0])).sum())
            test_n_losses_sum = int(folds_df.get("test_n_losses", pd.Series([0])).sum())
            test_winrate_overall = float(test_n_wins_sum / test_n_trades_sum) if test_n_trades_sum > 0 else 0.0

            trial.set_user_attr("test_equity_end_sum", test_equity_end_sum)
            trial.set_user_attr("test_max_dd_min", test_max_dd_min)
            trial.set_user_attr("test_n_trades_sum", test_n_trades_sum)
            trial.set_user_attr("test_winrate_overall", test_winrate_overall)

            if print_trial_log:
                print(
                    (
                        f"trial={trial.number} value={float(score):.6f} "
                        f"test_equity_sum={test_equity_end_sum:.6f} "
                        f"test_max_dd_min={test_max_dd_min:.6f} "
                        f"test_trades={test_n_trades_sum} test_winrate={test_winrate_overall:.4f}"
                    ),
                    flush=True,
                )
        elif print_trial_log:
            print(f"trial={trial.number} score={float(score):.6f} folds=0", flush=True)

        return float(score)

    study = optuna.create_study(study_name=study_name, direction=direction, sampler=sam)
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial

    trials_rows: list[dict[str, object]] = []
    for t in study.trials:
        r: dict[str, object] = {
            "number": int(t.number),
            "value": None if t.value is None else float(t.value),
        }
        for k, v in (t.params or {}).items():
            r[f"param.{k}"] = v
        for k, v in (t.user_attrs or {}).items():
            r[f"attr.{k}"] = v
        trials_rows.append(r)

    trials_df = pd.DataFrame(trials_rows).sort_values("value", ascending=(direction != "maximize"))
    trials_csv = out_dir / "optuna_trials.csv"
    trials_df.to_csv(trials_csv, index=False, float_format="%.10f")

    best_cfg_dict = copy.deepcopy(base_cfg_dict)
    for k, v in best.params.items():
        path = [p for p in str(k).split(".") if p]
        _deep_set(best_cfg_dict, path, v)
    _apply_series_profiles(best_cfg_dict)
    _apply_series_flags(best_cfg_dict)
    best_cfg_dict.setdefault("output", {})
    if isinstance(best_cfg_dict.get("output"), dict):
        best_cfg_dict["output"]["png"] = True

    best_cfg_path = out_dir / "best_backtest_config.yaml"
    best_cfg_path.write_text(yaml.safe_dump(best_cfg_dict, sort_keys=False))

    best_cfg = load_config_dict(best_cfg_dict)
    df_best = ensure_indicators_df(df_raw, cfg=best_cfg, force=True)
    best_score, best_folds_df = _walk_forward_score_rows(
        cfg=best_cfg,
        df=df_best,
        start_ts=(None if start_ts is None else int(start_ts)),
        end_ts=(None if end_ts is None else int(end_ts)),
        fold_count=fold_count,
        train_pct=train_pct,
        test_pct=test_pct,
        step=step,
        score_cfg=score_cfg,
    )
    best_folds_path = out_dir / "best_walk_forward_folds.csv"
    best_folds_df.to_csv(best_folds_path, index=False, float_format="%.6f")

    summary = {
        "best_value": float(best.value) if best.value is not None else None,
        "best_score_recomputed": float(best_score),
        "best_params": dict(best.params),
        "trials_csv": str(trials_csv),
        "best_backtest_config": str(best_cfg_path),
        "best_folds_csv": str(best_folds_path),
    }
    summary_path = out_dir / "optuna_summary.yaml"
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))

    print("Optuna walk-forward optimization:")
    print(f"- study_name: {study.study_name}")
    print(f"- best_value: {best.value}")
    print(f"- best_score_recomputed: {best_score}")
    print(f"Wrote: {trials_csv}")
    print(f"Wrote: {best_cfg_path}")
    print(f"Wrote: {best_folds_path}")
    print(f"Wrote: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
