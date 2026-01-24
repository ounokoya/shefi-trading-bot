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

from libs.data_loader import fetch_bybit_klines_range
from libs.new_strategie.backtest_config import load_config_dict
from libs.new_strategie.backtest_flip import run_backtest_flip
from libs.new_strategie.config import NewStrategieConfig
from libs.new_strategie.indicators import ensure_indicators_df
from libs.new_strategie.pivots import build_top_pivots
from libs.new_strategie.signals import find_signals


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


def _tf_minutes(tf: str) -> int:
    t = str(tf or "").strip().lower()
    if t.endswith("m") and t[:-1].isdigit():
        return int(t[:-1])
    if t.endswith("min") and t[:-3].isdigit():
        return int(t[:-3])
    if t.endswith("h") and t[:-1].isdigit():
        return int(t[:-1]) * 60
    raise ValueError(f"Unsupported tf: {tf!r}")


def _year_bounds(year: int) -> tuple[str, str]:
    y = int(year)
    return (f"{y:04d}-01-01", f"{y:04d}-12-31")


def _warmup_bars_from_search_space(base_cfg_dict: dict[str, Any], search_space: dict[str, Any]) -> int:
    def _get_int(path: list[str], default: int) -> int:
        cur: Any = base_cfg_dict
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return int(default)
            cur = cur[k]
        try:
            return int(cur)
        except Exception:
            return int(default)

    def _get_high(space_path: list[str], default: int) -> int:
        cur: Any = search_space
        for k in space_path:
            if not isinstance(cur, dict) or k not in cur:
                return int(default)
            cur = cur[k]
        if isinstance(cur, dict) and str(cur.get("suggest")) == "int":
            try:
                return int(cur.get("high"))
            except Exception:
                return int(default)
        return int(default)

    macd_slow = int(_get_int(["indicators", "macd", "slow"], 26))
    macd_signal = int(_get_int(["indicators", "macd", "signal"], 9))
    dmi_period = max(
        _get_int(["indicators", "dmi", "period"], 14),
        _get_high(["indicators", "dmi", "period"], 14),
    )
    dmi_smooth = max(
        _get_int(["indicators", "dmi", "adx_smoothing"], 6),
        _get_high(["indicators", "dmi", "adx_smoothing"], 6),
    )
    stoch_k = max(
        _get_int(["indicators", "stoch", "k"], 12),
        _get_high(["indicators", "stoch", "k"], 12),
    )
    stoch_k_smooth = max(
        _get_int(["indicators", "stoch", "k_smooth"], 2),
        _get_high(["indicators", "stoch", "k_smooth"], 2),
    )
    stoch_d = max(
        _get_int(["indicators", "stoch", "d"], 3),
        _get_high(["indicators", "stoch", "d"], 3),
    )
    cci_p = max(
        _get_int(["indicators", "cci", "period"], 20),
        _get_high(["indicators", "cci", "period"], 20),
    )

    warmup = max(
        int(macd_slow) + int(macd_signal) + 5,
        int(dmi_period) + int(dmi_smooth) + 5,
        int(stoch_k) + int(stoch_k_smooth) + int(stoch_d) + 5,
        int(cci_p) + 5,
    )
    return int(max(50, warmup))


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

    out_cfg = raw.get("output") or {}
    if not isinstance(out_cfg, dict):
        out_cfg = {}
    out_dir = Path(str(out_cfg.get("out_dir") or "data/processed/backtests/optuna_new_strategie_flip_multi"))
    out_dir.mkdir(parents=True, exist_ok=True)

    optuna_cfg = raw.get("optuna") or {}
    if not isinstance(optuna_cfg, dict):
        optuna_cfg = {}

    study_name = str(optuna_cfg.get("study_name") or "new_strategie_flip_multi_2024")

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

    optuna_log_level = str(optuna_cfg.get("log_level") or "").strip().lower()
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

    if isinstance(search_space.get("indicators"), dict) and "macd" in (search_space.get("indicators") or {}):
        raise ValueError(
            "search_space.indicators.macd is not supported for this optimizer. "
            "MACD is fixed from base_backtest_config. Remove search_space.indicators.macd."
        )

    constraints = raw.get("constraints") or {}
    if not isinstance(constraints, dict):
        constraints = {}
    min_trades = int(constraints.get("min_trades") or 1)

    fixed_period = raw.get("fixed_period") or {}
    if not isinstance(fixed_period, dict):
        fixed_period = {}
    fixed_start = str(fixed_period.get("start") or "2024-01-01")
    fixed_end = str(fixed_period.get("end") or "2024-12-31")

    universe = raw.get("universe") or {}
    if not isinstance(universe, dict):
        universe = {}

    u_symbols = universe.get("symbols")
    symbols = [str(x) for x in (u_symbols or [])] if isinstance(u_symbols, list) else []

    u_intervals = universe.get("intervals")
    intervals = [str(x) for x in (u_intervals or [])] if isinstance(u_intervals, list) else []

    u_years = universe.get("years")
    years = [int(x) for x in (u_years or [])] if isinstance(u_years, list) else []

    if not symbols:
        symbols = [str((base_cfg_dict.get("bybit") or {}).get("symbol") or "LINKUSDT")]
    if not intervals:
        intervals = [str((base_cfg_dict.get("bybit") or {}).get("interval") or "6h")]

    walk_forward = raw.get("walk_forward") or {}
    if not isinstance(walk_forward, dict):
        walk_forward = {}
    wf_mode = str(walk_forward.get("mode") or "single_split").strip().lower()
    wf_train_pct = float(walk_forward.get("train_pct") or 0.7)
    if wf_train_pct <= 0.0 or wf_train_pct >= 1.0:
        wf_train_pct = 0.7

    warmup_bars = _warmup_bars_from_search_space(base_cfg_dict=base_cfg_dict, search_space=search_space)

    def _fetch_one(*, symbol: str, interval: str, start_s: str, end_s: str) -> pd.DataFrame:
        cfg_dict_for_fetch = copy.deepcopy(base_cfg_dict)
        _deep_set(cfg_dict_for_fetch, ["bybit", "symbol"], str(symbol))
        _deep_set(cfg_dict_for_fetch, ["bybit", "interval"], str(interval))
        _deep_set(cfg_dict_for_fetch, ["bybit", "start"], str(start_s))
        _deep_set(cfg_dict_for_fetch, ["bybit", "end"], str(end_s))
        cfg = load_config_dict(cfg_dict_for_fetch)

        start_dt = pd.Timestamp(str(cfg.bybit.start), tz="UTC")
        end_dt = pd.Timestamp(str(cfg.bybit.end), tz="UTC")
        if start_dt > end_dt:
            return pd.DataFrame([])

        tf_min = _tf_minutes(str(cfg.bybit.interval))
        warmup_ms = int(warmup_bars) * int(tf_min) * 60_000 if int(tf_min) > 0 else 0
        fetch_start_dt = start_dt - pd.Timedelta(milliseconds=int(warmup_ms))
        fetch_end_dt = end_dt + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)

        df = fetch_bybit_klines_range(
            symbol=str(cfg.bybit.symbol),
            interval=str(cfg.bybit.interval),
            start_ms=int(fetch_start_dt.timestamp() * 1000),
            end_ms=int(fetch_end_dt.timestamp() * 1000),
            base_url=str(cfg.bybit.base_url),
            category=str(cfg.bybit.category),
        )
        if df.empty:
            return df
        return df.sort_values("ts").reset_index(drop=True)

    data_cache: dict[tuple[str, str, int], pd.DataFrame] = {}
    if years:
        for sym in symbols:
            for itv in intervals:
                for y in years:
                    ys, ye = _year_bounds(int(y))
                    data_cache[(str(sym), str(itv), int(y))] = _fetch_one(symbol=str(sym), interval=str(itv), start_s=ys, end_s=ye)
    else:
        for sym in symbols:
            for itv in intervals:
                data_cache[(str(sym), str(itv), 0)] = _fetch_one(symbol=str(sym), interval=str(itv), start_s=fixed_start, end_s=fixed_end)

    def _evaluate_cfg_one(cfg, *, df_base: pd.DataFrame, interval: str) -> dict[str, float]:
        ns_cfg = NewStrategieConfig(
            macd_fast=int(cfg.indicators.macd_fast),
            macd_slow=int(cfg.indicators.macd_slow),
            macd_signal=int(cfg.indicators.macd_signal),
            dmi_period=int(cfg.indicators.dmi_period),
            dmi_adx_smoothing=int(cfg.indicators.dmi_adx_smoothing),
            stoch_k_period=int(cfg.indicators.stoch_k),
            stoch_k_smooth_period=int(cfg.indicators.stoch_k_smooth),
            stoch_d_period=int(cfg.indicators.stoch_d),
            cci_period=int(cfg.indicators.cci_period),
            pivot_zone_pct=float(cfg.pivots.zone_pct),
            pivot_merge_pct=float(cfg.pivots.merge_pct),
            max_pivots=int(cfg.pivots.max_pivots),
            signal_condition_window_bars=int(cfg.signals.condition_window_bars),
        )

        df_ind = ensure_indicators_df(df_base, cfg=ns_cfg, force=True)

        ts = pd.to_numeric(df_ind["ts"], errors="coerce").astype("Int64")
        ts = ts.dropna()
        if ts.empty:
            return {"n_trades": 0.0, "equity_end": -1e6, "equity_dd": 1e6, "winrate": 0.0}

        if wf_mode != "single_split":
            raise ValueError(f"Unsupported walk_forward.mode: {wf_mode}")

        split_ts = int(ts.quantile(float(wf_train_pct)))
        df_test = df_ind.loc[pd.to_numeric(df_ind["ts"], errors="coerce") >= int(split_ts)].reset_index(drop=True)
        if len(df_test) < 10:
            return {"n_trades": 0.0, "equity_end": -1e6, "equity_dd": 1e6, "winrate": 0.0}

        tfm = _tf_minutes(str(interval))
        window_days = int(cfg.window.window_days)
        window_bars = int(round(float(window_days) * 24.0 * 60.0 / float(tfm))) if tfm > 0 else int(len(df_ind))
        if window_bars < 1:
            window_bars = 1
        if window_bars > int(len(df_test)):
            window_bars = int(len(df_test))

        df_win = df_test.iloc[-int(window_bars) :].reset_index(drop=True)

        pivots = build_top_pivots(df_win, cfg=ns_cfg)
        signals = find_signals(df_win, pivots=pivots, cfg=ns_cfg, max_signals=5000)
        if not bool(cfg.signals.enable_premature):
            signals = [s for s in signals if str(s.kind) != "premature"]

        res = run_backtest_flip(df_win, signals=signals, cfg=ns_cfg, sl_pct=float(cfg.backtest.sl_pct))
        summary = dict(res.get("summary") or {})

        n_trades = int(summary.get("n_trades") or 0)
        equity_end = float(summary.get("equity_end") or 0.0)
        dd = abs(float(summary.get("max_dd") or 0.0))
        winrate = float(summary.get("winrate") or 0.0)

        if int(n_trades) < int(min_trades):
            equity_end = float(equity_end) - 1e6
            dd = float(dd) + 1e6
            winrate = 0.0

        return {
            "n_trades": float(n_trades),
            "equity_end": float(equity_end),
            "equity_dd": float(dd),
            "winrate": float(winrate),
        }

    def _evaluate_cfg_robust(cfg) -> dict[str, float]:
        eqs: list[float] = []
        dds: list[float] = []
        wrs: list[float] = []
        ntr_total = 0

        for (sym, itv, y), df in data_cache.items():
            if df.empty:
                eqs.append(-1e6)
                dds.append(1e6)
                wrs.append(0.0)
                continue

            m = _evaluate_cfg_one(cfg, df_base=df, interval=str(itv))
            ntr_total += int(m.get("n_trades") or 0)
            eqs.append(float(m.get("equity_end") or 0.0))
            dds.append(float(m.get("equity_dd") or 0.0))
            wrs.append(float(m.get("winrate") or 0.0))

        if not eqs:
            return {"n_trades": 0.0, "equity_end": -1e6, "equity_dd": 1e6, "winrate": 0.0}

        if int(ntr_total) < int(min_trades):
            return {"n_trades": float(ntr_total), "equity_end": -1e6, "equity_dd": 1e6, "winrate": 0.0}

        return {
            "n_trades": float(ntr_total),
            "equity_end": float(sum(eqs) / float(len(eqs))),
            "equity_dd": float(sum(dds) / float(len(dds))),
            "winrate": float(sum(wrs) / float(len(wrs))),
        }

    def objective(trial):
        cfg_dict, chosen = _apply_search_space(trial, base_cfg_dict=base_cfg_dict, search_space=search_space)

        _deep_set(cfg_dict, ["bybit", "start"], fixed_start)
        _deep_set(cfg_dict, ["bybit", "end"], fixed_end)

        cfg = load_config_dict(cfg_dict)

        if int(cfg.indicators.macd_fast) >= int(cfg.indicators.macd_slow):
            raise optuna.TrialPruned()

        m = _evaluate_cfg_robust(cfg)

        for k, v in chosen.items():
            trial.set_user_attr(f"param.{k}", v)

        trial.set_user_attr("n_trades", int(m["n_trades"]))
        trial.set_user_attr("equity_end", float(m["equity_end"]))
        trial.set_user_attr("equity_dd", float(m["equity_dd"]))
        trial.set_user_attr("winrate", float(m["winrate"]))

        return (float(m["equity_end"]), float(m["equity_dd"]), float(m["winrate"]))

    directions = ["maximize", "minimize", "maximize"]
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
            "equity_end": float(t.values[0]),
            "equity_dd": float(t.values[1]),
            "winrate": float(t.values[2]),
        }
        for k, v in (t.params or {}).items():
            r[f"param.{k}"] = v
        pareto_rows.append(r)
    pareto_df = pd.DataFrame(pareto_rows)
    pareto_csv = out_dir / "pareto_trials.csv"
    pareto_df.to_csv(pareto_csv, index=False, float_format="%.10f")

    best_by_equity = None
    if len(pareto):
        best_by_equity = max(pareto, key=lambda t: float(t.values[0]))

    best_cfg_path = None
    if best_by_equity is not None:
        best_cfg_dict = copy.deepcopy(base_cfg_dict)
        _deep_set(best_cfg_dict, ["bybit", "start"], fixed_start)
        _deep_set(best_cfg_dict, ["bybit", "end"], fixed_end)
        for k, v in best_by_equity.params.items():
            path = [p for p in str(k).split(".") if p]
            _deep_set(best_cfg_dict, path, v)
        best_cfg_path = out_dir / "best_by_equity_backtest_config.yaml"
        best_cfg_path.write_text(yaml.safe_dump(best_cfg_dict, sort_keys=False))

    summary = {
        "study_name": str(study.study_name),
        "storage": str(storage),
        "load_if_exists": bool(load_if_exists),
        "n_trials": int(len(study.trials)),
        "trials_csv": str(trials_csv),
        "pareto_csv": str(pareto_csv),
        "best_by_equity_config": (None if best_cfg_path is None else str(best_cfg_path)),
        "fixed_start": str(fixed_start),
        "fixed_end": str(fixed_end),
        "warmup_bars": int(warmup_bars),
        "min_trades": int(min_trades),
    }

    summary_path = out_dir / "optuna_summary.yaml"
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False))

    print("Optuna finished")
    print(f"- study_name: {study.study_name}")
    print(f"- storage: {storage}")
    print(f"- n_trials: {len(study.trials)}")
    print(f"- trials_csv: {trials_csv}")
    print(f"- pareto_csv: {pareto_csv}")
    print(f"- summary: {summary_path}")
    if best_cfg_path is not None:
        print(f"- best_by_equity_config: {best_cfg_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
