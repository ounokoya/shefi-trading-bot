from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import optuna
import pandas as pd

from libs.zerem.dataset import load_zerem_df
from libs.zerem.metrics import (
    ZeremScore,
    monthly_ratio_objective_from_report,
    monthly_report_by_exit_month,
    regularity_penalty_from_monthly_report,
)
from libs.zerem.multitf import ensure_cci_tf_column
from libs.zerem.timeframes import indicator_params_for_tf, tf_to_minutes
from libs.zerem.trades import simulate_trades_from_stream


@dataclass(frozen=True)
class ZeremOptunaConfig:
    project_root: Path
    symbol: str
    start_date: str
    end_date: str

    timeframes: List[str]

    series_universe: List[str]
    max_combo_size: int

    cci_low: float
    cci_high: float
    trade_direction: str

    use_fixed_stop: bool
    stop_buffers_pct: List[float]

    extreme_confirm_bars: int
    entry_require_hist_abs_growth: bool

    entry_cci_tf_confluence: bool
    entry_cci_tf_max_combo_size: int

    exit_b_mode: str

    allow_warmup_before_start: bool
    offline: bool
    cache_validate: bool
    cache_max_missing: int

    objective_mode: str = "annual"
    monthly_objective: Optional[Dict[str, Any]] = None

    search_space: Optional[Dict[str, Any]] = None
    regularity_penalty: Optional[Dict[str, Any]] = None


def _suggest_from_space(trial: optuna.Trial, *, name: str, space: Optional[Dict[str, Any]], default: Any) -> Any:
    if not isinstance(space, dict):
        return default
    spec = space.get(str(name))
    if not isinstance(spec, dict):
        return default

    st = str(spec.get("suggest") or spec.get("type") or "").strip().lower()
    if st in {"int", "integer"}:
        low = int(spec.get("low"))
        high = int(spec.get("high"))
        step = int(spec.get("step") or 1)
        return int(trial.suggest_int(str(name), low=int(low), high=int(high), step=int(step)))
    if st in {"float"}:
        low = float(spec.get("low"))
        high = float(spec.get("high"))
        step = spec.get("step")
        if step is None:
            return float(trial.suggest_float(str(name), low=float(low), high=float(high)))
        return float(trial.suggest_float(str(name), low=float(low), high=float(high), step=float(step)))
    if st in {"categorical", "cat"}:
        choices = spec.get("choices")
        if not isinstance(choices, list):
            return default
        return trial.suggest_categorical(str(name), list(choices))
    if st in {"bool", "boolean"}:
        return bool(trial.suggest_categorical(str(name), [True, False]))
    return default


def _series_to_col() -> Dict[str, str]:
    return {
        "price": "close",
        "asi": "asi",
        "pvt": "pvt",
        "macd_line": "macd_line",
        "macd_signal": "macd_signal",
        "macd_hist": "macd_hist",
        "dmi_dx": "dx",
        "klinger_kvo": "kvo",
        "klinger_signal": "klinger_signal",
        "mfi": "mfi",
        "stoch_k": "stoch_k",
        "stoch_d": "stoch_d",
    }


def _tf_superset(*, base_tf: str, tfs: List[str]) -> List[str]:
    base_minutes = tf_to_minutes(str(base_tf))
    out = [t for t in list(tfs) if tf_to_minutes(str(t)) >= int(base_minutes) and str(t) != str(base_tf)]
    out = sorted(out, key=lambda x: tf_to_minutes(str(x)))
    return out


def _cci_tf_combos(*, base_tf: str, tfs: List[str], max_total: int) -> List[Tuple[str, ...]]:
    if int(max_total) < 1:
        max_total = 1
    max_add = max(0, int(max_total) - 1)

    tf_sup = _tf_superset(base_tf=str(base_tf), tfs=list(tfs))
    max_add = min(int(max_add), int(len(tf_sup)))

    combos: List[Tuple[str, ...]] = [tuple()]
    for k in range(0, int(max_add) + 1):
        for c in itertools.combinations(tf_sup, k):
            combos.append(tuple(c))

    uniq: List[Tuple[str, ...]] = []
    seen = set()
    for c in combos:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def _series_combos(series_universe: List[str], k: int) -> List[Tuple[str, ...]]:
    u = list(series_universe)
    if not u:
        return []
    k = int(k)
    if int(k) < 1:
        k = 1
    k = min(int(k), int(len(u)))
    combos: List[Tuple[str, ...]] = []
    for c in itertools.combinations(u, int(k)):
        combos.append(tuple(c))
    return combos


def _cached_timeframes_for_symbol(project_root: Path, symbol: str) -> List[str]:
    cache_dir = Path(project_root) / "data" / "raw" / "klines_cache"
    if not cache_dir.exists():
        return []
    pat = re.compile(rf"^{re.escape(str(symbol))}_(.+?)_\d{{4}}-\d{{2}}-\d{{2}}_\d{{4}}-\d{{2}}-\d{{2}}\.csv$")
    out: List[str] = []
    seen = set()
    for p in cache_dir.glob(f"{symbol}_*.csv"):
        m = pat.match(p.name)
        if not m:
            continue
        tf = str(m.group(1))
        if tf not in seen:
            out.append(tf)
            seen.add(tf)
    return out


def _objective(trial: optuna.Trial, *, cfg: ZeremOptunaConfig) -> float:
    tf = str(trial.suggest_categorical("timeframe", list(cfg.timeframes)))

    stop_buf = float(trial.suggest_categorical("stop_buffer_pct", list(cfg.stop_buffers_pct)))

    cci_extreme_abs_raw = _suggest_from_space(
        trial,
        name="cci_extreme_abs",
        space=cfg.search_space,
        default=min(abs(float(cfg.cci_low)), abs(float(cfg.cci_high))),
    )
    cci_extreme_abs = float(cci_extreme_abs_raw)
    cci_extreme_abs = max(0.0, min(150.0, float(cci_extreme_abs)))
    cci_low = -float(cci_extreme_abs)
    cci_high = float(cci_extreme_abs)

    extreme_confirm_bars = int(
        _suggest_from_space(trial, name="extreme_confirm_bars", space=cfg.search_space, default=int(cfg.extreme_confirm_bars))
    )

    use_fixed_stop = bool(
        _suggest_from_space(trial, name="use_fixed_stop", space=cfg.search_space, default=bool(cfg.use_fixed_stop))
    )

    entry_no_hist_abs_growth = bool(
        _suggest_from_space(
            trial,
            name="entry_no_hist_abs_growth",
            space=cfg.search_space,
            default=not bool(cfg.entry_require_hist_abs_growth),
        )
    )
    entry_require_hist_abs_growth = not bool(entry_no_hist_abs_growth)

    entry_cci_tf_confluence = bool(
        _suggest_from_space(
            trial, name="entry_cci_tf_confluence", space=cfg.search_space, default=bool(cfg.entry_cci_tf_confluence)
        )
    )

    exit_b_mode = str(_suggest_from_space(trial, name="exit_b_mode", space=cfg.search_space, default=str(cfg.exit_b_mode)))

    series_to_col = _series_to_col()
    universe = [s for s in cfg.series_universe if s in series_to_col]
    max_k_allowed = int(min(int(cfg.max_combo_size), int(len(universe))))
    if max_k_allowed < 1:
        raise optuna.TrialPruned()

    combo_k = int(_suggest_from_space(trial, name="max_combo_size", space=cfg.search_space, default=int(max_k_allowed)))
    combo_k = max(1, min(int(combo_k), int(max_k_allowed)))

    chosen_id = ""
    for k in range(1, int(max_k_allowed) + 1):
        combos_k = _series_combos(universe, int(k))
        combo_ids_k = ["|".join(c) for c in combos_k] or [""]
        cid_k = str(trial.suggest_categorical(f"series_combo_k{k}", combo_ids_k))
        if int(k) == int(combo_k):
            chosen_id = str(cid_k)
    selected_series = [s for s in str(chosen_id).split("|") if s]
    if not selected_series:
        raise optuna.TrialPruned()

    trial.set_user_attr("selected_series_combo", str(chosen_id))
    trial.set_user_attr("selected_series", "|".join(list(selected_series)))

    tf_combos = _cci_tf_combos(base_tf=str(tf), tfs=list(cfg.timeframes), max_total=int(cfg.entry_cci_tf_max_combo_size))
    tf_combo_ids = ["|".join(c) for c in tf_combos]
    if bool(entry_cci_tf_confluence):
        cci_tf_combo_id = str(trial.suggest_categorical(f"cci_tf_combo__{tf}__on", tf_combo_ids))
    else:
        cci_tf_combo_id = str(trial.suggest_categorical(f"cci_tf_combo__{tf}__off", [""]))
    cci_tf_combo = tuple([t for t in str(cci_tf_combo_id).split("|") if t])

    tf_params = indicator_params_for_tf(str(tf))
    warmup = max(200, int(tf_params["cci_period"]) * 3)

    df = load_zerem_df(
        project_root=Path(cfg.project_root),
        symbol=str(cfg.symbol),
        timeframe=str(tf),
        start_date=str(cfg.start_date),
        end_date=str(cfg.end_date),
        allow_warmup_before_start=bool(cfg.allow_warmup_before_start),
        warmup_bars=int(warmup),
        offline=bool(cfg.offline),
        cache_validate=bool(cfg.cache_validate),
        cache_max_missing=int(cfg.cache_max_missing),
    )
    if df is None or df.empty:
        raise optuna.TrialPruned()

    entry_cci_tf_cols: List[str] = []
    if bool(entry_cci_tf_confluence):
        for t2 in list(cci_tf_combo):
            entry_cci_tf_cols.append(ensure_cci_tf_column(df, base_tf=str(tf), target_tf=str(t2)))

    trades = simulate_trades_from_stream(
        df,
        series_to_col=series_to_col,
        mode="confluence",
        signal_from="price",
        selected_series=list(selected_series),
        cci_col="cci",
        cci_low=float(cci_low),
        cci_high=float(cci_high),
        macd_hist_col="macd_hist",
        trade_direction=str(cfg.trade_direction),
        min_confluence=int(len(selected_series)),
        use_fixed_stop=bool(use_fixed_stop),
        stop_buffer_pct=float(stop_buf),
        stop_ref_series="price",
        start_i=0,
        extreme_confirm_bars=int(extreme_confirm_bars),
        entry_require_hist_abs_growth=bool(entry_require_hist_abs_growth),
        entry_cci_tf_cols=list(entry_cci_tf_cols),
        exit_b_mode=str(exit_b_mode),
    )

    month_df, total_score = monthly_report_by_exit_month(
        df,
        trades,
        start_dt=pd.Timestamp(cfg.start_date, tz="UTC"),
        end_dt=pd.Timestamp(cfg.end_date, tz="UTC"),
    )

    objective_mode = str(getattr(cfg, "objective_mode", "annual") or "annual").strip().lower()
    if objective_mode == "monthly":
        mo = getattr(cfg, "monthly_objective", None)
        mad_weight = 0.0
        if isinstance(mo, dict):
            mad_weight = float(mo.get("mad_weight", 0.0) or 0.0)
        ratio_sum, mad, base_obj = monthly_ratio_objective_from_report(month_df, mad_weight=float(mad_weight))
        trial.set_user_attr("monthly_ratio_sum", float(ratio_sum))
        trial.set_user_attr("monthly_ratio_mad", float(mad))
        trial.set_user_attr("monthly_mad_weight", float(mad_weight))
        trial.set_user_attr("objective_mode", "monthly")
        objective_raw = float(ratio_sum)
        objective_base = float(base_obj)
    else:
        trial.set_user_attr("objective_mode", "annual")
        objective_raw = float(total_score.ratio_final)
        objective_base = float(total_score.ratio_final)

    penalty = 0.0
    penalty_stats: Dict[str, float] = {}
    weight_ratio = 0.0
    if isinstance(getattr(cfg, "regularity_penalty", None), dict) and bool(cfg.regularity_penalty.get("enabled", False)):
        min_active_months = int(cfg.regularity_penalty.get("min_active_months", 0) or 0)
        max_month_share = float(cfg.regularity_penalty.get("max_month_share", 1.0) or 1.0)
        min_trades = int(cfg.regularity_penalty.get("min_trades", 0) or 0)
        weight_ratio = float(cfg.regularity_penalty.get("weight_ratio", 0.0) or 0.0)
        penalty, penalty_stats = regularity_penalty_from_monthly_report(
            month_df,
            min_active_months=min_active_months,
            max_month_share=max_month_share,
            min_trades=min_trades,
        )

    trial.set_user_attr("equity", float(total_score.equity))
    trial.set_user_attr("max_dd", float(total_score.max_dd))
    trial.set_user_attr("worst_mae", float(total_score.worst_mae))
    trial.set_user_attr("n_trades", int(total_score.n_trades))
    trial.set_user_attr("winrate", float(total_score.winrate))

    trial.set_user_attr("ratio_final_raw", float(total_score.ratio_final))
    trial.set_user_attr("regularity_penalty", float(penalty))
    trial.set_user_attr("regularity_weight_ratio", float(weight_ratio))
    if penalty_stats:
        for k, v in penalty_stats.items():
            trial.set_user_attr(f"regularity_{k}", float(v))

    objective_adj = float(objective_base) - float(weight_ratio) * float(penalty)
    trial.set_user_attr("objective_raw", float(objective_raw))
    trial.set_user_attr("objective_base", float(objective_base))
    trial.set_user_attr("objective_adj", float(objective_adj))

    if objective_mode == "monthly":
        trial.set_user_attr("monthly_score_final", float(objective_adj))
    else:
        trial.set_user_attr("ratio_final_adj", float(objective_adj))

    return float(objective_adj)


def run_optuna_ratio(
    *,
    cfg: ZeremOptunaConfig,
    n_trials: int,
    db_path: Path,
    study_name: str,
    n_jobs: int = 1,
    seed: Optional[int] = None,
) -> optuna.Study:
    if bool(cfg.offline):
        cached_tfs = set(_cached_timeframes_for_symbol(Path(cfg.project_root), str(cfg.symbol)))
        kept = [t for t in list(cfg.timeframes) if str(t) in cached_tfs]
        if not kept:
            raise ValueError(
                f"offline=True but no cached timeframes found for symbol={cfg.symbol}. "
                "Either disable offline or build cache files in data/raw/klines_cache."
            )
        cfg = ZeremOptunaConfig(
            project_root=Path(cfg.project_root),
            symbol=str(cfg.symbol),
            start_date=str(cfg.start_date),
            end_date=str(cfg.end_date),
            timeframes=list(kept),
            series_universe=list(cfg.series_universe),
            max_combo_size=int(cfg.max_combo_size),
            cci_low=float(cfg.cci_low),
            cci_high=float(cfg.cci_high),
            trade_direction=str(cfg.trade_direction),
            use_fixed_stop=bool(cfg.use_fixed_stop),
            stop_buffers_pct=list(cfg.stop_buffers_pct),
            extreme_confirm_bars=int(cfg.extreme_confirm_bars),
            entry_require_hist_abs_growth=bool(cfg.entry_require_hist_abs_growth),
            entry_cci_tf_confluence=bool(cfg.entry_cci_tf_confluence),
            entry_cci_tf_max_combo_size=int(cfg.entry_cci_tf_max_combo_size),
            exit_b_mode=str(cfg.exit_b_mode),
            allow_warmup_before_start=bool(cfg.allow_warmup_before_start),
            offline=bool(cfg.offline),
            cache_validate=bool(cfg.cache_validate),
            cache_max_missing=int(cfg.cache_max_missing),
            objective_mode=str(getattr(cfg, "objective_mode", "annual") or "annual"),
            monthly_objective=(cfg.monthly_objective if isinstance(cfg.monthly_objective, dict) else None),
            search_space=(cfg.search_space if isinstance(cfg.search_space, dict) else None),
            regularity_penalty=(cfg.regularity_penalty if isinstance(cfg.regularity_penalty, dict) else None),
        )

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"sqlite:///{db_path}"
    storage = optuna.storages.RDBStorage(
        url=url,
        engine_kwargs={
            "connect_args": {
                "timeout": 120,
            }
        },
    )

    if int(n_jobs) != 1:
        n_jobs = 1

    sampler: optuna.samplers.BaseSampler
    if seed is None:
        sampler = optuna.samplers.TPESampler()
    else:
        sampler = optuna.samplers.TPESampler(seed=int(seed))

    study = optuna.create_study(
        study_name=str(study_name),
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    try:
        study.optimize(lambda t: _objective(t, cfg=cfg), n_trials=int(n_trials), n_jobs=int(n_jobs))
    except ValueError as e:
        msg = str(e)
        if "dynamic value space" in msg or (" not in (" in msg and "choices" not in msg):
            raise ValueError(
                "Optuna refused this run because the categorical search space changed for an existing study. "
                "This usually happens when reusing the same --db/--study-name with different --timeframes/--series/--max-combo-size/etc. "
                "Fix: choose a new --study-name (recommended) or use a fresh DB file. "
                "(This can also happen after code changes that modify categorical choices, e.g. series_combo_kN definitions.)"
            ) from e
        raise
    return study


def best_trial_summary(study: optuna.Study) -> Tuple[optuna.trial.FrozenTrial, ZeremScore]:
    bt = study.best_trial
    score = ZeremScore(
        equity=float(bt.user_attrs.get("equity", 0.0)),
        max_dd=float(bt.user_attrs.get("max_dd", 0.0)),
        eq_dd_ratio=float("nan"),
        eq_mae_ratio=float("nan"),
        ratio_final=float(bt.value),
        worst_mae=float(bt.user_attrs.get("worst_mae", 0.0)),
        n_trades=int(bt.user_attrs.get("n_trades", 0)),
        winrate=float(bt.user_attrs.get("winrate", 0.0)),
    )
    return bt, score
