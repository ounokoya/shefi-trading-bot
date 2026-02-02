#!/usr/bin/env python3

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import optuna
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj if isinstance(obj, dict) else {}


def _resolve_db_path(cfg: Dict[str, Any]) -> Path:
    raw = str(cfg.get("db") or "").strip()
    if not raw:
        raise ValueError("Missing 'db' in config")
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def _pick_study_name(storage_url: str, requested: str) -> str:
    requested = str(requested or "").strip()
    summaries = optuna.get_all_study_summaries(storage=storage_url)
    if not summaries:
        raise ValueError("No studies found in DB")

    if requested:
        if any(s.study_name == requested for s in summaries):
            return requested
        raise ValueError(f"study_name '{requested}' not found in DB. Available: {[s.study_name for s in summaries]}")

    if len(summaries) == 1:
        return summaries[0].study_name

    # Heuristic: prefer study with most trials.
    best = max(summaries, key=lambda s: int(getattr(s, "n_trials", 0) or 0))
    return best.study_name


def _flatten_trial(tr: optuna.trial.FrozenTrial, *, values: Tuple[Optional[float], ...]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "trial": int(tr.number),
        "state": str(tr.state.name),
    }

    if len(values) == 1:
        row["value"] = float(values[0]) if values[0] is not None else None
    else:
        for i, v in enumerate(values):
            row[f"value_{i}"] = float(v) if v is not None else None

    for k, v in (tr.user_attrs or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            row[f"ua:{k}"] = v
        else:
            row[f"ua:{k}"] = str(v)

    for k, v in (tr.params or {}).items():
        row[f"p:{k}"] = v

    return row


def export_ratio_top10(*, db_path: Path, study_name: str, out_dir: Path, top_n: int) -> Path:
    storage_url = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trials = sorted(trials, key=lambda t: float(t.value), reverse=True)
    trials = trials[: int(top_n)]

    rows: List[Dict[str, Any]] = []
    for t in trials:
        rows.append(_flatten_trial(t, values=(t.value,)))

    df = pd.DataFrame(rows)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"zerem_ratio_top{int(top_n)}_{db_path.stem}_{ts}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def export_multi_top10(*, db_path: Path, study_name: str, out_dir: Path, top_n: int) -> Path:
    storage_url = f"sqlite:///{db_path}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None]

    def _key(t: optuna.trial.FrozenTrial) -> Tuple[float, float, float]:
        v0 = float(t.values[0]) if t.values and t.values[0] is not None else float("-inf")
        v1 = float(t.values[1]) if t.values and len(t.values) > 1 and t.values[1] is not None else float("-inf")
        v2 = float(t.values[2]) if t.values and len(t.values) > 2 and t.values[2] is not None else float("-inf")
        return (v0, v1, v2)

    trials = sorted(trials, key=_key, reverse=True)
    trials = trials[: int(top_n)]

    rows: List[Dict[str, Any]] = []
    for t in trials:
        vals = tuple(t.values) if t.values is not None else (None, None, None)
        rows.append(_flatten_trial(t, values=vals))

    df = pd.DataFrame(rows)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"zerem_multi_top{int(top_n)}_{db_path.stem}_{ts}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio-config", default="configs/optuna_zerem_ratio_2025.yaml")
    ap.add_argument("--multi-config", default="configs/optuna_zerem_multi_2025.yaml")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--out-dir", default="data/processed/optuna/exports")
    ap.add_argument("--ratio-study-name", default="")
    ap.add_argument("--multi-study-name", default="")
    args = ap.parse_args()

    ratio_cfg_path = Path(str(args.ratio_config))
    if not ratio_cfg_path.is_absolute():
        ratio_cfg_path = PROJECT_ROOT / ratio_cfg_path
    multi_cfg_path = Path(str(args.multi_config))
    if not multi_cfg_path.is_absolute():
        multi_cfg_path = PROJECT_ROOT / multi_cfg_path

    ratio_cfg = _load_yaml(ratio_cfg_path)
    multi_cfg = _load_yaml(multi_cfg_path)

    ratio_db = _resolve_db_path(ratio_cfg)
    multi_db = _resolve_db_path(multi_cfg)

    out_dir = Path(str(args.out_dir))
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    ratio_storage = f"sqlite:///{ratio_db}"
    multi_storage = f"sqlite:///{multi_db}"

    ratio_study_name = _pick_study_name(ratio_storage, str(args.ratio_study_name or ratio_cfg.get("study_name") or ""))
    multi_study_name = _pick_study_name(multi_storage, str(args.multi_study_name or multi_cfg.get("study_name") or ""))

    ratio_out = export_ratio_top10(
        db_path=ratio_db,
        study_name=ratio_study_name,
        out_dir=out_dir,
        top_n=int(args.top_n),
    )
    multi_out = export_multi_top10(
        db_path=multi_db,
        study_name=multi_study_name,
        out_dir=out_dir,
        top_n=int(args.top_n),
    )

    print(f"ratio_db={ratio_db} study={ratio_study_name} -> {ratio_out}")
    print(f"multi_db={multi_db} study={multi_study_name} -> {multi_out}")


if __name__ == "__main__":
    main()
