from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.optimizations.zerem_ratio.runner import ZeremOptunaConfig, best_trial_summary, run_optuna_ratio


def _parse_list(raw: str) -> list[str]:
    s = str(raw or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _list_to_csv(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, (list, tuple)):
        return ",".join([str(v) for v in x])
    return str(x)


def _load_yaml_cfg(path: str) -> dict:
    p = Path(str(path))
    raw = yaml.safe_load(p.read_text())
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping")
    return raw


def _parse_pct_list(raw: str) -> list[float]:
    s = str(raw or "").strip()
    if not s:
        return []
    out: list[float] = []
    for part in s.split(","):
        p = str(part).strip().replace("%", "")
        if not p:
            continue
        v = float(p)
        if v >= 1.0:
            v = v / 100.0
        out.append(float(v))
    return out


def _slug(s: str) -> str:
    s = str(s or "")
    return "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in s)


def main() -> None:
    ap0 = argparse.ArgumentParser(add_help=False)
    ap0.add_argument("--config", default="")
    cfg_args, remaining = ap0.parse_known_args()
    cfg_dict = _load_yaml_cfg(cfg_args.config) if str(cfg_args.config or "").strip() else {}

    ap = argparse.ArgumentParser(parents=[ap0])
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--grid-year", type=int, default=cfg_dict.get("grid_year"))
    ap.add_argument("--timeframes", default=_list_to_csv(cfg_dict.get("timeframes", "5m,15m,1h")))
    ap.add_argument(
        "--series",
        default=_list_to_csv(
            cfg_dict.get(
                "series",
                "price,asi,pvt,macd_line,macd_signal,macd_hist,dmi_dx,klinger_kvo,klinger_signal,mfi,stoch_k,stoch_d",
            )
        ),
    )
    ap.add_argument("--max-combo-size", type=int, default=int(cfg_dict.get("max_combo_size", 3)))
    ap.add_argument("--cci-low", type=float, default=float(cfg_dict.get("cci_low", -100.0)))
    ap.add_argument("--cci-high", type=float, default=float(cfg_dict.get("cci_high", 100.0)))
    ap.add_argument("--trade-direction", default=str(cfg_dict.get("trade_direction", "both")), choices=["long", "short", "both"])
    ap.add_argument("--use-fixed-stop", action="store_true", default=bool(cfg_dict.get("use_fixed_stop", False)))
    ap.add_argument("--stop-buffers", default=_list_to_csv(cfg_dict.get("stop_buffers", "1,2,5,10")))
    ap.add_argument("--extreme-confirm-bars", type=int, default=int(cfg_dict.get("extreme_confirm_bars", 0)))
    ap.add_argument("--entry-no-hist-abs-growth", action="store_true", default=bool(cfg_dict.get("entry_no_hist_abs_growth", False)))
    ap.add_argument("--entry-cci-tf-confluence", action="store_true", default=bool(cfg_dict.get("entry_cci_tf_confluence", False)))
    ap.add_argument("--entry-cci-tf-max-combo-size", type=int, default=int(cfg_dict.get("entry_cci_tf_max_combo_size", 2)))
    ap.add_argument("--exit-b-mode", choices=["macd", "stoch", "klinger", "none"], default=str(cfg_dict.get("exit_b_mode", "macd")))
    ap.add_argument("--allow-warmup-before-start", action="store_true", default=bool(cfg_dict.get("allow_warmup_before_start", False)))
    ap.add_argument("--offline", action="store_true", default=bool(cfg_dict.get("offline", False)))
    ap.add_argument("--cache-no-validate", action="store_true", default=not bool(cfg_dict.get("cache_validate", True)))
    ap.add_argument("--cache-max-missing", type=int, default=int(cfg_dict.get("cache_max_missing", 5)))

    ap.add_argument(
        "--objective-mode",
        choices=["annual", "monthly"],
        default=str(cfg_dict.get("objective_mode", "annual")),
    )
    ap.add_argument(
        "--monthly-mad-weight",
        type=float,
        default=float((cfg_dict.get("monthly_objective") or {}).get("mad_weight", 0.0) or 0.0),
    )

    ap.add_argument("--n-trials", type=int, default=int(cfg_dict.get("n_trials", 200)))
    ap.add_argument("--n-jobs", type=int, default=int(cfg_dict.get("n_jobs", 1)))
    ap.add_argument("--seed", type=int, default=cfg_dict.get("seed"))
    ap.add_argument("--study-name", default=str(cfg_dict.get("study_name", "")))
    ap.add_argument("--db", default=str(cfg_dict.get("db", str(PROJECT_ROOT / "data" / "processed" / "optuna" / "zerem_ratio.db"))))

    args = ap.parse_args(remaining)

    cci_ext = min(abs(float(args.cci_low)), abs(float(args.cci_high)))
    cci_ext = max(0.0, min(150.0, float(cci_ext)))
    args.cci_low = -float(cci_ext)
    args.cci_high = float(cci_ext)

    if args.grid_year is None:
        raise SystemExit("Missing --grid-year (or set grid_year in --config)")

    if not str(args.study_name or "").strip():
        tf_slug = _slug(str(args.timeframes))
        series_slug = _slug(str(args.series))
        args.study_name = _slug(
            f"zerem_ratio_{args.symbol}_{args.grid_year}_{tf_slug}_k{args.max_combo_size}_tfk{args.entry_cci_tf_max_combo_size}_{'ccitf' if args.entry_cci_tf_confluence else 'notf'}_{args.exit_b_mode}_{series_slug}_combok_exact"
        )

    y = int(args.grid_year)
    start_date = f"{y:04d}-01-01"
    end_date = f"{y:04d}-12-31"

    cfg = ZeremOptunaConfig(
        project_root=PROJECT_ROOT,
        symbol=str(args.symbol),
        start_date=str(start_date),
        end_date=str(end_date),
        timeframes=_parse_list(args.timeframes),
        series_universe=_parse_list(args.series),
        max_combo_size=int(args.max_combo_size),
        cci_low=float(args.cci_low),
        cci_high=float(args.cci_high),
        trade_direction=str(args.trade_direction),
        use_fixed_stop=bool(args.use_fixed_stop),
        stop_buffers_pct=_parse_pct_list(args.stop_buffers),
        extreme_confirm_bars=int(args.extreme_confirm_bars),
        entry_require_hist_abs_growth=not bool(args.entry_no_hist_abs_growth),
        entry_cci_tf_confluence=bool(args.entry_cci_tf_confluence),
        entry_cci_tf_max_combo_size=int(args.entry_cci_tf_max_combo_size),
        exit_b_mode=str(args.exit_b_mode),
        allow_warmup_before_start=bool(args.allow_warmup_before_start),
        offline=bool(args.offline),
        cache_validate=not bool(args.cache_no_validate),
        cache_max_missing=int(args.cache_max_missing),
        objective_mode=str(args.objective_mode),
        monthly_objective={"mad_weight": float(args.monthly_mad_weight)},
        search_space=(cfg_dict.get("search_space") if isinstance(cfg_dict.get("search_space"), dict) else None),
        regularity_penalty=(
            cfg_dict.get("regularity_penalty") if isinstance(cfg_dict.get("regularity_penalty"), dict) else None
        ),
    )

    study = run_optuna_ratio(
        cfg=cfg,
        n_trials=int(args.n_trials),
        db_path=Path(args.db),
        study_name=str(args.study_name),
        n_jobs=int(args.n_jobs),
        seed=args.seed,
    )

    try:
        bt, sc = best_trial_summary(study)
    except Exception as e:
        print("\n=== BEST (ratio) ===")
        print(f"No completed trial found (all trials may have been pruned). {e}")
        return
    print("\n=== BEST (ratio) ===")
    objective_mode = str(bt.user_attrs.get("objective_mode", "annual"))
    pen = bt.user_attrs.get("regularity_penalty")
    extra = ""
    if objective_mode == "monthly":
        ratio_sum = bt.user_attrs.get("monthly_ratio_sum")
        mad = bt.user_attrs.get("monthly_ratio_mad")
        extra = f" monthly_sum={float(ratio_sum or 0.0):.6f} mad={float(mad or 0.0):.6f} reg_pen={float(pen or 0.0):.3f}"
    else:
        ratio_raw = bt.user_attrs.get("ratio_final_raw")
        extra = f" ratio_raw={float(ratio_raw or 0.0):.6f} reg_pen={float(pen or 0.0):.3f}"
    print(
        f"trial={bt.number} value={float(bt.value):.6f} equity={sc.equity:+.3f}% dd={sc.max_dd:+.3f}% mae={sc.worst_mae:+.3f}% n={sc.n_trades} winrate={sc.winrate:.1f}%" + extra
    )
    print(bt.params)


if __name__ == "__main__":
    main()
