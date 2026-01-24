from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.backtest.config import load_config_yaml
from libs.backtest.engine import run_backtest_from_config
from libs.backtest.indicators import ensure_indicators_df


_MS_PER_DAY = 24 * 60 * 60 * 1000


def _safe_int(x: object) -> int | None:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _ms_to_iso_utc(ms: object) -> str:
    try:
        msi = int(ms)
    except Exception:
        return ""
    return pd.to_datetime(msi, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")


def _summary_to_row(prefix: str, summary: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for k in (
        "n_trades",
        "n_wins",
        "n_losses",
        "winrate",
        "avg_win",
        "avg_loss",
        "equity_end",
        "max_dd",
        "ratio",
    ):
        if k in summary:
            out[f"{prefix}{k}"] = summary[k]
    return out


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
    trades_penalty = (
        float(trades_penalty_weight) * (float(trades_shortfall) / float(max(1, int(target_min_trades))))
    )

    neg_equity_penalty = float(neg_equity_penalty_weight) * max(0.0, -float(equity_end))

    score = float(equity_end) - float(dd_penalty) - float(trades_penalty) - float(neg_equity_penalty)

    return {
        "score": float(score),
        "dd_penalty": float(dd_penalty),
        "trades_penalty": float(trades_penalty),
        "neg_equity_penalty": float(neg_equity_penalty),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--start-ts", default=None)
    ap.add_argument("--end-ts", default=None)

    ap.add_argument("--train-days", type=int, default=30)
    ap.add_argument("--test-days", type=int, default=7)
    ap.add_argument("--step-days", type=int, default=None)

    ap.add_argument(
        "--fold-days",
        type=int,
        default=None,
        help="If set, use percent-based split on each fold (default 70/30).",
    )
    ap.add_argument(
        "--fold-count",
        type=int,
        default=None,
        help="If set, derive fold duration from the selected time range and split each fold by percent.",
    )
    ap.add_argument("--train-pct", type=float, default=0.70)
    ap.add_argument("--test-pct", type=float, default=0.30)

    ap.add_argument("--target-min-trades", type=int, default=50)
    ap.add_argument("--target-max-dd", type=float, default=0.15)
    ap.add_argument("--dd-penalty-weight", type=float, default=1.0)
    ap.add_argument("--trades-penalty-weight", type=float, default=0.02)
    ap.add_argument("--neg-equity-penalty-weight", type=float, default=1.0)

    args = ap.parse_args()

    cfg = load_config_yaml(str(args.config))

    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(Path(cfg.data.csv))
    df = ensure_indicators_df(df, cfg=cfg)

    ts_col = cfg.data.ts_col
    ts_vals = pd.to_numeric(df[ts_col], errors="coerce").dropna().astype(np.int64)
    if not len(ts_vals):
        raise ValueError("No valid timestamps")

    data_start_ts = int(ts_vals.iloc[0])
    data_end_ts = int(ts_vals.iloc[-1])

    start_ts = _safe_int(args.start_ts) if args.start_ts is not None else None
    end_ts = _safe_int(args.end_ts) if args.end_ts is not None else None

    start_ts_eff = data_start_ts if start_ts is None else int(start_ts)
    end_ts_eff = data_end_ts if end_ts is None else int(end_ts)

    if args.fold_days is not None and args.fold_count is not None:
        raise ValueError("Use only one of --fold-days or --fold-count")

    if args.fold_count is not None:
        fold_count = int(args.fold_count)
        if fold_count <= 0:
            raise ValueError("--fold-count must be > 0")

        train_pct = float(args.train_pct)
        test_pct = float(args.test_pct)
        if train_pct <= 0.0 or test_pct <= 0.0:
            raise ValueError("--train-pct and --test-pct must be > 0")
        tot = float(train_pct + test_pct)
        train_pct /= tot
        test_pct /= tot

        total_ms = int(end_ts_eff) - int(start_ts_eff) + 1
        if total_ms <= 0:
            raise ValueError("Invalid time range")
        fold_ms = int(max(_MS_PER_DAY, int(total_ms // int(fold_count))))

        train_ms = int(max(_MS_PER_DAY, int(round(float(fold_ms) * float(train_pct)))))
        train_ms = min(int(fold_ms - 1), int(train_ms))
        test_ms = int(fold_ms - train_ms)
        test_ms = max(_MS_PER_DAY, int(test_ms))

        step_ms = int(args.step_days) * _MS_PER_DAY if args.step_days is not None else int(test_ms)

    elif args.fold_days is not None:
        fold_days = int(args.fold_days)
        if fold_days <= 1:
            raise ValueError("--fold-days must be >= 2")

        train_pct = float(args.train_pct)
        test_pct = float(args.test_pct)
        if train_pct <= 0.0 or test_pct <= 0.0:
            raise ValueError("--train-pct and --test-pct must be > 0")
        tot = float(train_pct + test_pct)
        train_pct /= tot
        test_pct /= tot

        train_days = int(round(float(fold_days) * float(train_pct)))
        train_days = max(1, min(int(fold_days - 1), int(train_days)))
        test_days = int(fold_days - train_days)
        test_days = max(1, int(test_days))

        train_ms = int(train_days) * _MS_PER_DAY
        test_ms = int(test_days) * _MS_PER_DAY
        step_ms = int(args.step_days) * _MS_PER_DAY if args.step_days is not None else int(test_ms)
    else:
        train_ms = int(args.train_days) * _MS_PER_DAY
        test_ms = int(args.test_days) * _MS_PER_DAY
        step_ms = int(args.step_days) * _MS_PER_DAY if args.step_days is not None else int(test_ms)

    folds: list[dict[str, object]] = []
    test_trades_all: list[pd.DataFrame] = []

    fold_i = 0
    t0 = int(start_ts_eff)
    while True:
        train_start = int(t0)
        train_end = int(train_start + train_ms - 1)
        test_start = int(train_end + 1)
        test_end = int(test_start + test_ms - 1)

        if test_end > int(end_ts_eff):
            break

        train_res = run_backtest_from_config(
            cfg=cfg,
            df=df,
            start_ts=train_start,
            end_ts=train_end,
            ensure_indicators=False,
        )
        test_res = run_backtest_from_config(
            cfg=cfg,
            df=df,
            start_ts=test_start,
            end_ts=test_end,
            ensure_indicators=False,
        )

        train_sum = dict(train_res.get("summary") or {})
        test_sum = dict(test_res.get("summary") or {})

        train_score = _score_with_penalties(
            summary=train_sum,
            target_max_dd=float(args.target_max_dd),
            target_min_trades=int(args.target_min_trades),
            dd_penalty_weight=float(args.dd_penalty_weight),
            trades_penalty_weight=float(args.trades_penalty_weight),
            neg_equity_penalty_weight=float(args.neg_equity_penalty_weight),
        )
        test_score = _score_with_penalties(
            summary=test_sum,
            target_max_dd=float(args.target_max_dd),
            target_min_trades=int(args.target_min_trades),
            dd_penalty_weight=float(args.dd_penalty_weight),
            trades_penalty_weight=float(args.trades_penalty_weight),
            neg_equity_penalty_weight=float(args.neg_equity_penalty_weight),
        )

        row: dict[str, object] = {
            "fold": int(fold_i),
            "train_start_ts": int(train_start),
            "train_end_ts": int(train_end),
            "test_start_ts": int(test_start),
            "test_end_ts": int(test_end),
            "train_start_dt": _ms_to_iso_utc(train_start),
            "train_end_dt": _ms_to_iso_utc(train_end),
            "test_start_dt": _ms_to_iso_utc(test_start),
            "test_end_dt": _ms_to_iso_utc(test_end),
            "target_min_trades": int(args.target_min_trades),
            "target_max_dd": float(args.target_max_dd),
            "dd_penalty_weight": float(args.dd_penalty_weight),
            "trades_penalty_weight": float(args.trades_penalty_weight),
            "neg_equity_penalty_weight": float(args.neg_equity_penalty_weight),
        }
        row.update(_summary_to_row("train_", train_sum))
        row.update(_summary_to_row("test_", test_sum))
        row.update({f"train_{k}": v for k, v in train_score.items()})
        row.update({f"test_{k}": v for k, v in test_score.items()})
        folds.append(row)

        test_trades = test_res.get("trades")
        if isinstance(test_trades, pd.DataFrame) and len(test_trades):
            tdf = test_trades.copy()
            tdf.insert(0, "fold", int(fold_i))
            test_trades_all.append(tdf)

        fold_i += 1
        t0 = int(t0 + step_ms)

    folds_df = pd.DataFrame(folds)
    folds_path = out_dir / "walk_forward_folds.csv"
    folds_df.to_csv(folds_path, index=False, float_format="%.6f")

    if len(test_trades_all):
        test_trades_df = pd.concat(test_trades_all, ignore_index=True)
    else:
        test_trades_df = pd.DataFrame([])
    test_trades_path = out_dir / "walk_forward_test_trades.csv"
    test_trades_df.to_csv(test_trades_path, index=False, float_format="%.6f")

    equity_end = float(folds_df.get("test_equity_end", pd.Series([0.0])).sum()) if len(folds_df) else 0.0
    max_dd = float(folds_df.get("test_max_dd", pd.Series([0.0])).min()) if len(folds_df) else 0.0
    ratio = 0.0 if equity_end <= 0.0 else (float("inf") if max_dd == 0.0 else float(equity_end) / abs(max_dd))

    score_total = float(folds_df.get("test_score", pd.Series([0.0])).sum()) if len(folds_df) else 0.0
    dd_pen_total = float(folds_df.get("test_dd_penalty", pd.Series([0.0])).sum()) if len(folds_df) else 0.0
    trades_pen_total = float(folds_df.get("test_trades_penalty", pd.Series([0.0])).sum()) if len(folds_df) else 0.0
    neg_eq_pen_total = float(folds_df.get("test_neg_equity_penalty", pd.Series([0.0])).sum()) if len(folds_df) else 0.0

    if len(test_trades_df):
        net = pd.to_numeric(test_trades_df.get("net_ret"), errors="coerce").dropna().astype(float)
        n_trades = int(len(net))
        n_wins = int((net > 0.0).sum())
        n_losses = int((net <= 0.0).sum())
        winrate = float(n_wins / n_trades) if n_trades else 0.0
        avg_win = float(net[net > 0.0].mean()) if n_wins else 0.0
        avg_loss = float(net[net <= 0.0].mean()) if n_losses else 0.0
    else:
        n_trades = 0
        n_wins = 0
        n_losses = 0
        winrate = 0.0
        avg_win = 0.0
        avg_loss = 0.0

    agg = pd.DataFrame(
        [
            {
                "folds": int(len(folds_df)),
                "score": float(score_total),
                "dd_penalty": float(dd_pen_total),
                "trades_penalty": float(trades_pen_total),
                "neg_equity_penalty": float(neg_eq_pen_total),
                "n_trades": int(n_trades),
                "n_wins": int(n_wins),
                "n_losses": int(n_losses),
                "winrate": float(winrate),
                "avg_win": float(avg_win),
                "avg_loss": float(avg_loss),
                "equity_end": float(equity_end),
                "max_dd": float(max_dd),
                "ratio": ratio,
                "target_min_trades": int(args.target_min_trades),
                "target_max_dd": float(args.target_max_dd),
                "dd_penalty_weight": float(args.dd_penalty_weight),
                "trades_penalty_weight": float(args.trades_penalty_weight),
                "neg_equity_penalty_weight": float(args.neg_equity_penalty_weight),
            }
        ]
    )
    agg_path = out_dir / "walk_forward_summary.csv"
    agg.to_csv(agg_path, index=False, float_format="%.6f")

    print("Walk-forward summary (TEST folds aggregated):")
    print(f"- folds: {int(len(folds_df))}")
    print(f"- score: {float(score_total):.6f}")
    print(f"- dd_penalty: {float(dd_pen_total):.6f}")
    print(f"- trades_penalty: {float(trades_pen_total):.6f}")
    print(f"- neg_equity_penalty: {float(neg_eq_pen_total):.6f}")
    print(f"- n_trades: {int(n_trades)}")
    print(f"- n_wins: {int(n_wins)}")
    print(f"- n_losses: {int(n_losses)}")
    print(f"- winrate: {float(winrate):.4f}")
    print(f"- avg_win: {float(avg_win):.6f}")
    print(f"- avg_loss: {float(avg_loss):.6f}")
    print(f"- equity_end: {float(equity_end):.6f}")
    print(f"- max_dd: {float(max_dd):.6f}")
    print(f"- ratio: {ratio}")

    print(f"Wrote: {folds_path}")
    print(f"Wrote: {test_trades_path}")
    print(f"Wrote: {agg_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
