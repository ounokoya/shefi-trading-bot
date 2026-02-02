from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from libs.zerem.trades import safe_float


def max_drawdown_abs(equity_curve: List[float]) -> float:
    peak = 0.0
    max_dd = 0.0
    for v in equity_curve:
        peak = max(float(peak), float(v))
        dd = float(peak) - float(v)
        max_dd = max(float(max_dd), float(dd))
    return float(max_dd)


def trade_mae_pct(df: pd.DataFrame, trade: Dict) -> Optional[float]:
    entry_i = trade.get("entry_i")
    exit_i = trade.get("exit_i")
    side = trade.get("side")
    entry_price = safe_float(trade.get("entry_price"))
    if entry_price is None or float(entry_price) == 0.0:
        return None
    if entry_i is None or exit_i is None or side is None:
        return None
    try:
        a = int(entry_i)
        b = int(exit_i)
    except Exception:
        return None
    if a < 0 or b < 0 or a >= len(df) or b >= len(df):
        return None
    if b < a:
        a, b = b, a
    sl = df.iloc[a : b + 1]
    if sl.empty:
        return None
    if int(side) == 1:
        lo = pd.to_numeric(sl.get("low"), errors="coerce").astype(float)
        m = float(lo.min()) if len(lo) else math.nan
        if not math.isfinite(m):
            return None
        return 100.0 * ((float(m) - float(entry_price)) / float(entry_price))
    if int(side) == -1:
        hi = pd.to_numeric(sl.get("high"), errors="coerce").astype(float)
        m = float(hi.max()) if len(hi) else math.nan
        if not math.isfinite(m):
            return None
        return -100.0 * ((float(m) - float(entry_price)) / float(entry_price))
    return None


@dataclass(frozen=True)
class ZeremScore:
    equity: float
    max_dd: float
    eq_dd_ratio: float
    eq_mae_ratio: float
    ratio_final: float
    worst_mae: float
    n_trades: int
    winrate: float


def regularity_penalty_from_monthly_report(
    monthly_report: pd.DataFrame,
    *,
    min_active_months: int,
    max_month_share: float,
    min_trades: int = 0,
) -> tuple[float, Dict[str, float]]:
    if monthly_report is None or monthly_report.empty:
        total_trades = 0.0
        active_months = 0.0
        max_share = 0.0
    else:
        n = pd.to_numeric(monthly_report.get("n_trades"), errors="coerce").fillna(0.0).astype(float)
        total_trades = float(n.sum())
        active_months = float((n > 0.0).sum())
        max_share = float(n.max() / total_trades) if float(total_trades) > 0.0 else 0.0

    min_active_months = int(max(0, int(min_active_months)))
    min_trades = int(max(0, int(min_trades)))
    max_month_share = float(max(0.0, min(1.0, float(max_month_share))))

    if int(min_active_months) <= 0:
        p_active = 0.0
    else:
        p_active = max(0.0, (float(min_active_months) - float(active_months)) / float(min_active_months))

    if float(max_month_share) <= 0.0:
        p_share = float(max_share > 0.0)
    elif float(max_month_share) >= 1.0:
        p_share = 0.0
    else:
        p_share = max(0.0, (float(max_share) - float(max_month_share)) / (1.0 - float(max_month_share)))

    if int(min_trades) <= 0:
        p_trades = 0.0
    else:
        p_trades = max(0.0, (float(min_trades) - float(total_trades)) / float(min_trades))

    penalty = float(p_active) + float(p_share) + float(p_trades)
    stats = {
        "total_trades": float(total_trades),
        "active_months": float(active_months),
        "max_month_share": float(max_share),
        "p_active": float(p_active),
        "p_share": float(p_share),
        "p_trades": float(p_trades),
        "penalty": float(penalty),
    }
    return float(penalty), stats


def score_trades(df: pd.DataFrame, trades: List[Dict]) -> ZeremScore:
    risk_floor_pct = 1.0
    pcts = [float(t.get("pct") or 0.0) for t in trades]
    equity = float(sum(pcts))

    curve: List[float] = []
    cur = 0.0
    for p in pcts:
        cur += float(p)
        curve.append(float(cur))

    max_dd_abs = max_drawdown_abs(curve)

    dd_denom = max(float(max_dd_abs), float(risk_floor_pct))
    eq_dd_ratio = float(equity) / float(dd_denom)

    maes: List[float] = []
    for t in trades:
        mae = trade_mae_pct(df, t)
        if mae is not None and math.isfinite(float(mae)):
            maes.append(float(mae))
    worst_mae = float(min(maes)) if maes else 0.0

    mae_denom = max(abs(float(worst_mae)), float(risk_floor_pct))
    eq_mae_ratio = float(equity) / float(mae_denom)

    ratio_final = (float(eq_dd_ratio) + float(eq_mae_ratio)) / 2.0

    wins = sum(1 for p in pcts if float(p) > 0.0)
    n_trades = int(len(pcts))
    winrate = (100.0 * float(wins) / float(n_trades)) if n_trades > 0 else 0.0

    return ZeremScore(
        equity=float(equity),
        max_dd=float(max_dd_abs),
        eq_dd_ratio=float(eq_dd_ratio),
        eq_mae_ratio=float(eq_mae_ratio),
        ratio_final=float(ratio_final),
        worst_mae=float(worst_mae),
        n_trades=int(n_trades),
        winrate=float(winrate),
    )


def _month_key(dt: pd.Timestamp) -> str:
    d = pd.Timestamp(dt)
    if d.tz is None:
        d = d.tz_localize("UTC")
    else:
        d = d.tz_convert("UTC")
    return f"{int(d.year):04d}-{int(d.month):02d}"


def monthly_report_by_exit_month(
    df: pd.DataFrame,
    trades: List[Dict],
    *,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> Tuple[pd.DataFrame, ZeremScore]:
    risk_floor_pct = 1.0
    s = pd.Timestamp(start_dt)
    e = pd.Timestamp(end_dt)
    if s.tz is None:
        s = s.tz_localize("UTC")
    else:
        s = s.tz_convert("UTC")
    if e.tz is None:
        e = e.tz_localize("UTC")
    else:
        e = e.tz_convert("UTC")

    start_month = pd.Timestamp(year=int(s.year), month=int(s.month), day=1, tz="UTC")
    end_month = pd.Timestamp(year=int(e.year), month=int(e.month), day=1, tz="UTC")

    months: List[pd.Timestamp] = []
    cur = start_month
    while cur <= end_month:
        months.append(cur)
        cur = cur + pd.offsets.MonthBegin(1)

    rows: List[Dict] = []
    equity_total = 0.0
    max_dd_total = 0.0
    worst_mae_total = 0.0
    wins_total = 0
    n_trades_total = 0

    for m0 in months:
        m1 = (m0 + pd.offsets.MonthBegin(1))
        month_key = _month_key(m0)

        month_trades: List[Dict] = []
        for t in trades:
            xdt = t.get("exit_dt")
            if xdt is None:
                xdt = t.get("exit_ts")
            if xdt is None:
                continue
            x = pd.Timestamp(xdt)
            if x.tz is None:
                x = x.tz_localize("UTC")
            else:
                x = x.tz_convert("UTC")
            if x >= m0 and x < m1:
                month_trades.append(t)

        month_trades = sorted(
            month_trades,
            key=lambda t: int(t.get("exit_ts") or 0),
        )

        sc = score_trades(df, month_trades)

        wins_total += sum(1 for t in month_trades if float(t.get("pct") or 0.0) > 0.0)
        n_trades_total += int(sc.n_trades)

        equity_total += float(sc.equity)
        max_dd_total += -float(sc.max_dd)
        worst_mae_total += float(sc.worst_mae)

        rows.append(
            {
                "month": str(month_key),
                "n_trades": int(sc.n_trades),
                "equity": float(sc.equity),
                "max_dd": -float(sc.max_dd),
                "worst_mae": float(sc.worst_mae),
                "ratio_final": float(sc.ratio_final),
                "winrate": float(sc.winrate),
            }
        )

    max_dd_abs_total = abs(float(max_dd_total))
    mae_abs_total = abs(float(worst_mae_total))

    dd_denom_total = max(float(max_dd_abs_total), float(risk_floor_pct))
    mae_denom_total = max(float(mae_abs_total), float(risk_floor_pct))
    eq_dd_ratio = float(equity_total) / float(dd_denom_total)
    eq_mae_ratio = float(equity_total) / float(mae_denom_total)

    ratio_final = (float(eq_dd_ratio) + float(eq_mae_ratio)) / 2.0

    total_score = ZeremScore(
        equity=float(equity_total),
        max_dd=float(max_dd_total),
        eq_dd_ratio=float(eq_dd_ratio),
        eq_mae_ratio=float(eq_mae_ratio),
        ratio_final=float(ratio_final),
        worst_mae=float(worst_mae_total),
        n_trades=int(n_trades_total),
        winrate=(100.0 * float(wins_total) / float(n_trades_total)) if int(n_trades_total) > 0 else 0.0,
    )

    return pd.DataFrame(rows), total_score


def _mad(values: List[float]) -> float:
    if not values:
        return 0.0
    s = pd.to_numeric(pd.Series(list(values), dtype=float), errors="coerce").dropna().astype(float)
    if s.empty:
        return 0.0
    med = float(s.median())
    mad = float((s - float(med)).abs().median())
    if not math.isfinite(float(mad)):
        return 0.0
    return float(mad)


def monthly_ratio_objective_from_report(
    monthly_report: pd.DataFrame,
    *,
    mad_weight: float = 0.0,
) -> tuple[float, float, float]:
    if monthly_report is None or monthly_report.empty:
        ratio_sum = 0.0
        ratios: List[float] = []
    else:
        r = pd.to_numeric(monthly_report.get("ratio_final"), errors="coerce").fillna(0.0).astype(float)
        ratios = [float(x) for x in list(r.values)]
        ratio_sum = float(r.sum())

    mad = _mad(list(ratios))
    mad_weight = float(mad_weight or 0.0)
    score_adj = float(ratio_sum) - float(mad_weight) * float(mad)
    return float(ratio_sum), float(mad), float(score_adj)
