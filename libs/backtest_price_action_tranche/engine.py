from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from libs.backtest_price_action_tranche.config import FullConfig
from libs.backtest_price_action_tranche.indicators import ensure_indicators_df
from libs.backtest_price_action_tranche.signals import price_action_filters_ok, trigger_signal
from libs.extremes.window_extremes import extract_window_close_extremes
from libs.pivots.grid_confluence import get_or_refresh_execution_pivot_table
from libs.pivots.mtf_confluence import role_from_price, zone_representative_event
from libs.pivots.pivot_registry import PivotRegistry
from libs.presets.extreme_confluence_presets import get_extreme_confluence_preset


@dataclass
class PositionState:
    side: str
    entry_i: int
    entry_ts: int
    entry_price: float
    tp_price: float | None
    tp_pivot: dict[str, object] | None
    entry_pivot_table: list[dict[str, object]] | None
    peak: float
    trough: float
    atr_at_entry: float | None


@dataclass
class TradeRecord:
    side: str
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    exit_reason: str

    entry_signal_ts: int | None
    entry_signal_cand_ts: int | None
    entry_confirmed_series: str | None
    entry_trend_filter: str | None
    entry_trend_ok: bool | None
    entry_trend_vortex_side: str | None
    entry_trend_dmi_side: str | None

    exit_signal_ts: int | None

    gross_ret: float
    net_ret: float

    mfe: float
    mae: float


def _gross_ret(*, side: str, entry: float, exit_: float) -> float:
    if entry == 0.0 or (not np.isfinite(float(entry))) or (not np.isfinite(float(exit_))):
        return float("nan")
    if side == "LONG":
        return (float(exit_) / float(entry)) - 1.0
    if side == "SHORT":
        return (float(entry) / float(exit_)) - 1.0
    raise ValueError(f"Unexpected side: {side}")


def _pct_dist(a: float, b: float) -> float:
    aa = float(a)
    bb = float(b)
    if bb == 0.0:
        return float("inf")
    return abs(float(aa) / float(bb) - 1.0)


def _pivot_confluence_metrics(
    regs_by_tf: dict[str, PivotRegistry] | None,
    *,
    cfg: FullConfig,
    target_price: float | None,
    ref_price: float,
    prefix: str,
) -> dict[str, object]:
    if regs_by_tf is None or target_price is None or (not np.isfinite(float(target_price))):
        return {
            f"{prefix}_confluence_n_tfs": None,
            f"{prefix}_confluence_tfs": None,
            f"{prefix}_confluence_2of3": None,
            f"{prefix}_confluence_3of3": None,
        }

    tp = float(target_price)
    role_target = role_from_price(float(tp), current_price=float(ref_price))
    if str(role_target) not in {"support", "resistance"}:
        return {
            f"{prefix}_confluence_n_tfs": 0,
            f"{prefix}_confluence_tfs": "",
            f"{prefix}_confluence_2of3": False,
            f"{prefix}_confluence_3of3": False,
        }

    fallback_r = float(cfg.pivot_grid.grid_pct)
    radius_by_tf: dict[str, float] = {}
    if str(cfg.pivot_grid.mode).strip().lower() == "zones" and isinstance(cfg.pivot_grid.zones_cfg, dict):
        for key in ("macro", "context", "execution"):
            z = cfg.pivot_grid.zones_cfg.get(key) or {}
            if not isinstance(z, dict):
                continue
            tf = str(z.get("tf") or "").strip()
            rp = z.get("radius_pct")
            if tf and rp is not None:
                radius_by_tf[str(tf)] = float(rp)

    present: list[str] = []
    for tf, reg in regs_by_tf.items():
        r = float(radius_by_tf.get(str(tf), fallback_r))
        if r <= 0:
            continue

        ok = False
        for zid in reg.zones.keys():
            ev = zone_representative_event(reg, str(zid), prefer="last")
            if not ev:
                continue
            try:
                lvl = float(ev["level"])
            except Exception:
                continue
            if lvl <= 0:
                continue
            if role_from_price(float(lvl), current_price=float(ref_price)) != str(role_target):
                continue
            if _pct_dist(float(lvl), float(tp)) <= float(r):
                ok = True
                break
        if ok:
            present.append(str(tf))

    order = ["5m", "1h", "4h"]
    present_sorted = [tf for tf in order if tf in set(present)] + [tf for tf in present if tf not in set(order)]
    n = int(len(set(present)))
    return {
        f"{prefix}_confluence_n_tfs": int(n),
        f"{prefix}_confluence_tfs": "|".join(present_sorted),
        f"{prefix}_confluence_2of3": bool(n >= 2),
        f"{prefix}_confluence_3of3": bool(n >= 3),
    }


def _as_float(x: object) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(float(v)):
        return None
    return float(v)


def _pivot_row_for_price(
    pivot_table: list[dict[str, object]] | None,
    *,
    price: float | None,
) -> dict[str, object] | None:
    if pivot_table is None or price is None or (not np.isfinite(float(price))):
        return None
    p0 = float(price)
    for r in pivot_table:
        try:
            p = float(r.get("price"))
        except Exception:
            continue
        if np.isfinite(float(p)) and float(p) == float(p0):
            return dict(r)
    return None


def _pivot_nearest_levels(
    pivot_table: list[dict[str, object]] | None,
    *,
    current_price: float,
    n: int = 3,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    arr = list(pivot_table or [])
    supports: list[dict[str, object]] = []
    resistances: list[dict[str, object]] = []

    for r in arr:
        try:
            p = float(r.get("price"))
        except Exception:
            continue
        if not np.isfinite(float(p)) or float(p) <= 0:
            continue
        if float(p) < float(current_price):
            supports.append(dict(r))
        elif float(p) > float(current_price):
            resistances.append(dict(r))

    supports.sort(key=lambda rr: float(rr.get("price") or 0.0), reverse=True)
    resistances.sort(key=lambda rr: float(rr.get("price") or 0.0))
    return supports[: int(n)], resistances[: int(n)]


def _pivot_touch_favorable_adverse(
    pivot_table: list[dict[str, object]] | None,
    *,
    side: str,
    entry: float,
    peak: float,
    trough: float,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    arr = list(pivot_table or [])
    s = str(side).upper()
    entry_f = float(entry)
    peak_f = float(peak)
    trough_f = float(trough)

    fav: dict[str, object] | None = None
    adv: dict[str, object] | None = None

    if s == "LONG":
        fav_candidates: list[dict[str, object]] = []
        adv_candidates: list[dict[str, object]] = []
        for r in arr:
            try:
                p = float(r.get("price"))
            except Exception:
                continue
            if not np.isfinite(float(p)):
                continue
            if float(p) > float(entry_f) and float(p) <= float(peak_f):
                fav_candidates.append(dict(r))
            if float(p) < float(entry_f) and float(p) >= float(trough_f):
                adv_candidates.append(dict(r))
        if fav_candidates:
            fav_candidates.sort(key=lambda rr: float(rr.get("price") or 0.0), reverse=True)
            fav = fav_candidates[0]
        if adv_candidates:
            adv_candidates.sort(key=lambda rr: float(rr.get("price") or 0.0))
            adv = adv_candidates[0]
        return fav, adv

    if s == "SHORT":
        fav_candidates2: list[dict[str, object]] = []
        adv_candidates2: list[dict[str, object]] = []
        for r in arr:
            try:
                p = float(r.get("price"))
            except Exception:
                continue
            if not np.isfinite(float(p)):
                continue
            if float(p) < float(entry_f) and float(p) >= float(trough_f):
                fav_candidates2.append(dict(r))
            if float(p) > float(entry_f) and float(p) <= float(peak_f):
                adv_candidates2.append(dict(r))
        if fav_candidates2:
            fav_candidates2.sort(key=lambda rr: float(rr.get("price") or 0.0))
            fav = fav_candidates2[0]
        if adv_candidates2:
            adv_candidates2.sort(key=lambda rr: float(rr.get("price") or 0.0), reverse=True)
            adv = adv_candidates2[0]
        return fav, adv

    raise ValueError(f"Unexpected side: {side}")


def _infer_tf_ms(ts: np.ndarray) -> int:
    if ts.size < 2:
        return 0
    diffs = np.diff(ts.astype(np.int64))
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0
    return int(np.median(diffs))


def _tf_from_ms(tf_ms: int) -> str:
    ms = int(tf_ms)
    if ms <= 0:
        return ""
    if ms % 60_000 == 0:
        minutes = int(ms // 60_000)
        if minutes % 60 == 0:
            hours = int(minutes // 60)
            return f"{hours}h"
        return f"{minutes}m"
    if ms % 3_600_000 == 0:
        return f"{int(ms // 3_600_000)}h"
    if ms % 86_400_000 == 0:
        return f"{int(ms // 86_400_000)}d"
    return ""


def _tp_price(*, side: str, entry: float, mode: str, tp_pct: float | None) -> float | None:
    m = str(mode).lower()
    if m in {"", "none", "off", "0"}:
        return None
    if m == "fixed_pct":
        if tp_pct is None:
            return None
        if side == "LONG":
            return float(entry) * (1.0 + float(tp_pct))
        if side == "SHORT":
            return float(entry) * (1.0 - float(tp_pct))
        raise ValueError(f"Unexpected side: {side}")
    if m == "pivot_grid":
        return None
    raise ValueError(f"Unexpected tp.mode: {mode}")


def _pivot_grid_tp_price(
    *,
    side: str,
    entry: float,
    tp_pct: float | None,
    pivot_table: list[dict[str, object]] | None,
) -> float | None:
    if not pivot_table:
        return None

    s = str(side).upper()
    if s == "LONG":
        entry_f = float(entry)
        target = entry_f if tp_pct is None else entry_f * (1.0 + float(tp_pct))
        best: float | None = None
        for r in pivot_table:
            try:
                p = float(r.get("price"))
            except Exception:
                continue
            if not np.isfinite(float(p)) or float(p) < float(target):
                continue
            if best is None or float(p) < float(best):
                best = float(p)
        if best is not None:
            return best

        best2: float | None = None
        for r in pivot_table:
            try:
                p = float(r.get("price"))
            except Exception:
                continue
            if not np.isfinite(float(p)) or float(p) <= float(entry_f):
                continue
            if best2 is None or float(p) < float(best2):
                best2 = float(p)
        return best2

    if s == "SHORT":
        entry_f = float(entry)
        target = entry_f if tp_pct is None else entry_f * (1.0 - float(tp_pct))
        best2: float | None = None
        for r in pivot_table:
            try:
                p = float(r.get("price"))
            except Exception:
                continue
            if not np.isfinite(float(p)) or float(p) > float(target):
                continue
            if best2 is None or float(p) > float(best2):
                best2 = float(p)
        if best2 is not None:
            return best2

        best3: float | None = None
        for r in pivot_table:
            try:
                p = float(r.get("price"))
            except Exception:
                continue
            if not np.isfinite(float(p)) or float(p) >= float(entry_f):
                continue
            if best3 is None or float(p) > float(best3):
                best3 = float(p)
        return best3

    raise ValueError(f"Unexpected side: {side}")


def _sl_price(*, side: str, pos: PositionState, mode: str, sl_pct: float | None, trail_pct: float | None, atr_mult: float | None) -> float | None:
    m = str(mode).lower()
    if m in {"", "none", "off", "0"}:
        return None

    if m == "fixed_pct":
        if sl_pct is None:
            return None
        if side == "LONG":
            return float(pos.entry_price) * (1.0 - float(sl_pct))
        if side == "SHORT":
            return float(pos.entry_price) * (1.0 + float(sl_pct))
        raise ValueError(f"Unexpected side: {side}")

    if m == "trailing_pct":
        if trail_pct is None:
            return None
        if side == "LONG":
            return float(pos.peak) * (1.0 - float(trail_pct))
        if side == "SHORT":
            return float(pos.trough) * (1.0 + float(trail_pct))
        raise ValueError(f"Unexpected side: {side}")

    if m == "atr":
        if atr_mult is None:
            return None
        if pos.atr_at_entry is None or (not np.isfinite(float(pos.atr_at_entry))):
            return None
        dist = float(atr_mult) * float(pos.atr_at_entry)
        if side == "LONG":
            return float(pos.entry_price) - dist
        if side == "SHORT":
            return float(pos.entry_price) + dist
        raise ValueError(f"Unexpected side: {side}")

    raise ValueError(f"Unexpected sl.mode: {mode}")


def _intrabar_exit(
    *,
    side: str,
    high: float,
    low: float,
    tp_price: float | None,
    sl_price: float | None,
) -> tuple[float, str] | None:
    if side == "LONG":
        tp_hit = tp_price is not None and float(high) >= float(tp_price)
        sl_hit = sl_price is not None and float(low) <= float(sl_price)
        if sl_hit:
            return float(sl_price), "SL"
        if tp_hit:
            return float(tp_price), "TP"
        return None

    if side == "SHORT":
        tp_hit = tp_price is not None and float(low) <= float(tp_price)
        sl_hit = sl_price is not None and float(high) >= float(sl_price)
        if sl_hit:
            return float(sl_price), "SL"
        if tp_hit:
            return float(tp_price), "TP"
        return None

    raise ValueError(f"Unexpected side: {side}")


def run_backtest_from_config(
    *,
    cfg: FullConfig,
    df: pd.DataFrame | None = None,
    start_ts: int | None = None,
    end_ts: int | None = None,
    ensure_indicators: bool = True,
) -> dict[str, object]:
    if df is None:
        df = pd.read_csv(Path(cfg.data.csv))
    if bool(ensure_indicators):
        df = ensure_indicators_df(df, cfg=cfg)

    ts_col = cfg.data.ts_col
    open_col = cfg.data.ohlc.open
    high_col = cfg.data.ohlc.high
    low_col = cfg.data.ohlc.low
    close_col = cfg.data.ohlc.close

    atr_len_eff = int(cfg.indicators.atr_len)
    if str(cfg.sl.mode).lower() == "atr" and cfg.sl.atr_len is not None:
        atr_len_eff = int(cfg.sl.atr_len)
    atr_col = f"atr_{int(atr_len_eff)}"

    n = int(len(df))
    if n < 3:
        raise ValueError(f"Not enough rows: {n}")

    ts_vals = pd.to_numeric(df[ts_col], errors="coerce").to_numpy()
    start_i = 0
    end_i = n - 1
    if start_ts is not None:
        start_i = int(np.searchsorted(ts_vals, int(start_ts), side="left"))
    if end_ts is not None:
        end_i = int(np.searchsorted(ts_vals, int(end_ts), side="right") - 1)
    if start_i < 0:
        start_i = 0
    if end_i > n - 1:
        end_i = n - 1
    if start_i > end_i:
        raise ValueError(f"Empty backtest range: start_ts={start_ts} end_ts={end_ts}")

    pos: PositionState | None = None
    pending_entry: dict[str, object] | None = None
    pending_exit: dict[str, object] | None = None
    last_entry_meta: dict[str, object] | None = None
    last_exit_signal_ts: int | None = None

    armed_side: str | None = None
    armed_meta: dict[str, object] | None = None
    armed_break_i: int | None = None
    armed_break_expire_i: int | None = None
    armed_break_ts: int | None = None

    equity = 0.0
    equity_points: list[dict[str, object]] = [
        {"ts": int(df[ts_col].iloc[start_i]), "equity": float(equity), "event": "START"}
    ]
    trades: list[TradeRecord] = []
    pivot_map_rows: list[dict[str, object]] = []

    win = int(cfg.backtest.window_size)
    fee_rate = float(cfg.backtest.fee_rate)
    min_net_for_signal_exit = 4.0 * float(fee_rate)

    pivot_regs_by_tf: dict[str, PivotRegistry] | None = None
    pivot_table: list[dict[str, object]] | None = None
    if str(cfg.tp.mode).strip().lower() == "pivot_grid":
        regs_cfg = dict(cfg.pivot_grid.registries or {})
        pivot_regs_by_tf = {
            "5m": PivotRegistry.from_json(str(regs_cfg["5m"])),
            "1h": PivotRegistry.from_json(str(regs_cfg["1h"])),
            "4h": PivotRegistry.from_json(str(regs_cfg["4h"])),
        }

    for i in range(start_i, end_i + 1):
        bar = df.iloc[i]
        ts = int(bar[ts_col])
        o = float(bar[open_col])
        h = float(bar[high_col])
        l = float(bar[low_col])

        if pivot_regs_by_tf is not None and np.isfinite(float(o)) and float(o) > 0:
            rr = get_or_refresh_execution_pivot_table(
                symbol=str(cfg.pivot_grid.symbol or ""),
                current_price=float(o),
                now_ts_ms=int(ts),
                regs_by_tf=pivot_regs_by_tf,
                grid_pct=float(cfg.pivot_grid.grid_pct),
                prev_table=(None if pivot_table is None else list(pivot_table)),
                min_supports=int(cfg.pivot_grid.min_supports),
                min_resistances=int(cfg.pivot_grid.min_resistances),
                keep_top2_5m=bool(cfg.pivot_grid.keep_top2_5m),
                mode=str(cfg.pivot_grid.mode),
                zones_cfg=cfg.pivot_grid.zones_cfg,
            )
            pivot_table = list(rr.get("table") or [])

        if pos is not None and pending_exit is not None and int(pending_exit["exec_i"]) == i:
            exit_price = float(o)
            gross = _gross_ret(side=pos.side, entry=pos.entry_price, exit_=exit_price)
            net = gross - (2.0 * fee_rate)

            peak2 = max(float(pos.peak), float(exit_price))
            trough2 = min(float(pos.trough), float(exit_price))
            if pos.side == "LONG":
                mfe = (float(peak2) / float(pos.entry_price)) - 1.0
                mae = (float(trough2) / float(pos.entry_price)) - 1.0
            else:
                mfe = (float(pos.entry_price) / float(trough2)) - 1.0
                mae = (float(pos.entry_price) / float(peak2)) - 1.0

            is_signal_exit = str(pending_exit.get("reason", "SIGNAL")).upper() == "SIGNAL"
            if is_signal_exit and ((not np.isfinite(float(net))) or float(net) < float(min_net_for_signal_exit)):
                pending_exit = None
                last_exit_signal_ts = None
            else:
                fav_pivot, adv_pivot = _pivot_touch_favorable_adverse(
                    pos.entry_pivot_table,
                    side=str(pos.side),
                    entry=float(pos.entry_price),
                    peak=float(peak2),
                    trough=float(trough2),
                )
                entry_supports, entry_resistances = _pivot_nearest_levels(
                    pos.entry_pivot_table,
                    current_price=float(pos.entry_price),
                    n=3,
                )
                exit_supports, exit_resistances = _pivot_nearest_levels(
                    pos.entry_pivot_table,
                    current_price=float(exit_price),
                    n=3,
                )
                row = {
                        "side": str(pos.side),
                        "entry_ts": int(pos.entry_ts),
                        "exit_ts": int(ts),
                        "entry_price": float(pos.entry_price),
                        "exit_price": float(exit_price),
                        "exit_reason": str(pending_exit.get("reason", "SIGNAL")),
                        "tp_pivot_price": (None if pos.tp_pivot is None else pos.tp_pivot.get("price")),
                        "tp_pivot_weight": (None if pos.tp_pivot is None else pos.tp_pivot.get("weight")),
                        "tp_pivot_zone_id": (None if pos.tp_pivot is None else pos.tp_pivot.get("zone_id")),
                        "tp_pivot_event_id": (None if pos.tp_pivot is None else pos.tp_pivot.get("event_id")),
                        "fav_pivot_price": (None if fav_pivot is None else fav_pivot.get("price")),
                        "fav_pivot_weight": (None if fav_pivot is None else fav_pivot.get("weight")),
                        "adv_pivot_price": (None if adv_pivot is None else adv_pivot.get("price")),
                        "adv_pivot_weight": (None if adv_pivot is None else adv_pivot.get("weight")),
                        "entry_support1_price": (None if len(entry_supports) < 1 else entry_supports[0].get("price")),
                        "entry_support1_weight": (None if len(entry_supports) < 1 else entry_supports[0].get("weight")),
                        "entry_support2_price": (None if len(entry_supports) < 2 else entry_supports[1].get("price")),
                        "entry_support2_weight": (None if len(entry_supports) < 2 else entry_supports[1].get("weight")),
                        "entry_support3_price": (None if len(entry_supports) < 3 else entry_supports[2].get("price")),
                        "entry_support3_weight": (None if len(entry_supports) < 3 else entry_supports[2].get("weight")),
                        "entry_res1_price": (None if len(entry_resistances) < 1 else entry_resistances[0].get("price")),
                        "entry_res1_weight": (None if len(entry_resistances) < 1 else entry_resistances[0].get("weight")),
                        "entry_res2_price": (None if len(entry_resistances) < 2 else entry_resistances[1].get("price")),
                        "entry_res2_weight": (None if len(entry_resistances) < 2 else entry_resistances[1].get("weight")),
                        "entry_res3_price": (None if len(entry_resistances) < 3 else entry_resistances[2].get("price")),
                        "entry_res3_weight": (None if len(entry_resistances) < 3 else entry_resistances[2].get("weight")),
                        "exit_support1_price": (None if len(exit_supports) < 1 else exit_supports[0].get("price")),
                        "exit_support1_weight": (None if len(exit_supports) < 1 else exit_supports[0].get("weight")),
                        "exit_res1_price": (None if len(exit_resistances) < 1 else exit_resistances[0].get("price")),
                        "exit_res1_weight": (None if len(exit_resistances) < 1 else exit_resistances[0].get("weight")),
                }
                row.update(
                    _pivot_confluence_metrics(
                        pivot_regs_by_tf,
                        cfg=cfg,
                        target_price=_as_float(row.get("tp_pivot_price")),
                        ref_price=float(pos.entry_price),
                        prefix="tp",
                    )
                )
                row.update(
                    _pivot_confluence_metrics(
                        pivot_regs_by_tf,
                        cfg=cfg,
                        target_price=_as_float(row.get("fav_pivot_price")),
                        ref_price=float(pos.entry_price),
                        prefix="fav",
                    )
                )
                row.update(
                    _pivot_confluence_metrics(
                        pivot_regs_by_tf,
                        cfg=cfg,
                        target_price=_as_float(row.get("adv_pivot_price")),
                        ref_price=float(pos.entry_price),
                        prefix="adv",
                    )
                )
                row.update(
                    _pivot_confluence_metrics(
                        pivot_regs_by_tf,
                        cfg=cfg,
                        target_price=_as_float(row.get("entry_support1_price")),
                        ref_price=float(pos.entry_price),
                        prefix="entry_support1",
                    )
                )
                row.update(
                    _pivot_confluence_metrics(
                        pivot_regs_by_tf,
                        cfg=cfg,
                        target_price=_as_float(row.get("entry_res1_price")),
                        ref_price=float(pos.entry_price),
                        prefix="entry_res1",
                    )
                )
                row.update(
                    _pivot_confluence_metrics(
                        pivot_regs_by_tf,
                        cfg=cfg,
                        target_price=_as_float(row.get("exit_support1_price")),
                        ref_price=float(exit_price),
                        prefix="exit_support1",
                    )
                )
                row.update(
                    _pivot_confluence_metrics(
                        pivot_regs_by_tf,
                        cfg=cfg,
                        target_price=_as_float(row.get("exit_res1_price")),
                        ref_price=float(exit_price),
                        prefix="exit_res1",
                    )
                )
                pivot_map_rows.append(row)

                trades.append(
                    TradeRecord(
                        side=pos.side,
                        entry_ts=int(pos.entry_ts),
                        exit_ts=int(ts),
                        entry_price=float(pos.entry_price),
                        exit_price=float(exit_price),
                        exit_reason=str(pending_exit.get("reason", "SIGNAL")),
                        entry_signal_ts=(None if last_entry_meta is None else int(last_entry_meta.get("signal_ts") or 0) or None),
                        entry_signal_cand_ts=(
                            None if last_entry_meta is None else int(last_entry_meta.get("cand_ts") or 0) or None
                        ),
                        entry_confirmed_series=(
                            None
                            if last_entry_meta is None
                            else "|".join(list(last_entry_meta.get("confirmed_series") or []))
                        ),
                        entry_trend_filter=(None if last_entry_meta is None else last_entry_meta.get("trend_filter")),
                        entry_trend_ok=(None if last_entry_meta is None else last_entry_meta.get("trend_ok")),
                        entry_trend_vortex_side=(
                            None if last_entry_meta is None else last_entry_meta.get("trend_vortex_side")
                        ),
                        entry_trend_dmi_side=(None if last_entry_meta is None else last_entry_meta.get("trend_dmi_side")),
                        exit_signal_ts=(None if last_exit_signal_ts is None else int(last_exit_signal_ts)),
                        gross_ret=float(gross),
                        net_ret=float(net),
                        mfe=float(mfe),
                        mae=float(mae),
                    )
                )

                equity += float(net)
                equity_points.append({"ts": int(ts), "equity": float(equity), "event": "EXIT"})

                pos = None
                pending_exit = None
                last_exit_signal_ts = None

        if pos is None and pending_entry is not None and int(pending_entry["exec_i"]) == i:
            side = str(pending_entry["side"]).upper()

            if str(cfg.signals_entry.entry_mode).strip().lower() == "vwma_break":
                if i - 1 >= 0:
                    vwma_fast_col = f"vwma_{int(cfg.indicators.vwma_fast)}"
                    v_prev = float(pd.to_numeric(df[vwma_fast_col].iloc[i - 1], errors="coerce"))
                    if np.isfinite(float(v_prev)):
                        if side == "LONG" and not (float(o) > float(v_prev)):
                            pending_entry = None
                            armed_break_i = None
                            armed_break_expire_i = None
                            armed_break_ts = None
                            continue
                        if side == "SHORT" and not (float(o) < float(v_prev)):
                            pending_entry = None
                            armed_break_i = None
                            armed_break_expire_i = None
                            armed_break_ts = None
                            continue

            atr_val = None
            if i - 1 >= 0:
                atr_val = float(pd.to_numeric(df[atr_col].iloc[i - 1], errors="coerce"))
                if not np.isfinite(float(atr_val)):
                    atr_val = None

            tp_for_entry = (
                None
                if str(cfg.tp.mode).strip().lower() != "pivot_grid"
                else _pivot_grid_tp_price(
                    side=str(side),
                    entry=float(o),
                    tp_pct=cfg.tp.tp_pct,
                    pivot_table=pivot_table,
                )
            )

            pos = PositionState(
                side=side,
                entry_i=int(i),
                entry_ts=int(ts),
                entry_price=float(o),
                tp_price=tp_for_entry,
                tp_pivot=(
                    None
                    if str(cfg.tp.mode).strip().lower() != "pivot_grid"
                    else _pivot_row_for_price(pivot_table, price=tp_for_entry)
                ),
                entry_pivot_table=(None if pivot_table is None else list(pivot_table)),
                peak=float(o),
                trough=float(o),
                atr_at_entry=atr_val,
            )
            pending_entry = None
            last_exit_signal_ts = None
            armed_side = None
            armed_meta = None
            armed_break_i = None
            armed_break_expire_i = None
            armed_break_ts = None

        if pos is not None:
            prev_peak = float(pos.peak)
            prev_trough = float(pos.trough)

            if str(cfg.tp.mode).strip().lower() == "pivot_grid":
                tp_p = None if pos.tp_price is None else float(pos.tp_price)
            else:
                tp_p = _tp_price(side=pos.side, entry=pos.entry_price, mode=cfg.tp.mode, tp_pct=cfg.tp.tp_pct)
            sl_p = _sl_price(
                side=pos.side,
                pos=pos,
                mode=cfg.sl.mode,
                sl_pct=cfg.sl.sl_pct,
                trail_pct=cfg.sl.trail_pct,
                atr_mult=cfg.sl.atr_mult,
            )

            hit = _intrabar_exit(side=pos.side, high=h, low=l, tp_price=tp_p, sl_price=sl_p)
            if hit is not None:
                exit_price, reason = hit
                gross = _gross_ret(side=pos.side, entry=pos.entry_price, exit_=exit_price)
                net = gross - (2.0 * fee_rate)

                peak2 = max(prev_peak, float(h))
                trough2 = min(prev_trough, float(l))

                if pos.side == "LONG":
                    mfe = (float(peak2) / float(pos.entry_price)) - 1.0
                    mae = (float(trough2) / float(pos.entry_price)) - 1.0
                else:
                    mfe = (float(pos.entry_price) / float(trough2)) - 1.0
                    mae = (float(pos.entry_price) / float(peak2)) - 1.0

                fav_pivot, adv_pivot = _pivot_touch_favorable_adverse(
                    pos.entry_pivot_table,
                    side=str(pos.side),
                    entry=float(pos.entry_price),
                    peak=float(peak2),
                    trough=float(trough2),
                )
                entry_supports, entry_resistances = _pivot_nearest_levels(
                    pos.entry_pivot_table,
                    current_price=float(pos.entry_price),
                    n=3,
                )
                exit_supports, exit_resistances = _pivot_nearest_levels(
                    pos.entry_pivot_table,
                    current_price=float(exit_price),
                    n=3,
                )
                row = {
                    "side": str(pos.side),
                    "entry_ts": int(pos.entry_ts),
                    "exit_ts": int(ts),
                    "entry_price": float(pos.entry_price),
                    "exit_price": float(exit_price),
                    "exit_reason": str(reason),
                    "tp_pivot_price": (None if pos.tp_pivot is None else pos.tp_pivot.get("price")),
                    "tp_pivot_weight": (None if pos.tp_pivot is None else pos.tp_pivot.get("weight")),
                    "tp_pivot_zone_id": (None if pos.tp_pivot is None else pos.tp_pivot.get("zone_id")),
                    "tp_pivot_event_id": (None if pos.tp_pivot is None else pos.tp_pivot.get("event_id")),
                    "fav_pivot_price": (None if fav_pivot is None else fav_pivot.get("price")),
                    "fav_pivot_weight": (None if fav_pivot is None else fav_pivot.get("weight")),
                    "adv_pivot_price": (None if adv_pivot is None else adv_pivot.get("price")),
                    "adv_pivot_weight": (None if adv_pivot is None else adv_pivot.get("weight")),
                    "entry_support1_price": (None if len(entry_supports) < 1 else entry_supports[0].get("price")),
                    "entry_support1_weight": (None if len(entry_supports) < 1 else entry_supports[0].get("weight")),
                    "entry_support2_price": (None if len(entry_supports) < 2 else entry_supports[1].get("price")),
                    "entry_support2_weight": (None if len(entry_supports) < 2 else entry_supports[1].get("weight")),
                    "entry_support3_price": (None if len(entry_supports) < 3 else entry_supports[2].get("price")),
                    "entry_support3_weight": (None if len(entry_supports) < 3 else entry_supports[2].get("weight")),
                    "entry_res1_price": (None if len(entry_resistances) < 1 else entry_resistances[0].get("price")),
                    "entry_res1_weight": (None if len(entry_resistances) < 1 else entry_resistances[0].get("weight")),
                    "entry_res2_price": (None if len(entry_resistances) < 2 else entry_resistances[1].get("price")),
                    "entry_res2_weight": (None if len(entry_resistances) < 2 else entry_resistances[1].get("weight")),
                    "entry_res3_price": (None if len(entry_resistances) < 3 else entry_resistances[2].get("price")),
                    "entry_res3_weight": (None if len(entry_resistances) < 3 else entry_resistances[2].get("weight")),
                    "exit_support1_price": (None if len(exit_supports) < 1 else exit_supports[0].get("price")),
                    "exit_support1_weight": (None if len(exit_supports) < 1 else exit_supports[0].get("weight")),
                    "exit_res1_price": (None if len(exit_resistances) < 1 else exit_resistances[0].get("price")),
                    "exit_res1_weight": (None if len(exit_resistances) < 1 else exit_resistances[0].get("weight")),
                }
                row.update(
                    _pivot_confluence_metrics(
                        pivot_regs_by_tf,
                        cfg=cfg,
                        target_price=_as_float(row.get("tp_pivot_price")),
                        ref_price=float(pos.entry_price),
                        prefix="tp",
                    )
                )
                row.update(
                    _pivot_confluence_metrics(
                        pivot_regs_by_tf,
                        cfg=cfg,
                        target_price=_as_float(row.get("fav_pivot_price")),
                        ref_price=float(pos.entry_price),
                        prefix="fav",
                    )
                )
                row.update(
                    _pivot_confluence_metrics(
                        pivot_regs_by_tf,
                        cfg=cfg,
                        target_price=_as_float(row.get("adv_pivot_price")),
                        ref_price=float(pos.entry_price),
                        prefix="adv",
                    )
                )
                row.update(
                    _pivot_confluence_metrics(
                        pivot_regs_by_tf,
                        cfg=cfg,
                        target_price=_as_float(row.get("entry_support1_price")),
                        ref_price=float(pos.entry_price),
                        prefix="entry_support1",
                    )
                )
                row.update(
                    _pivot_confluence_metrics(
                        pivot_regs_by_tf,
                        cfg=cfg,
                        target_price=_as_float(row.get("entry_res1_price")),
                        ref_price=float(pos.entry_price),
                        prefix="entry_res1",
                    )
                )
                row.update(
                    _pivot_confluence_metrics(
                        pivot_regs_by_tf,
                        cfg=cfg,
                        target_price=_as_float(row.get("exit_support1_price")),
                        ref_price=float(exit_price),
                        prefix="exit_support1",
                    )
                )
                row.update(
                    _pivot_confluence_metrics(
                        pivot_regs_by_tf,
                        cfg=cfg,
                        target_price=_as_float(row.get("exit_res1_price")),
                        ref_price=float(exit_price),
                        prefix="exit_res1",
                    )
                )
                pivot_map_rows.append(row)

                trades.append(
                    TradeRecord(
                        side=pos.side,
                        entry_ts=int(pos.entry_ts),
                        exit_ts=int(ts),
                        entry_price=float(pos.entry_price),
                        exit_price=float(exit_price),
                        exit_reason=str(reason),
                        entry_signal_ts=(None if last_entry_meta is None else int(last_entry_meta.get("signal_ts") or 0) or None),
                        entry_signal_cand_ts=(
                            None if last_entry_meta is None else int(last_entry_meta.get("cand_ts") or 0) or None
                        ),
                        entry_confirmed_series=(
                            None
                            if last_entry_meta is None
                            else "|".join(list(last_entry_meta.get("confirmed_series") or []))
                        ),
                        entry_trend_filter=(None if last_entry_meta is None else last_entry_meta.get("trend_filter")),
                        entry_trend_ok=(None if last_entry_meta is None else last_entry_meta.get("trend_ok")),
                        entry_trend_vortex_side=(
                            None if last_entry_meta is None else last_entry_meta.get("trend_vortex_side")
                        ),
                        entry_trend_dmi_side=(None if last_entry_meta is None else last_entry_meta.get("trend_dmi_side")),
                        exit_signal_ts=None,
                        gross_ret=float(gross),
                        net_ret=float(net),
                        mfe=float(mfe),
                        mae=float(mae),
                    )
                )
                equity += float(net)
                equity_points.append({"ts": int(ts), "equity": float(equity), "event": str(reason)})

                pos = None
                pending_exit = None
                last_exit_signal_ts = None
            else:
                pos.peak = max(prev_peak, float(h))
                pos.trough = min(prev_trough, float(l))

        if i >= end_i:
            continue

        win_start_i = max(0, int(i) - int(win) + 1)
        window = df.iloc[win_start_i : i + 1]

        if pos is None:
            if pending_entry is None:
                if armed_side is None:
                    dec = trigger_signal(window, cfg=cfg)
                    if dec.side in ("LONG", "SHORT") and bool(cfg.pivot_temporal_memory.enabled):
                        preset = get_extreme_confluence_preset(cfg.signals_trigger.name)
                        cci_fast_thr = (
                            cfg.signals_trigger.cci_fast_threshold
                            if cfg.signals_trigger.cci_fast_threshold is not None
                            else preset.cci_fast_threshold
                        )
                        cci_medium_thr = (
                            cfg.signals_trigger.cci_medium_threshold
                            if cfg.signals_trigger.cci_medium_threshold is not None
                            else preset.cci_medium_threshold
                        )
                        cci_slow_thr = (
                            cfg.signals_trigger.cci_slow_threshold
                            if cfg.signals_trigger.cci_slow_threshold is not None
                            else preset.cci_slow_threshold
                        )

                        eps = float(cfg.pivot_temporal_memory.radius_pct)
                        if not np.isfinite(float(eps)) or float(eps) <= 0:
                            eps = 0.01

                        tf_ms = _infer_tf_ms(pd.to_numeric(window[ts_col], errors="coerce").dropna().to_numpy())
                        tf_s = _tf_from_ms(int(tf_ms))
                        if not tf_s:
                            tf_s = "5m"

                        symbol = str(cfg.pivot_grid.symbol or "")
                        if not symbol:
                            symbol = "unknown"

                        piv = PivotRegistry.empty(symbol=str(symbol), tf=str(tf_s), eps=float(eps))
                        cci_fast_col = f"cci_{int(cfg.indicators.cci_fast)}"
                        cci_medium_col = f"cci_{int(cfg.indicators.cci_medium)}"
                        cci_slow_col = f"cci_{int(cfg.indicators.cci_slow)}"
                        try:
                            extremes_df = extract_window_close_extremes(
                                window,
                                ts_col=str(ts_col),
                                close_col=str(close_col),
                                hist_col="macd_hist",
                                cci_fast_col=str(cci_fast_col),
                                cci_medium_col=str(cci_medium_col),
                                cci_slow_col=str(cci_slow_col),
                                cci_fast_threshold=float(cci_fast_thr),
                                cci_medium_threshold=float(cci_medium_thr),
                                cci_slow_threshold=float(cci_slow_thr),
                                zone_radius_pct=float(cfg.pivot_temporal_memory.radius_pct),
                                max_bars_ago=int(win),
                            )
                            piv.update_from_extremes_df(extremes_df)
                            dec = trigger_signal(window, cfg=cfg, pivot_registry=piv)
                        except Exception:
                            dec = dec
                    if dec.side in ("LONG", "SHORT"):
                        mode = str(cfg.signals_entry.tranche_hist_trend_mode).strip().lower()
                        if mode == "healthy":
                            ok_tranche = bool(dec.meta.get("tranche_hist_trend_is_healthy")) and bool(
                                dec.meta.get("tranche_hist_trend_contrarian_ok")
                            )
                        elif mode == "trend":
                            ok_tranche = bool(dec.meta.get("tranche_hist_trend_is_trend")) and bool(
                                dec.meta.get("tranche_hist_trend_contrarian_ok")
                            )
                        else:
                            ok_tranche = True

                        if bool(ok_tranche):
                            armed_side = str(dec.side)
                            armed_meta = dict(dec.meta)
                            armed_meta["signal_ts"] = int(ts)
                            armed_break_i = None
                            armed_break_expire_i = None
                            armed_break_ts = None

                if armed_side is not None:
                    if str(cfg.signals_entry.entry_mode).strip().lower() == "vwma_break":
                        close_col = str(cfg.data.ohlc.close)
                        vwma_fast_col = f"vwma_{int(cfg.indicators.vwma_fast)}"
                        if {close_col, vwma_fast_col}.issubset(set(window.columns)) and len(window) >= 2:
                            c_prev = float(pd.to_numeric(window[close_col].iloc[-2], errors="coerce"))
                            c_now = float(pd.to_numeric(window[close_col].iloc[-1], errors="coerce"))
                            v_prev = float(pd.to_numeric(window[vwma_fast_col].iloc[-2], errors="coerce"))
                            v_now = float(pd.to_numeric(window[vwma_fast_col].iloc[-1], errors="coerce"))
                            if np.isfinite(float(c_prev)) and np.isfinite(float(c_now)) and np.isfinite(float(v_prev)) and np.isfinite(float(v_now)):
                                if str(armed_side).upper() == "LONG":
                                    is_break = (float(c_prev) <= float(v_prev)) and (float(c_now) > float(v_now))
                                    is_opposite_break = (float(c_prev) >= float(v_prev)) and (float(c_now) < float(v_now))
                                    is_beyond = float(c_now) > float(v_now)
                                else:
                                    is_break = (float(c_prev) >= float(v_prev)) and (float(c_now) < float(v_now))
                                    is_opposite_break = (float(c_prev) <= float(v_prev)) and (float(c_now) > float(v_now))
                                    is_beyond = float(c_now) < float(v_now)

                                if armed_break_i is None:
                                    if bool(is_break):
                                        armed_break_i = int(i)
                                        armed_break_ts = int(ts)
                                        armed_break_expire_i = int(i) + int(cfg.signals_entry.vwma_break_max_bars)

                                        ok, meta = price_action_filters_ok(
                                            window,
                                            cfg=cfg,
                                            pa=cfg.signals_entry,
                                            side=str(armed_side),
                                            phase="entry",
                                        )
                                        if bool(ok) and bool(is_beyond):
                                            pending_entry = {
                                                "exec_i": int(i + 1),
                                                "side": str(armed_side),
                                                "signal_ts": int(ts),
                                            }
                                            m = dict(armed_meta or {})
                                            m.update(dict(meta))
                                            m["signal_ts"] = int(ts)
                                            m["vwma_break_ts"] = (None if armed_break_ts is None else int(armed_break_ts))
                                            m["vwma_break_i"] = (None if armed_break_i is None else int(armed_break_i))
                                            m["vwma_break_max_bars"] = int(cfg.signals_entry.vwma_break_max_bars)
                                            last_entry_meta = m
                                        else:
                                            last_entry_meta = dict(armed_meta or {})
                                            last_entry_meta.update(dict(meta))
                                            last_entry_meta["vwma_break_ts"] = (None if armed_break_ts is None else int(armed_break_ts))
                                            last_entry_meta["vwma_break_i"] = (None if armed_break_i is None else int(armed_break_i))
                                            last_entry_meta["vwma_break_max_bars"] = int(cfg.signals_entry.vwma_break_max_bars)
                                else:
                                    if bool(is_opposite_break):
                                        armed_break_i = None
                                        armed_break_ts = None
                                        armed_break_expire_i = None
                                    elif armed_break_expire_i is not None and int(i) > int(armed_break_expire_i):
                                        armed_break_i = None
                                        armed_break_ts = None
                                        armed_break_expire_i = None
                                    elif bool(is_beyond):
                                        ok, meta = price_action_filters_ok(
                                            window,
                                            cfg=cfg,
                                            pa=cfg.signals_entry,
                                            side=str(armed_side),
                                            phase="entry",
                                        )
                                        if bool(ok):
                                            pending_entry = {
                                                "exec_i": int(i + 1),
                                                "side": str(armed_side),
                                                "signal_ts": int(ts),
                                            }
                                            m = dict(armed_meta or {})
                                            m.update(dict(meta))
                                            m["signal_ts"] = int(ts)
                                            m["vwma_break_ts"] = (None if armed_break_ts is None else int(armed_break_ts))
                                            m["vwma_break_i"] = (None if armed_break_i is None else int(armed_break_i))
                                            m["vwma_break_max_bars"] = int(cfg.signals_entry.vwma_break_max_bars)
                                            last_entry_meta = m
                                        else:
                                            last_entry_meta = dict(armed_meta or {})
                                            last_entry_meta.update(dict(meta))
                                            last_entry_meta["vwma_break_ts"] = (None if armed_break_ts is None else int(armed_break_ts))
                                            last_entry_meta["vwma_break_i"] = (None if armed_break_i is None else int(armed_break_i))
                                            last_entry_meta["vwma_break_max_bars"] = int(cfg.signals_entry.vwma_break_max_bars)
                    else:
                        ok, meta = price_action_filters_ok(
                            window,
                            cfg=cfg,
                            pa=cfg.signals_entry,
                            side=str(armed_side),
                            phase="entry",
                        )
                        if bool(ok):
                            pending_entry = {"exec_i": int(i + 1), "side": str(armed_side), "signal_ts": int(ts)}
                            m = dict(armed_meta or {})
                            m.update(dict(meta))
                            m["signal_ts"] = int(ts)
                            last_entry_meta = m
                        else:
                            last_entry_meta = dict(armed_meta or {})
                            last_entry_meta.update(dict(meta))
        else:
            if bool(cfg.exit_policy.allow_exit_signal) and pending_exit is None:
                ok, _meta = price_action_filters_ok(
                    window,
                    cfg=cfg,
                    pa=cfg.signals_exit,
                    side=str(pos.side),
                    phase="exit",
                )
                if bool(ok):
                    pending_exit = {"exec_i": int(i + 1), "reason": "SIGNAL", "signal_ts": int(ts)}
                    last_exit_signal_ts = int(ts)

    if pos is not None:
        last_bar = df.iloc[int(end_i)]
        ts = int(last_bar[ts_col])
        exit_price = float(last_bar[close_col])
        gross = _gross_ret(side=pos.side, entry=pos.entry_price, exit_=exit_price)
        net = gross - (2.0 * fee_rate)

        if pos.side == "LONG":
            mfe = (float(pos.peak) / float(pos.entry_price)) - 1.0
            mae = (float(pos.trough) / float(pos.entry_price)) - 1.0
        else:
            mfe = (float(pos.entry_price) / float(pos.trough)) - 1.0
            mae = (float(pos.entry_price) / float(pos.peak)) - 1.0

        fav_pivot, adv_pivot = _pivot_touch_favorable_adverse(
            pos.entry_pivot_table,
            side=str(pos.side),
            entry=float(pos.entry_price),
            peak=float(pos.peak),
            trough=float(pos.trough),
        )
        entry_supports, entry_resistances = _pivot_nearest_levels(
            pos.entry_pivot_table,
            current_price=float(pos.entry_price),
            n=3,
        )
        exit_supports, exit_resistances = _pivot_nearest_levels(
            pos.entry_pivot_table,
            current_price=float(exit_price),
            n=3,
        )
        row = {
                "side": str(pos.side),
                "entry_ts": int(pos.entry_ts),
                "exit_ts": int(ts),
                "entry_price": float(pos.entry_price),
                "exit_price": float(exit_price),
                "exit_reason": "EOD",
                "tp_pivot_price": (None if pos.tp_pivot is None else pos.tp_pivot.get("price")),
                "tp_pivot_weight": (None if pos.tp_pivot is None else pos.tp_pivot.get("weight")),
                "tp_pivot_zone_id": (None if pos.tp_pivot is None else pos.tp_pivot.get("zone_id")),
                "tp_pivot_event_id": (None if pos.tp_pivot is None else pos.tp_pivot.get("event_id")),
                "fav_pivot_price": (None if fav_pivot is None else fav_pivot.get("price")),
                "fav_pivot_weight": (None if fav_pivot is None else fav_pivot.get("weight")),
                "adv_pivot_price": (None if adv_pivot is None else adv_pivot.get("price")),
                "adv_pivot_weight": (None if adv_pivot is None else adv_pivot.get("weight")),
                "entry_support1_price": (None if len(entry_supports) < 1 else entry_supports[0].get("price")),
                "entry_support1_weight": (None if len(entry_supports) < 1 else entry_supports[0].get("weight")),
                "entry_support2_price": (None if len(entry_supports) < 2 else entry_supports[1].get("price")),
                "entry_support2_weight": (None if len(entry_supports) < 2 else entry_supports[1].get("weight")),
                "entry_support3_price": (None if len(entry_supports) < 3 else entry_supports[2].get("price")),
                "entry_support3_weight": (None if len(entry_supports) < 3 else entry_supports[2].get("weight")),
                "entry_res1_price": (None if len(entry_resistances) < 1 else entry_resistances[0].get("price")),
                "entry_res1_weight": (None if len(entry_resistances) < 1 else entry_resistances[0].get("weight")),
                "entry_res2_price": (None if len(entry_resistances) < 2 else entry_resistances[1].get("price")),
                "entry_res2_weight": (None if len(entry_resistances) < 2 else entry_resistances[1].get("weight")),
                "entry_res3_price": (None if len(entry_resistances) < 3 else entry_resistances[2].get("price")),
                "entry_res3_weight": (None if len(entry_resistances) < 3 else entry_resistances[2].get("weight")),
                "exit_support1_price": (None if len(exit_supports) < 1 else exit_supports[0].get("price")),
                "exit_support1_weight": (None if len(exit_supports) < 1 else exit_supports[0].get("weight")),
                "exit_res1_price": (None if len(exit_resistances) < 1 else exit_resistances[0].get("price")),
                "exit_res1_weight": (None if len(exit_resistances) < 1 else exit_resistances[0].get("weight")),
        }
        row.update(
            _pivot_confluence_metrics(
                pivot_regs_by_tf,
                cfg=cfg,
                target_price=_as_float(row.get("tp_pivot_price")),
                ref_price=float(pos.entry_price),
                prefix="tp",
            )
        )
        row.update(
            _pivot_confluence_metrics(
                pivot_regs_by_tf,
                cfg=cfg,
                target_price=_as_float(row.get("fav_pivot_price")),
                ref_price=float(pos.entry_price),
                prefix="fav",
            )
        )
        row.update(
            _pivot_confluence_metrics(
                pivot_regs_by_tf,
                cfg=cfg,
                target_price=_as_float(row.get("adv_pivot_price")),
                ref_price=float(pos.entry_price),
                prefix="adv",
            )
        )
        row.update(
            _pivot_confluence_metrics(
                pivot_regs_by_tf,
                cfg=cfg,
                target_price=_as_float(row.get("entry_support1_price")),
                ref_price=float(pos.entry_price),
                prefix="entry_support1",
            )
        )
        row.update(
            _pivot_confluence_metrics(
                pivot_regs_by_tf,
                cfg=cfg,
                target_price=_as_float(row.get("entry_res1_price")),
                ref_price=float(pos.entry_price),
                prefix="entry_res1",
            )
        )
        row.update(
            _pivot_confluence_metrics(
                pivot_regs_by_tf,
                cfg=cfg,
                target_price=_as_float(row.get("exit_support1_price")),
                ref_price=float(exit_price),
                prefix="exit_support1",
            )
        )
        row.update(
            _pivot_confluence_metrics(
                pivot_regs_by_tf,
                cfg=cfg,
                target_price=_as_float(row.get("exit_res1_price")),
                ref_price=float(exit_price),
                prefix="exit_res1",
            )
        )
        pivot_map_rows.append(row)

        trades.append(
            TradeRecord(
                side=pos.side,
                entry_ts=int(pos.entry_ts),
                exit_ts=int(ts),
                entry_price=float(pos.entry_price),
                exit_price=float(exit_price),
                exit_reason="EOD",
                entry_signal_ts=(None if last_entry_meta is None else int(last_entry_meta.get("signal_ts") or 0) or None),
                entry_signal_cand_ts=(None if last_entry_meta is None else int(last_entry_meta.get("cand_ts") or 0) or None),
                entry_confirmed_series=(
                    None if last_entry_meta is None else "|".join(list(last_entry_meta.get("confirmed_series") or []))
                ),
                entry_trend_filter=(None if last_entry_meta is None else last_entry_meta.get("trend_filter")),
                entry_trend_ok=(None if last_entry_meta is None else last_entry_meta.get("trend_ok")),
                entry_trend_vortex_side=(None if last_entry_meta is None else last_entry_meta.get("trend_vortex_side")),
                entry_trend_dmi_side=(None if last_entry_meta is None else last_entry_meta.get("trend_dmi_side")),
                exit_signal_ts=None,
                gross_ret=float(gross),
                net_ret=float(net),
                mfe=float(mfe),
                mae=float(mae),
            )
        )
        equity += float(net)
        equity_points.append({"ts": int(ts), "equity": float(equity), "event": "EOD"})

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    if len(trades_df) and "mae" in trades_df.columns and "dd_float" not in trades_df.columns:
        trades_df["dd_float"] = trades_df["mae"]
    equity_df = pd.DataFrame(equity_points)
    pivot_map_df = pd.DataFrame(pivot_map_rows)

    equity_df["peak"] = equity_df["equity"].cummax()
    equity_df["dd"] = equity_df["equity"] - equity_df["peak"]
    max_dd = float(equity_df["dd"].min()) if len(equity_df) else 0.0

    equity_end = float(equity_df["equity"].iloc[-1]) if len(equity_df) else 0.0
    if equity_end <= 0.0:
        ratio = 0.0
    else:
        ratio = float("inf") if max_dd == 0.0 else float(equity_end) / abs(float(max_dd))

    wins_mask = pd.to_numeric(trades_df.get("net_ret"), errors="coerce") > 0.0 if len(trades_df) else pd.Series([], dtype=bool)
    losses_mask = pd.to_numeric(trades_df.get("net_ret"), errors="coerce") < 0.0 if len(trades_df) else pd.Series([], dtype=bool)
    n_wins = int(wins_mask.sum()) if len(trades_df) else 0
    n_losses = int(losses_mask.sum()) if len(trades_df) else 0
    winrate = float(n_wins) / float(len(trades_df)) if len(trades_df) else 0.0
    avg_win = float(pd.to_numeric(trades_df.loc[wins_mask, "net_ret"], errors="coerce").mean()) if n_wins else 0.0
    avg_loss = float(pd.to_numeric(trades_df.loc[losses_mask, "net_ret"], errors="coerce").mean()) if n_losses else 0.0

    return {
        "df": df,
        "trades": trades_df,
        "equity": equity_df,
        "pivot_map": pivot_map_df,
        "summary": {
            "n_trades": int(len(trades_df)),
            "equity_end": float(equity_end),
            "max_dd": float(max_dd),
            "ratio": float(ratio),
            "n_wins": int(n_wins),
            "n_losses": int(n_losses),
            "winrate": float(winrate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
        },
    }
