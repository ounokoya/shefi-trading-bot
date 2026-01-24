from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from libs.backtest_price_action_tranche.config import FullConfig, PriceActionSignalConfig, TriggerSignalConfig
from libs.blocks.get_current_tranche_extreme_zone_confluence_signal import (
    get_current_tranche_extreme_zone_confluence_signal,
    get_current_tranche_extreme_zone_confluence_tranche_last_signal,
)
from libs.pivots.pivot_registry import PivotRegistry
from libs.presets.extreme_confluence_presets import get_extreme_confluence_preset


@dataclass(frozen=True)
class TriggerDecision:
    side: str | None
    meta: dict[str, object]


def _target_close_extreme_kind(mode: str) -> str | None:
    m = str(mode).lower()
    if m == "long":
        return "LOW"
    if m == "short":
        return "HIGH"
    if m == "both":
        return None
    raise ValueError(f"Unexpected mode: {mode}")


def _opposite_side(side: str) -> str:
    s = str(side).upper()
    if s == "LONG":
        return "SHORT"
    if s == "SHORT":
        return "LONG"
    raise ValueError(f"Unexpected side: {side}")


def _confluence_signal(
    window: pd.DataFrame,
    *,
    cfg: FullConfig,
    sig: TriggerSignalConfig,
    trend_filter: str,
) -> dict[str, object]:
    preset = get_extreme_confluence_preset(sig.name)

    cci_fast_col = f"cci_{int(cfg.indicators.cci_fast)}"
    cci_medium_col = f"cci_{int(cfg.indicators.cci_medium)}"
    cci_slow_col = f"cci_{int(cfg.indicators.cci_slow)}"
    vwma_fast_col = f"vwma_{int(cfg.indicators.vwma_fast)}"
    vwma_medium_col = f"vwma_{int(cfg.indicators.vwma_medium)}"

    aliases = {
        "cci_fast": cci_fast_col,
        "cci_medium": cci_medium_col,
        "cci_slow": cci_slow_col,
        "vwma_fast": vwma_fast_col,
        "vwma_medium": vwma_medium_col,
        "cci_30": cci_fast_col,
        "cci_120": cci_medium_col,
        "cci_300": cci_slow_col,
        "vwma_4": vwma_fast_col,
        "vwma_12": vwma_medium_col,
    }

    def _resolve_series_name(name: object) -> str:
        k = str(name).strip()
        if not k:
            return ""
        return str(aliases.get(k, k))

    series_cols_raw = list(preset.series_cols)
    for c in sig.series_add:
        cc = str(c).strip()
        if cc and cc not in series_cols_raw:
            series_cols_raw.append(cc)

    series_cols: list[str] = []
    for c in series_cols_raw:
        cc = _resolve_series_name(c)
        if cc and cc not in series_cols:
            series_cols.append(cc)

    excludes_raw = [str(c).strip() for c in sig.series_exclude if str(c).strip()]
    excludes = {_resolve_series_name(c) for c in excludes_raw if _resolve_series_name(c)}
    if excludes:
        series_cols = [c for c in series_cols if c not in excludes]

    if not series_cols:
        raise ValueError("signals.trigger.params.series leads to empty series_cols")

    if sig.min_confirmed is None:
        required_min = len(series_cols)
    else:
        required_min = int(sig.min_confirmed)
        if required_min < 1:
            raise ValueError("min_confirmed must be >= 1")
        if required_min > len(series_cols):
            raise ValueError(f"min_confirmed={required_min} is greater than number of series ({len(series_cols)})")

    cci_fast_thr = sig.cci_fast_threshold if sig.cci_fast_threshold is not None else preset.cci_fast_threshold
    cci_medium_thr = sig.cci_medium_threshold if sig.cci_medium_threshold is not None else preset.cci_medium_threshold
    cci_slow_thr = sig.cci_slow_threshold if sig.cci_slow_threshold is not None else preset.cci_slow_threshold

    target = _target_close_extreme_kind(sig.mode)

    missing = [c for c in series_cols if c not in set(window.columns)]
    if missing:
        raise ValueError(f"Missing series columns in window: {missing}")

    if str(sig.confluence_type).lower() == "tranche_last":
        return get_current_tranche_extreme_zone_confluence_tranche_last_signal(
            window,
            ts_col=cfg.data.ts_col,
            hist_col="macd_hist",
            close_col=cfg.data.ohlc.close,
            series_cols=series_cols,
            target_close_extreme_kind=target,
            cci_fast_threshold=cci_fast_thr,
            cci_medium_threshold=cci_medium_thr,
            cci_slow_threshold=cci_slow_thr,
            cci_fast_col=cci_fast_col,
            cci_medium_col=cci_medium_col,
            cci_slow_col=cci_slow_col,
            min_confirmed=required_min,
            trend_filter=trend_filter,
        )

    return get_current_tranche_extreme_zone_confluence_signal(
        window,
        ts_col=cfg.data.ts_col,
        hist_col="macd_hist",
        close_col=cfg.data.ohlc.close,
        series_cols=series_cols,
        target_close_extreme_kind=target,
        cci_fast_threshold=cci_fast_thr,
        cci_medium_threshold=cci_medium_thr,
        cci_slow_threshold=cci_slow_thr,
        cci_fast_col=cci_fast_col,
        cci_medium_col=cci_medium_col,
        cci_slow_col=cci_slow_col,
        min_confirmed=required_min,
        trend_filter=trend_filter,
    )


def trigger_signal(window: pd.DataFrame, *, cfg: FullConfig, pivot_registry: PivotRegistry | None = None) -> TriggerDecision:
    tf = "none"
    if bool(cfg.trend_filter.enabled):
        tf = str(cfg.trend_filter.mode)

    z = _confluence_signal(window, cfg=cfg, sig=cfg.signals_trigger, trend_filter=tf)
    if not bool(z.get("is_zone")):
        return TriggerDecision(side=None, meta={"is_zone": False})

    side = z.get("open_side")
    if side not in ("LONG", "SHORT"):
        return TriggerDecision(side=None, meta={"is_zone": True, "open_side": side})

    meta = dict(z)
    ts_col = str(cfg.data.ts_col)
    tranche_start_ts = meta.get("tranche_start_ts")

    def _sign(v: object) -> int:
        try:
            x = float(v)
        except Exception:
            return 0
        if x > 0.0:
            return 1
        if x < 0.0:
            return -1
        return 0

    tranche_hist_trend_side: str | None = None
    tranche_hist_trend_is_trend = False
    tranche_hist_trend_is_healthy = False
    tranche_hist_trend_contrarian_ok = False

    if tranche_start_ts is not None and ts_col in set(window.columns):
        try:
            start_ts_int = int(tranche_start_ts)
        except Exception:
            start_ts_int = 0

        tranche = window.loc[pd.to_numeric(window[ts_col], errors="coerce") >= float(start_ts_int)]
        if len(tranche) and {"macd_line", "macd_signal", "macd_hist"}.issubset(set(tranche.columns)):
            aligned_signs: list[int] = []
            all_aligned = True
            for _i, row in tranche.iterrows():
                s1 = _sign(row.get("macd_line"))
                s2 = _sign(row.get("macd_signal"))
                s3 = _sign(row.get("macd_hist"))
                ok = (s1 != 0) and (s1 == s2 == s3)
                if ok:
                    aligned_signs.append(int(s1))
                else:
                    all_aligned = False

            tranche_hist_trend_is_trend = bool(len(aligned_signs) > 0)
            tranche_hist_trend_is_healthy = bool(tranche_hist_trend_is_trend and all_aligned)
            if tranche_hist_trend_is_trend:
                uniq = set(aligned_signs)
                if len(uniq) == 1:
                    sgn = list(uniq)[0]
                    tranche_hist_trend_side = "LONG" if int(sgn) > 0 else "SHORT"

    if tranche_hist_trend_side in ("LONG", "SHORT"):
        tranche_hist_trend_contrarian_ok = bool(_opposite_side(str(tranche_hist_trend_side)) == str(side))

    meta["tranche_hist_trend_side"] = tranche_hist_trend_side
    meta["tranche_hist_trend_is_trend"] = bool(tranche_hist_trend_is_trend)
    meta["tranche_hist_trend_is_healthy"] = bool(tranche_hist_trend_is_healthy)
    meta["tranche_hist_trend_contrarian_ok"] = bool(tranche_hist_trend_contrarian_ok)

    if bool(cfg.pivot_temporal_memory.enabled) and pivot_registry is not None:
        tranche_start_ts0 = meta.get("tranche_start_ts")
        try:
            tranche_start_ts_i = None if tranche_start_ts0 is None else int(tranche_start_ts0)
        except Exception:
            tranche_start_ts_i = None

        try:
            now_ts = int(meta.get("now_ts") or 0)
        except Exception:
            now_ts = 0
        if now_ts <= 0:
            try:
                now_ts = int(meta.get("cand_ts") or 0)
            except Exception:
                now_ts = 0

        kind_req = "LOW" if str(side).upper() == "LONG" else "HIGH"
        best_level = None
        best_dt = None
        if tranche_start_ts_i is not None:
            for ev in pivot_registry.iter_events():
                ts0 = ev.get("tranche_start_ts")
                try:
                    ts0_i = int(ts0) if ts0 is not None else None
                except Exception:
                    ts0_i = None
                if ts0_i is None or int(ts0_i) != int(tranche_start_ts_i):
                    continue
                if str(ev.get("kind") or "").strip().upper() != str(kind_req):
                    continue
                try:
                    dt_ms = int(ev.get("dt_ms") or 0)
                except Exception:
                    dt_ms = 0
                try:
                    lvl = float(ev.get("level") or 0.0)
                except Exception:
                    continue
                if best_dt is None or int(dt_ms) > int(best_dt):
                    best_dt = int(dt_ms)
                    best_level = float(lvl)

        if best_level is None:
            close_col = str(cfg.data.ohlc.close)
            if close_col in set(window.columns) and len(window) >= 2:
                try:
                    best_level = float(pd.to_numeric(window[close_col].iloc[-2], errors="coerce"))
                except Exception:
                    best_level = None

        if best_level is not None and now_ts > 0:
            mem = pivot_registry.temporal_memory_solidity(
                current_ts=int(now_ts),
                current_price=float(best_level),
                side=str(side),
                radius_pct=float(cfg.pivot_temporal_memory.radius_pct),
                exclude_tranche_start_ts=tranche_start_ts_i,
                min_fast=int(cfg.pivot_temporal_memory.min_fast),
                min_medium=int(cfg.pivot_temporal_memory.min_medium),
                min_slow=int(cfg.pivot_temporal_memory.min_slow),
                max_events=int(cfg.pivot_temporal_memory.max_events),
            )
            meta["pivot_temporal_memory"] = dict(mem)
            meta["pivot_temporal_memory"]["trigger_level"] = float(best_level)
            if not bool(mem.get("is_solid")):
                return TriggerDecision(side=None, meta=meta)

    return TriggerDecision(side=str(side), meta=meta)


def _enabled_filters(pa: PriceActionSignalConfig, *, defaults: list[str]) -> set[str]:
    out = set(str(x).strip().lower() for x in defaults if str(x).strip())
    for x in pa.filters_add:
        k = str(x).strip().lower()
        if k:
            out.add(k)
    for x in pa.filters_exclude:
        k = str(x).strip().lower()
        if k and k in out:
            out.remove(k)
    return out


def _macd_hist_delta_ok(hist: list[float], *, side: str, phase: str) -> bool:
    if len(hist) < 2:
        return False
    dh = float(hist[-1]) - float(hist[-2])
    s = str(side).upper()
    p = str(phase).lower()
    if p == "entry":
        return (dh > 0.0) if s == "LONG" else (dh < 0.0)
    if p == "exit":
        return (dh < 0.0) if s == "LONG" else (dh > 0.0)
    raise ValueError(f"Unexpected phase: {phase}")


def _macd_hist_sign_change_ok(hist: list[float], *, side: str, phase: str, mode: str) -> bool:
    if len(hist) < 1:
        return False

    s = str(side).upper()
    p = str(phase).lower()
    m = str(mode).strip().lower()
    if m == "sign":
        if p == "entry":
            return bool(float(hist[-1]) > 0.0) if s == "LONG" else bool(float(hist[-1]) < 0.0)
        if p == "exit":
            return bool(float(hist[-1]) < 0.0) if s == "LONG" else bool(float(hist[-1]) > 0.0)
        raise ValueError(f"Unexpected phase: {phase}")

    if len(hist) < 2:
        return False

    if p == "entry":
        if s == "LONG":
            return bool((float(hist[-2]) <= 0.0) and (float(hist[-1]) > 0.0))
        if s == "SHORT":
            return bool((float(hist[-2]) >= 0.0) and (float(hist[-1]) < 0.0))
        raise ValueError(f"Unexpected side: {side}")

    if p == "exit":
        if s == "LONG":
            return bool((float(hist[-2]) >= 0.0) and (float(hist[-1]) < 0.0))
        if s == "SHORT":
            return bool((float(hist[-2]) <= 0.0) and (float(hist[-1]) > 0.0))
        raise ValueError(f"Unexpected side: {side}")

    raise ValueError(f"Unexpected phase: {phase}")


def _macd_hist_accel_ok(hist: list[float], *, side: str, phase: str, mode: str) -> bool:
    if len(hist) < 3:
        return False

    s = str(side).upper()
    p = str(phase).lower()

    m = str(mode).strip().lower()
    if m == "mono":
        if p == "entry":
            return (float(hist[-1]) > float(hist[-2]) > float(hist[-3])) if s == "LONG" else (
                float(hist[-1]) < float(hist[-2]) < float(hist[-3])
            )
        if p == "exit":
            return (float(hist[-1]) < float(hist[-2]) < float(hist[-3])) if s == "LONG" else (
                float(hist[-1]) > float(hist[-2]) > float(hist[-3])
            )
        raise ValueError(f"Unexpected phase: {phase}")

    d1 = float(hist[-1]) - float(hist[-2])
    d0 = float(hist[-2]) - float(hist[-3])
    dd = d1 - d0
    if p == "entry":
        return (dd > 0.0) if s == "LONG" else (dd < 0.0)
    if p == "exit":
        return (dd < 0.0) if s == "LONG" else (dd > 0.0)
    raise ValueError(f"Unexpected phase: {phase}")


def price_action_filters_ok(
    window: pd.DataFrame,
    *,
    cfg: FullConfig,
    pa: PriceActionSignalConfig,
    side: str,
    phase: str,
) -> tuple[bool, dict[str, object]]:
    ph = str(phase).lower()
    if ph == "entry":
        em = str(pa.entry_mode).strip().lower()
        if em == "vwma_break":
            defaults = ["vwma_fast_slope", "candle_color", "macd_hist_slope", "macd_hist_sign"]
        else:
            defaults = ["vwma_fast_confirm", "stoch_cross", "macd_hist_slope", "macd_hist_sign"]
    elif ph == "exit":
        defaults = ["vwma_fast_confirm", "stoch_cross", "macd_hist_slope", "macd_hist_sign_change"]
    else:
        raise ValueError(f"Unexpected phase: {phase}")

    enabled = _enabled_filters(pa, defaults=defaults)

    close_col = str(cfg.data.ohlc.close)
    vwma_fast_col = f"vwma_{int(cfg.indicators.vwma_fast)}"
    vwma_medium_col = f"vwma_{int(cfg.indicators.vwma_medium)}"

    required_cols = {close_col, "macd_hist"}
    if "stoch_cross" in enabled:
        required_cols |= {"stoch_k", "stoch_d"}
    if "vwma_fast_confirm" in enabled:
        required_cols |= {vwma_fast_col}
    if "vwma_medium_confirm" in enabled:
        required_cols |= {vwma_medium_col}
    if "vwma_fast_slope" in enabled:
        required_cols |= {vwma_fast_col}
    if "candle_color" in enabled:
        required_cols |= {str(cfg.data.ohlc.open)}

    missing = [c for c in sorted(list(required_cols)) if c not in set(window.columns)]
    if missing:
        raise ValueError(f"Missing required columns for price action: {missing}")

    n_confirm = int(pa.vwma_confirm_bars)
    if n_confirm < 1:
        n_confirm = 1

    if len(window) < n_confirm:
        return False, {"enabled_filters": sorted(list(enabled)), "ok": False, "reason": "not_enough_bars"}

    w_confirm = window.iloc[-n_confirm:]
    last = window.iloc[-1]

    hist = pd.to_numeric(window["macd_hist"], errors="coerce").astype(float).tolist()

    s = str(side).upper()
    dir_side = s if ph == "entry" else _opposite_side(s)

    checks: dict[str, bool] = {}

    if "vwma_fast_confirm" in enabled:
        v = pd.to_numeric(w_confirm[vwma_fast_col], errors="coerce").astype(float)
        c2 = pd.to_numeric(w_confirm[close_col], errors="coerce").astype(float)
        checks["vwma_fast_confirm"] = bool((c2 > v).all()) if dir_side == "LONG" else bool((c2 < v).all())

    if "vwma_medium_confirm" in enabled:
        v = pd.to_numeric(w_confirm[vwma_medium_col], errors="coerce").astype(float)
        c2 = pd.to_numeric(w_confirm[close_col], errors="coerce").astype(float)
        checks["vwma_medium_confirm"] = bool((c2 > v).all()) if dir_side == "LONG" else bool((c2 < v).all())

    if "stoch_cross" in enabled:
        k = float(pd.to_numeric(last["stoch_k"], errors="coerce"))
        d = float(pd.to_numeric(last["stoch_d"], errors="coerce"))
        checks["stoch_cross"] = bool(k > d) if dir_side == "LONG" else bool(k < d)

    if "macd_hist_slope" in enabled:
        mode = str(pa.macd_hist_slope_mode).strip().lower()
        if mode == "accel":
            checks["macd_hist_slope"] = _macd_hist_accel_ok(
                hist,
                side=s,
                phase=ph,
                mode=str(pa.macd_hist_accel_mode),
            )
        else:
            checks["macd_hist_slope"] = _macd_hist_delta_ok(hist, side=s, phase=ph)

    if "macd_hist_sign" in enabled and ph == "entry":
        checks["macd_hist_sign"] = bool(float(hist[-1]) > 0.0) if s == "LONG" else bool(float(hist[-1]) < 0.0)

    if "macd_hist_sign_change" in enabled:
        checks["macd_hist_sign_change"] = _macd_hist_sign_change_ok(
            hist,
            side=s,
            phase=ph,
            mode=str(pa.exit_hist_sign_change_mode),
        )

    if "vwma_fast_slope" in enabled:
        v = pd.to_numeric(window[vwma_fast_col], errors="coerce").astype(float)
        if len(v) < 2:
            checks["vwma_fast_slope"] = False
        else:
            dv = float(v.iloc[-1]) - float(v.iloc[-2])
            checks["vwma_fast_slope"] = bool(dv > 0.0) if dir_side == "LONG" else bool(dv < 0.0)

    if "candle_color" in enabled:
        o = float(pd.to_numeric(last[str(cfg.data.ohlc.open)], errors="coerce"))
        c = float(pd.to_numeric(last[close_col], errors="coerce"))
        checks["candle_color"] = bool(c > o) if dir_side == "LONG" else bool(c < o)

    ok = all(bool(v) for v in checks.values())
    return ok, {
        "enabled_filters": sorted(list(enabled)),
        "checks": checks,
        "ok": bool(ok),
        "phase": ph,
        "side": s,
    }
