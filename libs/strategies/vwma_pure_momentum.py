from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from libs.backtest.signals import SignalDecision


@dataclass(frozen=True)
class VwmaPureMomentumConfig:
    ts_col: str = "ts"
    dt_col: str = "dt"

    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"

    vwma_fast_col: str = "vwma_4"
    vwma_mid_col: str = "vwma_12"
    vwma_slow_col: str = "vwma_48"

    macd_hist_col: str = "macd_hist"

    cci_fast_col: str = "cci_30"
    cci_mid_col: str = "cci_120"
    cci_slow_col: str = "cci_300"

    hist_eps: float = 0.0

    cci_extreme_level: float = 100.0
    cci_min_confluence: int = 2

    cci_filter_use_exec_bar: bool = True


def iter_vwma_pure_momentum_decisions(
    df: pd.DataFrame,
    *,
    cfg: VwmaPureMomentumConfig,
    start_pos: int = 0,
    end_pos: int | None = None,
):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")

    n = int(len(df))
    if n <= 0:
        return

    start_i = int(start_pos)
    if int(start_i) < 0:
        start_i = 0
    last_i = int(n - 1) if end_pos is None else int(end_pos)
    if int(last_i) < 0:
        return
    if int(last_i) >= int(n):
        last_i = int(n - 1)
    if int(start_i) > int(last_i):
        return

    need_cols = {
        cfg.ts_col,
        cfg.open_col,
        cfg.high_col,
        cfg.low_col,
        cfg.close_col,
        cfg.vwma_fast_col,
        cfg.vwma_mid_col,
        cfg.vwma_slow_col,
        cfg.macd_hist_col,
        cfg.cci_fast_col,
        cfg.cci_mid_col,
        cfg.cci_slow_col,
    }
    missing = [c for c in sorted(need_cols) if c not in set(df.columns)]
    if missing:
        for i in range(int(start_i), int(last_i) + 1):
            yield SignalDecision(
                side=None,
                meta={
                    "signal_i": int(i),
                    "status": "REJECT",
                    "reason": "MISSING_COLUMNS",
                    "missing": missing,
                },
            )
        return

    ts_arr = pd.to_numeric(df[cfg.ts_col], errors="coerce").astype("Int64").to_numpy()
    open_arr = pd.to_numeric(df[cfg.open_col], errors="coerce").astype(float).to_numpy()
    high_arr = pd.to_numeric(df[cfg.high_col], errors="coerce").astype(float).to_numpy()
    low_arr = pd.to_numeric(df[cfg.low_col], errors="coerce").astype(float).to_numpy()

    fast_arr = pd.to_numeric(df[cfg.vwma_fast_col], errors="coerce").astype(float).to_numpy()
    mid_arr = pd.to_numeric(df[cfg.vwma_mid_col], errors="coerce").astype(float).to_numpy()
    slow_arr = pd.to_numeric(df[cfg.vwma_slow_col], errors="coerce").astype(float).to_numpy()

    hist_arr = pd.to_numeric(df[cfg.macd_hist_col], errors="coerce").astype(float).to_numpy()

    cci_f_arr = pd.to_numeric(df[cfg.cci_fast_col], errors="coerce").astype(float).to_numpy()
    cci_m_arr = pd.to_numeric(df[cfg.cci_mid_col], errors="coerce").astype(float).to_numpy()
    cci_s_arr = pd.to_numeric(df[cfg.cci_slow_col], errors="coerce").astype(float).to_numpy()

    hist_eps = float(cfg.hist_eps)
    if not np.isfinite(float(hist_eps)) or float(hist_eps) < 0.0:
        hist_eps = 0.0

    cci_extreme_level = float(cfg.cci_extreme_level)
    if not np.isfinite(float(cci_extreme_level)) or float(cci_extreme_level) <= 0.0:
        cci_extreme_level = 100.0

    cci_min_confluence = int(cfg.cci_min_confluence)
    if cci_min_confluence < 1:
        cci_min_confluence = 1

    def _dt_at(pos: int) -> str:
        if cfg.dt_col in df.columns:
            try:
                return str(df[cfg.dt_col].iloc[int(pos)])
            except Exception:
                return ""
        try:
            ts = int(ts_arr[int(pos)] or 0)
        except Exception:
            ts = 0
        if ts <= 0:
            return ""
        return pd.to_datetime(int(ts), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC")

    def _aligned_side(pos: int) -> str | None:
        try:
            f = float(fast_arr[int(pos)])
            m = float(mid_arr[int(pos)])
            s = float(slow_arr[int(pos)])
        except Exception:
            return None
        if not (np.isfinite(float(f)) and np.isfinite(float(m)) and np.isfinite(float(s))):
            return None
        if float(f) > float(m) > float(s):
            return "LONG"
        if float(f) < float(m) < float(s):
            return "SHORT"
        return None

    def _hist_eff_sign(v: float, prev_sign: int) -> int:
        try:
            x = float(v)
        except Exception:
            return int(prev_sign)
        if not np.isfinite(float(x)):
            return int(prev_sign)
        if float(x) > float(hist_eps):
            return 1
        if float(x) < -float(hist_eps):
            return -1
        return int(prev_sign)

    def _nanmean3(a: float, b: float, c: float) -> float:
        xs = [float(a), float(b), float(c)]
        xs2 = [x for x in xs if np.isfinite(float(x))]
        if not xs2:
            return float("nan")
        return float(np.mean(xs2))

    trend_side: str | None = None
    pos_run_high: float | None = None
    neg_run_low: float | None = None
    pos_run_high_i: int | None = None
    neg_run_low_i: int | None = None

    prev_eff_hist_sign = 0
    for j in range(int(max(0, int(start_i)))):
        prev_eff_hist_sign = _hist_eff_sign(float(hist_arr[int(j)]), int(prev_eff_hist_sign))

    for i in range(int(start_i), int(last_i) + 1):
        exec_i = int(i + 1) if int(i + 1) < int(n) else None

        side_i = _aligned_side(int(i))
        if str(side_i or "") != str(trend_side or ""):
            trend_side = side_i
            pos_run_high = None
            neg_run_low = None
            pos_run_high_i = None
            neg_run_low_i = None

        prev_eff_sign_i0 = int(prev_eff_hist_sign)
        prev_hist = float(hist_arr[int(i - 1)]) if int(i - 1) >= 0 else float("nan")
        cur_hist = float(hist_arr[int(i)])
        cur_eff_sign = int(_hist_eff_sign(float(cur_hist), int(prev_eff_sign_i0)))
        cross_up = int(prev_eff_sign_i0) < 0 and int(cur_eff_sign) > 0
        cross_down = int(prev_eff_sign_i0) > 0 and int(cur_eff_sign) < 0

        if trend_side in {"LONG", "SHORT"}:
            if int(cur_eff_sign) > 0 and np.isfinite(float(high_arr[int(i)])):
                if int(prev_eff_sign_i0) <= 0:
                    pos_run_high = float(high_arr[int(i)])
                    pos_run_high_i = int(i)
                else:
                    if pos_run_high is None or float(high_arr[int(i)]) > float(pos_run_high):
                        pos_run_high = float(high_arr[int(i)])
                        pos_run_high_i = int(i)
            elif int(cur_eff_sign) < 0 and np.isfinite(float(low_arr[int(i)])):
                if int(prev_eff_sign_i0) >= 0:
                    neg_run_low = float(low_arr[int(i)])
                    neg_run_low_i = int(i)
                else:
                    if neg_run_low is None or float(low_arr[int(i)]) < float(neg_run_low):
                        neg_run_low = float(low_arr[int(i)])
                        neg_run_low_i = int(i)

        prev_eff_hist_sign = int(cur_eff_sign)

        hist_next = float("nan")
        if exec_i is not None:
            try:
                hist_next = float(hist_arr[int(exec_i)])
            except Exception:
                hist_next = float("nan")

        meta: dict[str, object] = {
            "signal_i": int(i),
            "signal_dt": _dt_at(int(i)),
            "exec_i": (None if exec_i is None else int(exec_i)),
            "exec_dt": ("" if exec_i is None else _dt_at(int(exec_i))),
            "hist_prev": (None if not np.isfinite(float(prev_hist)) else float(prev_hist)),
            "hist": (None if not np.isfinite(float(cur_hist)) else float(cur_hist)),
            "hist_next": (None if not np.isfinite(float(hist_next)) else float(hist_next)),
            "hist_prev_eff_sign": int(prev_eff_sign_i0),
            "hist_eff_sign": int(cur_eff_sign),
            "kind": ("HIST_CROSS_UP" if bool(cross_up) else ("HIST_CROSS_DOWN" if bool(cross_down) else "")),
            "vwma_fast": (None if not np.isfinite(float(fast_arr[int(i)])) else float(fast_arr[int(i)])),
            "vwma_mid": (None if not np.isfinite(float(mid_arr[int(i)])) else float(mid_arr[int(i)])),
            "vwma_slow": (None if not np.isfinite(float(slow_arr[int(i)])) else float(slow_arr[int(i)])),
            "trend_side": (None if trend_side is None else str(trend_side)),
            "hist_eps": float(hist_eps),
            "cci_extreme_level": float(cci_extreme_level),
            "cci_min_confluence": int(cci_min_confluence),
        }

        if not bool(cross_up) and not bool(cross_down):
            meta["status"] = "NO_SIGNAL"
            yield SignalDecision(side=None, meta=meta)
            continue

        if trend_side is None:
            meta["status"] = "REJECT"
            meta["reason"] = "NOT_ALIGNED"
            yield SignalDecision(side=None, meta=meta)
            continue

        if bool(cross_up) and str(trend_side) == "SHORT":
            meta["status"] = "REJECT"
            meta["reason"] = "OPPOSITE_TO_TREND"
            meta["trend"] = str(trend_side)
            yield SignalDecision(side=None, meta=meta)
            continue

        if bool(cross_down) and str(trend_side) == "LONG":
            meta["status"] = "REJECT"
            meta["reason"] = "OPPOSITE_TO_TREND"
            meta["trend"] = str(trend_side)
            yield SignalDecision(side=None, meta=meta)
            continue

        if str(trend_side) == "LONG" and bool(cross_up):
            side = "LONG"
            cci_f_sig = float(cci_f_arr[int(i)])
            cci_m_sig = float(cci_m_arr[int(i)])
            cci_s_sig = float(cci_s_arr[int(i)])
            cci_f_exec = float("nan")
            cci_m_exec = float("nan")
            cci_s_exec = float("nan")
            if exec_i is not None:
                cci_f_exec = float(cci_f_arr[int(exec_i)])
                cci_m_exec = float(cci_m_arr[int(exec_i)])
                cci_s_exec = float(cci_s_arr[int(exec_i)])

            cci_dir_mean_sig = _nanmean3(float(cci_f_sig), float(cci_m_sig), float(cci_s_sig))
            cci_dir_mean_exec = _nanmean3(float(cci_f_exec), float(cci_m_exec), float(cci_s_exec))
            cci_dir_mean = float(np.nanmax([cci_dir_mean_sig, cci_dir_mean_exec]))

            def _cci_is_extreme_long(v: float) -> bool:
                return bool(np.isfinite(float(v)) and float(v) >= float(cci_extreme_level))

            ext_n_sig = int(sum(1 for v in (cci_f_sig, cci_m_sig, cci_s_sig) if _cci_is_extreme_long(float(v))))
            ext_n_exec = int(sum(1 for v in (cci_f_exec, cci_m_exec, cci_s_exec) if _cci_is_extreme_long(float(v))))

            meta.update(
                {
                    "status": "ACCEPT",
                    "side": side,
                    "cci_dir_mean": (None if not np.isfinite(float(cci_dir_mean)) else float(cci_dir_mean)),
                    "cci_dir_mean_sig": (None if not np.isfinite(float(cci_dir_mean_sig)) else float(cci_dir_mean_sig)),
                    "cci_dir_mean_exec": (None if not np.isfinite(float(cci_dir_mean_exec)) else float(cci_dir_mean_exec)),
                    "cci_ext_n_sig": int(ext_n_sig),
                    "cci_ext_n_exec": int(ext_n_exec),
                    "cci_sig_fast": (None if not np.isfinite(float(cci_f_sig)) else float(cci_f_sig)),
                    "cci_sig_med": (None if not np.isfinite(float(cci_m_sig)) else float(cci_m_sig)),
                    "cci_sig_slow": (None if not np.isfinite(float(cci_s_sig)) else float(cci_s_sig)),
                    "cci_exec_fast": (None if not np.isfinite(float(cci_f_exec)) else float(cci_f_exec)),
                    "cci_exec_med": (None if not np.isfinite(float(cci_m_exec)) else float(cci_m_exec)),
                    "cci_exec_slow": (None if not np.isfinite(float(cci_s_exec)) else float(cci_s_exec)),
                }
            )

            if int(ext_n_sig) >= int(cci_min_confluence) or (
                bool(cfg.cci_filter_use_exec_bar) and int(ext_n_exec) >= int(cci_min_confluence)
            ):
                meta["status"] = "REJECT"
                meta["reason"] = "CCI_EXTREME"
                yield SignalDecision(side=None, meta=meta)
                continue

            if neg_run_low is None or neg_run_low_i is None or (not np.isfinite(float(neg_run_low))) or float(neg_run_low) <= 0.0:
                meta["status"] = "REJECT"
                meta["reason"] = "MISSING_OPPOSITE_RUN_EXTREME"
                yield SignalDecision(side=None, meta=meta)
                continue

            meta["extreme_i"] = int(neg_run_low_i)
            meta["extreme_dt"] = _dt_at(int(neg_run_low_i))
            meta["extreme"] = float(neg_run_low)
            if exec_i is not None:
                entry = float(open_arr[int(exec_i)])
                meta["entry"] = (None if not np.isfinite(float(entry)) else float(entry))

            yield SignalDecision(side=side, meta=meta)
            continue

        if str(trend_side) == "SHORT" and bool(cross_down):
            side = "SHORT"
            cci_f_sig = float(cci_f_arr[int(i)])
            cci_m_sig = float(cci_m_arr[int(i)])
            cci_s_sig = float(cci_s_arr[int(i)])
            cci_f_exec = float("nan")
            cci_m_exec = float("nan")
            cci_s_exec = float("nan")
            if exec_i is not None:
                cci_f_exec = float(cci_f_arr[int(exec_i)])
                cci_m_exec = float(cci_m_arr[int(exec_i)])
                cci_s_exec = float(cci_s_arr[int(exec_i)])

            cci_dir_mean_sig = _nanmean3(-float(cci_f_sig), -float(cci_m_sig), -float(cci_s_sig))
            cci_dir_mean_exec = _nanmean3(-float(cci_f_exec), -float(cci_m_exec), -float(cci_s_exec))
            cci_dir_mean = float(np.nanmax([cci_dir_mean_sig, cci_dir_mean_exec]))

            def _cci_is_extreme_short(v: float) -> bool:
                return bool(np.isfinite(float(v)) and float(v) <= -float(cci_extreme_level))

            ext_n_sig = int(sum(1 for v in (cci_f_sig, cci_m_sig, cci_s_sig) if _cci_is_extreme_short(float(v))))
            ext_n_exec = int(sum(1 for v in (cci_f_exec, cci_m_exec, cci_s_exec) if _cci_is_extreme_short(float(v))))

            meta.update(
                {
                    "status": "ACCEPT",
                    "side": side,
                    "cci_dir_mean": (None if not np.isfinite(float(cci_dir_mean)) else float(cci_dir_mean)),
                    "cci_dir_mean_sig": (None if not np.isfinite(float(cci_dir_mean_sig)) else float(cci_dir_mean_sig)),
                    "cci_dir_mean_exec": (None if not np.isfinite(float(cci_dir_mean_exec)) else float(cci_dir_mean_exec)),
                    "cci_ext_n_sig": int(ext_n_sig),
                    "cci_ext_n_exec": int(ext_n_exec),
                    "cci_sig_fast": (None if not np.isfinite(float(cci_f_sig)) else float(cci_f_sig)),
                    "cci_sig_med": (None if not np.isfinite(float(cci_m_sig)) else float(cci_m_sig)),
                    "cci_sig_slow": (None if not np.isfinite(float(cci_s_sig)) else float(cci_s_sig)),
                    "cci_exec_fast": (None if not np.isfinite(float(cci_f_exec)) else float(cci_f_exec)),
                    "cci_exec_med": (None if not np.isfinite(float(cci_m_exec)) else float(cci_m_exec)),
                    "cci_exec_slow": (None if not np.isfinite(float(cci_s_exec)) else float(cci_s_exec)),
                }
            )

            if int(ext_n_sig) >= int(cci_min_confluence) or (
                bool(cfg.cci_filter_use_exec_bar) and int(ext_n_exec) >= int(cci_min_confluence)
            ):
                meta["status"] = "REJECT"
                meta["reason"] = "CCI_EXTREME"
                yield SignalDecision(side=None, meta=meta)
                continue

            if pos_run_high is None or pos_run_high_i is None or (not np.isfinite(float(pos_run_high))) or float(pos_run_high) <= 0.0:
                meta["status"] = "REJECT"
                meta["reason"] = "MISSING_OPPOSITE_RUN_EXTREME"
                yield SignalDecision(side=None, meta=meta)
                continue

            meta["extreme_i"] = int(pos_run_high_i)
            meta["extreme_dt"] = _dt_at(int(pos_run_high_i))
            meta["extreme"] = float(pos_run_high)
            if exec_i is not None:
                entry = float(open_arr[int(exec_i)])
                meta["entry"] = (None if not np.isfinite(float(entry)) else float(entry))

            yield SignalDecision(side=side, meta=meta)
            continue

        meta["status"] = "NO_SIGNAL"
        yield SignalDecision(side=None, meta=meta)


def last_closed_vwma_pure_momentum_signal(
    window: pd.DataFrame,
    *,
    cfg: VwmaPureMomentumConfig,
    last_closed_pos: int | None = None,
    start_pos: int = 0,
) -> SignalDecision:
    if not isinstance(window, pd.DataFrame):
        raise TypeError("window must be a pandas.DataFrame")

    n = int(len(window))
    if n <= 0:
        return SignalDecision(side=None, meta={"status": "REJECT", "reason": "EMPTY_WINDOW"})

    i = int(last_closed_pos) if last_closed_pos is not None else (n - 2 if n >= 2 else n - 1)
    if int(i) < 0:
        i = int(n - 1)
    if int(i) >= int(n):
        i = int(n - 1)

    dec: SignalDecision | None = None
    for dec in iter_vwma_pure_momentum_decisions(window, cfg=cfg, start_pos=int(start_pos), end_pos=int(i)):
        pass
    if dec is None:
        return SignalDecision(side=None, meta={"status": "REJECT", "reason": "NO_DECISION"})
    return dec
