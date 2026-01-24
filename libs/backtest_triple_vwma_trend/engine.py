from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from libs.agents.triple_vwma_cycle_agent import TripleVwmaCycleAgent, TripleVwmaCycleAgentConfig
from libs.backtest_triple_vwma_trend.config import FullConfig
from libs.indicators.momentum.macd_tv import macd_tv
from libs.indicators.moving_averages.vwma_tv import vwma_tv


@dataclass
class PositionState:
    side: str
    entry_i: int
    entry_ts: int
    entry_price: float
    sl_price: float
    tp_price: float | None


@dataclass
class TradeRecord:
    side: str
    entry_ts: int
    exit_ts: int
    entry_price: float
    exit_price: float
    exit_reason: str
    gross_ret: float
    net_ret: float


def _gross_ret(*, side: str, entry: float, exit_: float) -> float:
    if entry == 0.0 or (not np.isfinite(float(entry))) or (not np.isfinite(float(exit_))):
        return float("nan")
    if side == "LONG":
        return (float(exit_) / float(entry)) - 1.0
    if side == "SHORT":
        return (float(entry) / float(exit_)) - 1.0
    raise ValueError(f"Unexpected side: {side}")


def _intrabar_exit(*, side: str, high: float, low: float, tp_price: float | None, sl_price: float | None) -> tuple[float, str] | None:
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


def _ensure_vwma_cols(df: pd.DataFrame, *, cfg: FullConfig) -> pd.DataFrame:
    df2 = df.copy()

    ts_col = str(cfg.data.ts_col)
    o = str(cfg.data.ohlc.open)
    h = str(cfg.data.ohlc.high)
    l = str(cfg.data.ohlc.low)
    c = str(cfg.data.ohlc.close)
    v = str(cfg.data.ohlc.volume)

    for col in (ts_col, o, h, l, c):
        if col not in df2.columns:
            raise ValueError(f"Missing required column: {col}")

    df2[ts_col] = pd.to_numeric(df2[ts_col], errors="coerce").astype("Int64")
    df2 = df2.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    for col in (o, h, l, c):
        df2[col] = pd.to_numeric(df2[col], errors="coerce")

    if v not in df2.columns:
        df2[v] = 1.0
    df2[v] = pd.to_numeric(df2[v], errors="coerce")
    df2 = df2.dropna(subset=[o, h, l, c, v]).reset_index(drop=True)

    close = pd.to_numeric(df2[c], errors="coerce").astype(float).tolist()
    vol = pd.to_numeric(df2[v], errors="coerce").astype(float).tolist()

    fast_len = int(cfg.vwma.fast)
    mid_len = int(cfg.vwma.mid)
    slow_len = int(cfg.vwma.slow)

    vwma_fast_col = f"vwma_{fast_len}"
    vwma_mid_col = f"vwma_{mid_len}"
    vwma_slow_col = f"vwma_{slow_len}"

    if vwma_fast_col not in df2.columns:
        df2[vwma_fast_col] = vwma_tv(close, vol, int(fast_len))
    if vwma_mid_col not in df2.columns:
        df2[vwma_mid_col] = vwma_tv(close, vol, int(mid_len))
    if vwma_slow_col not in df2.columns:
        df2[vwma_slow_col] = vwma_tv(close, vol, int(slow_len))

    return df2


def run_backtest_from_config(
    *,
    cfg: FullConfig,
    df: pd.DataFrame | None = None,
    trace: bool = False,
    trace_every: int = 500,
) -> dict[str, object]:
    if df is None:
        df = pd.read_csv(Path(cfg.data.csv))

    t_start = perf_counter()
    if bool(trace):
        print(f"[triple_vwma_bt] loaded df rows={len(df)}", flush=True)

    df = _ensure_vwma_cols(df, cfg=cfg)

    if bool(trace):
        print(f"[triple_vwma_bt] ensured VWMA cols; rows={len(df)}", flush=True)

    ts_col = str(cfg.data.ts_col)
    o_col = str(cfg.data.ohlc.open)
    h_col = str(cfg.data.ohlc.high)
    l_col = str(cfg.data.ohlc.low)
    c_col = str(cfg.data.ohlc.close)

    fast_len = int(cfg.vwma.fast)
    mid_len = int(cfg.vwma.mid)
    slow_len = int(cfg.vwma.slow)

    vwma_fast_col = f"vwma_{fast_len}"
    vwma_mid_col = f"vwma_{mid_len}"
    vwma_slow_col = f"vwma_{slow_len}"

    n = int(len(df))
    if n < 5:
        raise ValueError(f"Not enough rows: {n}")

    ts = pd.to_numeric(df[ts_col], errors="coerce").astype("Int64").to_numpy()
    o = pd.to_numeric(df[o_col], errors="coerce").astype(float).to_numpy()
    h = pd.to_numeric(df[h_col], errors="coerce").astype(float).to_numpy()
    l = pd.to_numeric(df[l_col], errors="coerce").astype(float).to_numpy()
    c = pd.to_numeric(df[c_col], errors="coerce").astype(float).to_numpy()

    fast = pd.to_numeric(df[vwma_fast_col], errors="coerce").astype(float).to_numpy()
    mid = pd.to_numeric(df[vwma_mid_col], errors="coerce").astype(float).to_numpy()
    slow = pd.to_numeric(df[vwma_slow_col], errors="coerce").astype(float).to_numpy()

    zf = float(cfg.zones.fast_radius_pct)
    zm = float(cfg.zones.mid_radius_pct)
    zs = float(cfg.zones.slow_radius_pct)
    if zf < 0:
        zf = 0.0
    if zm < 0:
        zm = 0.0
    if zs < 0:
        zs = 0.0

    def _bounds(x: np.ndarray, pct: float) -> tuple[np.ndarray, np.ndarray]:
        return x * (1.0 + float(pct)), x * (1.0 - float(pct))

    mu, ml = _bounds(mid, zm)
    su, sl = _bounds(slow, zs)

    finite_ms = np.isfinite(mu) & np.isfinite(ml) & np.isfinite(su) & np.isfinite(sl)
    sep_ms = np.zeros(int(n), dtype=int)
    sep_ms = np.where(finite_ms & (ml > su), 1, sep_ms)
    sep_ms = np.where(finite_ms & (mu < sl), -1, sep_ms)

    def _macro_side(i: int) -> str | None:
        s = int(sep_ms[i])
        if s == 1:
            return "LONG"
        if s == -1:
            return "SHORT"
        return None

    fs_diff = fast - slow
    fs_sign = np.sign(fs_diff)

    # MACD (TradingView default params). Used by scalp mode.
    _, _, macd_hist_list = macd_tv(pd.to_numeric(df[c_col], errors="coerce").astype(float).tolist(), 12, 26, 9)
    macd_hist = np.asarray(macd_hist_list, dtype=float)

    mode = str(cfg.strategy.mode).strip().lower()
    no_lookahead = bool(cfg.backtest.no_lookahead)

    # Pullback agent is only needed for swing mode.
    agent: TripleVwmaCycleAgent | None = None
    if mode == "swing":
        pull_cfg = TripleVwmaCycleAgentConfig(
            ts_col=ts_col,
            open_col=o_col,
            high_col=h_col,
            low_col=l_col,
            close_col=str(cfg.data.ohlc.close),
            vwma_fast_col=vwma_fast_col,
            vwma_mid_col=vwma_mid_col,
            vwma_slow_col=vwma_slow_col,
            zone_fast_radius_pct=float(cfg.zones.fast_radius_pct),
            zone_mid_radius_pct=float(cfg.zones.mid_radius_pct),
            zone_slow_radius_pct=float(cfg.zones.slow_radius_pct),
            zone_large_mult=float(cfg.zones.zone_large_mult),
            break_confirm_bars=int(cfg.zones.break_confirm_bars),
            spread_ref_pct=0.002,
            min_cycle_len=1,
            min_score=0.0,
        )
        agent = TripleVwmaCycleAgent(cfg=pull_cfg)

    def _fmt_dt(ts_ms: int) -> str:
        try:
            return str(pd.to_datetime(int(ts_ms), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC"))
        except Exception:
            return ""

    if bool(trace):
        warmup = max(int(cfg.vwma.fast), int(cfg.vwma.mid), int(cfg.vwma.slow))
        ts0 = int(ts[0]) if int(n) > 0 else 0
        ts1 = int(ts[int(n - 1)]) if int(n) > 0 else 0
        print(
            "[triple_vwma_bt] start"
            f" mode={mode}"
            f" no_lookahead={no_lookahead}"
            f" rows={n}"
            f" warmup={warmup}"
            f" range='{_fmt_dt(ts0)}'..'{_fmt_dt(ts1)}'",
            flush=True,
        )
        if bool(no_lookahead) and int(n) > 2000:
            print(
                "[triple_vwma_bt] WARNING: no_lookahead=true recalculates events each bar (can be very slow for large datasets).",
                flush=True,
            )

    pull_events: dict[int, list[dict[str, object]]] = {}
    cycles = []
    if mode == "swing" and (not bool(no_lookahead)):
        if agent is None:
            raise RuntimeError("agent must not be None when mode='swing'")
        cycles = agent.analyze_df(df, max_cycles=0)

    def _to_global_event(e: dict[str, object], *, cycle_start_i: int) -> dict[str, object] | None:
        try:
            local_pos = int(e.get("pos"))
        except Exception:
            return None
        out = dict(e)
        out["pos"] = int(cycle_start_i) + int(local_pos)
        meta = out.get("meta")
        if isinstance(meta, dict):
            meta2 = dict(meta)
            for k in ("start_pos", "end_pos", "extreme_pos"):
                if k in meta2 and meta2[k] is not None:
                    try:
                        meta2[k] = int(cycle_start_i) + int(meta2[k])
                    except Exception:
                        pass
            out["meta"] = meta2
        return out

    for cyc in cycles:
        for e in cyc.events:
            if not isinstance(e, dict):
                continue
            eg = _to_global_event(e, cycle_start_i=int(cyc.start_i))
            if eg is None:
                continue
            pull_events.setdefault(int(eg["pos"]), []).append(eg)

    _events_cache_i: int | None = None
    _events_cache: list[dict[str, object]] = []
    _events_time_s = 0.0
    _events_calls = 0

    def _events_at(pos_i: int) -> list[dict[str, object]]:
        nonlocal _events_cache_i, _events_cache
        nonlocal _events_time_s, _events_calls

        if mode != "swing":
            return []

        if not bool(no_lookahead):
            return list(pull_events.get(int(pos_i), []))

        if _events_cache_i is not None and int(_events_cache_i) == int(pos_i):
            return list(_events_cache)

        if agent is None:
            return []

        t0 = perf_counter()
        dfw = df.iloc[: int(pos_i) + 1]
        cyc_now = agent.analyze_df(dfw, max_cycles=0)
        dt = perf_counter() - t0
        _events_time_s += float(dt)
        _events_calls += 1
        out: list[dict[str, object]] = []
        for cc in cyc_now:
            for ee in cc.events:
                if not isinstance(ee, dict):
                    continue
                eg = _to_global_event(ee, cycle_start_i=int(cc.start_i))
                if eg is None:
                    continue
                if int(eg.get("pos") or -1) == int(pos_i):
                    out.append(eg)

        _events_cache_i = int(pos_i)
        _events_cache = list(out)
        return list(out)

    fee_rate = float(cfg.backtest.fee_rate)
    max_sl_pct = cfg.strategy.max_sl_pct

    def _aligned_side(i: int) -> str | None:
        if not (np.isfinite(float(fast[i])) and np.isfinite(float(mid[i])) and np.isfinite(float(slow[i]))):
            return None
        if float(fast[i]) > float(mid[i]) > float(slow[i]):
            return "LONG"
        if float(fast[i]) < float(mid[i]) < float(slow[i]):
            return "SHORT"
        return None

    equity = 0.0
    equity_points: list[dict[str, object]] = [{"ts": int(ts[0]), "equity": float(equity), "event": "START"}]
    trades: list[TradeRecord] = []

    pos: PositionState | None = None
    pending_entry: dict[str, object] | None = None
    pending_exit: dict[str, object] | None = None

    swing_regime_side: str | None = None
    swing_cross_level: float | None = None
    swing_regime_active = False
    swing_first_pullback_taken = False

    warmup = max(int(cfg.vwma.fast), int(cfg.vwma.mid), int(cfg.vwma.slow))

    scalp_trend_side: str | None = None
    scalp_pos_run_high: float | None = None
    scalp_neg_run_low: float | None = None

    def _trace_progress(i: int) -> None:
        if not bool(trace):
            return
        if int(trace_every) <= 0:
            return
        if int(i) == int(warmup) or (int(i) % int(trace_every) == 0):
            elapsed = perf_counter() - t_start
            done = max(1, int(i) - int(warmup))
            total = max(1, int(n) - int(warmup))
            pct = 100.0 * float(done) / float(total)
            per_bar = float(elapsed) / float(done)
            eta = float(per_bar) * float(max(0, int(total) - int(done)))
            pos_side = None if pos is None else str(pos.side)
            ev_avg_ms = (1000.0 * float(_events_time_s) / float(max(1, int(_events_calls))))

            tsi = int(ts[int(i)]) if 0 <= int(i) < int(len(ts)) else 0
            dti = _fmt_dt(int(tsi))
            print(
                "[triple_vwma_bt]"
                f" i={i}/{n}"
                f" {pct:.2f}%"
                f" ts={tsi}"
                f" dt='{dti}'"
                f" elapsed={elapsed:.1f}s"
                f" eta={eta:.1f}s"
                f" trades={len(trades)}"
                f" pos={pos_side}"
                f" ev_avg_ms={ev_avg_ms:.2f}"
                f" ev_calls={_events_calls}",
                flush=True,
            )

    for i in range(int(warmup), int(n)):
        if i + 1 >= int(n):
            break

        _trace_progress(int(i))

        ts_i = int(ts[i])
        o_i = float(o[i])
        h_i = float(h[i])
        l_i = float(l[i])

        if pos is not None and pending_exit is not None and int(pending_exit["exec_i"]) == int(i):
            exit_price = float(o_i)
            gross = _gross_ret(side=pos.side, entry=pos.entry_price, exit_=exit_price)
            net = float(gross) - (2.0 * float(fee_rate))
            trades.append(
                TradeRecord(
                    side=str(pos.side),
                    entry_ts=int(pos.entry_ts),
                    exit_ts=int(ts_i),
                    entry_price=float(pos.entry_price),
                    exit_price=float(exit_price),
                    exit_reason=str(pending_exit.get("reason") or "EXIT"),
                    gross_ret=float(gross),
                    net_ret=float(net),
                )
            )
            equity += float(net)
            equity_points.append({"ts": int(ts_i), "equity": float(equity), "event": str(pending_exit.get("reason") or "EXIT")})
            if bool(trace):
                dt_i = _fmt_dt(int(ts_i))
                print(
                    "[triple_vwma_bt] EXIT"
                    f" i={i}"
                    f" ts={int(ts_i)}"
                    f" dt='{dt_i}'"
                    f" side={pos.side}"
                    f" entry={pos.entry_price:.6f}"
                    f" exit={exit_price:.6f}"
                    f" reason={str(pending_exit.get('reason') or 'EXIT')}"
                    f" gross_ret_pct={100.0 * float(gross):.3f}"
                    f" net_ret_pct={100.0 * float(net):.3f}",
                    flush=True,
                )
            pos = None
            pending_exit = None

        if pos is None and pending_entry is not None and int(pending_entry["exec_i"]) == int(i):
            side = str(pending_entry["side"]).upper()
            entry_price = float(o_i)
            sl_price = float(pending_entry["sl_price"])

            if max_sl_pct is not None:
                extreme_price = pending_entry.get("extreme_price")
                try:
                    exf = float(extreme_price) if extreme_price is not None else None
                except Exception:
                    exf = None
                if exf is not None and np.isfinite(float(exf)) and float(entry_price) > 0.0:
                    dist = abs(float(entry_price) - float(exf)) / float(entry_price)
                    if float(dist) > float(max_sl_pct):
                        if bool(pending_entry.get("swing_mark_first_pullback") or False):
                            swing_first_pullback_taken = False
                        if bool(trace):
                            dt_i = _fmt_dt(int(ts_i))
                            print(
                                "[triple_vwma_bt] SKIP_ENTRY"
                                f" i={i}"
                                f" ts={int(ts_i)}"
                                f" dt='{dt_i}'"
                                f" side={side}"
                                f" entry={entry_price:.6f}"
                                f" extreme={float(exf):.6f}"
                                f" dist_ext_pct={100.0 * float(dist):.3f}"
                                f" max_sl_pct={100.0 * float(max_sl_pct):.3f}",
                                flush=True,
                            )
                        pending_entry = None
                        continue

            tp_price = None
            if pending_entry.get("tp_pct") is not None:
                tp_pct = float(pending_entry["tp_pct"])
                if side == "LONG":
                    tp_price = float(entry_price) * (1.0 + float(tp_pct))
                else:
                    tp_price = float(entry_price) * (1.0 - float(tp_pct))
            elif pending_entry.get("tp_price") is not None:
                tp_price = float(pending_entry["tp_price"])

            mark_first = bool(pending_entry.get("swing_mark_first_pullback") or False)
            pos = PositionState(
                side=str(side),
                entry_i=int(i),
                entry_ts=int(ts_i),
                entry_price=float(entry_price),
                sl_price=float(sl_price),
                tp_price=(None if tp_price is None else float(tp_price)),
            )
            if bool(mark_first):
                swing_first_pullback_taken = True
            if bool(trace):
                dt_i = _fmt_dt(int(ts_i))
                print(
                    "[triple_vwma_bt] ENTRY"
                    f" i={i}"
                    f" ts={int(ts_i)}"
                    f" dt='{dt_i}'"
                    f" side={pos.side}"
                    f" entry={pos.entry_price:.6f}"
                    f" sl={pos.sl_price:.6f}"
                    f" tp={(None if pos.tp_price is None else round(float(pos.tp_price), 6))}",
                    flush=True,
                )
            pending_entry = None

        if pos is not None:
            ex = _intrabar_exit(side=str(pos.side), high=float(h_i), low=float(l_i), tp_price=pos.tp_price, sl_price=pos.sl_price)
            if ex is not None:
                exit_price, reason = ex
                gross = _gross_ret(side=pos.side, entry=pos.entry_price, exit_=float(exit_price))
                net = float(gross) - (2.0 * float(fee_rate))
                trades.append(
                    TradeRecord(
                        side=str(pos.side),
                        entry_ts=int(pos.entry_ts),
                        exit_ts=int(ts_i),
                        entry_price=float(pos.entry_price),
                        exit_price=float(exit_price),
                        exit_reason=str(reason),
                        gross_ret=float(gross),
                        net_ret=float(net),
                    )
                )
                equity += float(net)
                equity_points.append({"ts": int(ts_i), "equity": float(equity), "event": str(reason)})
                if bool(trace):
                    dt_i = _fmt_dt(int(ts_i))
                    print(
                        "[triple_vwma_bt] EXIT"
                        f" i={i}"
                        f" ts={int(ts_i)}"
                        f" dt='{dt_i}'"
                        f" side={pos.side}"
                        f" entry={pos.entry_price:.6f}"
                        f" exit={float(exit_price):.6f}"
                        f" reason={str(reason)}"
                        f" gross_ret_pct={100.0 * float(gross):.3f}"
                        f" net_ret_pct={100.0 * float(net):.3f}",
                        flush=True,
                    )
                pos = None
                pending_exit = None
                continue

        cross_up = False
        cross_down = False
        if i - 1 >= 0 and np.isfinite(float(fs_sign[i - 1])) and np.isfinite(float(fs_sign[i])):
            cross_up = bool(float(fs_sign[i - 1]) <= 0.0 and float(fs_sign[i]) > 0.0)
            cross_down = bool(float(fs_sign[i - 1]) >= 0.0 and float(fs_sign[i]) < 0.0)

        if mode == "swing":
            if cross_up:
                swing_regime_side = "LONG"
                swing_cross_level = float((float(fast[i]) + float(slow[i])) / 2.0) if np.isfinite(float(fast[i])) and np.isfinite(float(slow[i])) else None
                swing_regime_active = False
                swing_first_pullback_taken = False
            elif cross_down:
                swing_regime_side = "SHORT"
                swing_cross_level = float((float(fast[i]) + float(slow[i])) / 2.0) if np.isfinite(float(fast[i])) and np.isfinite(float(slow[i])) else None
                swing_regime_active = False
                swing_first_pullback_taken = False

            if swing_regime_side == "LONG":
                aligned = bool(np.isfinite(float(fast[i])) and np.isfinite(float(mid[i])) and np.isfinite(float(slow[i])) and float(fast[i]) > float(mid[i]) > float(slow[i]))
                clear = bool(int(sep_ms[i]) == 1)
                if (not swing_regime_active) and bool(aligned) and bool(clear):
                    swing_regime_active = True
            elif swing_regime_side == "SHORT":
                aligned = bool(np.isfinite(float(fast[i])) and np.isfinite(float(mid[i])) and np.isfinite(float(slow[i])) and float(fast[i]) < float(mid[i]) < float(slow[i]))
                clear = bool(int(sep_ms[i]) == -1)
                if (not swing_regime_active) and bool(aligned) and bool(clear):
                    swing_regime_active = True

            if pos is not None and pending_exit is None:
                if (pos.side == "LONG" and cross_down) or (pos.side == "SHORT" and cross_up):
                    pending_exit = {"exec_i": int(i + 1), "reason": "CROSS_INVERSE"}

            if pos is not None:
                cur_sep = int(sep_ms[i])
                same_macro = bool((pos.side == "LONG" and cur_sep == 1) or (pos.side == "SHORT" and cur_sep == -1))
                evs = _events_at(int(i))
                for e in evs:
                    kind = str(e.get("kind") or "")
                    if kind != "pullback_strong":
                        continue
                    meta = e.get("meta") or {}
                    extreme_price = meta.get("extreme_price")
                    try:
                        exf = float(extreme_price) if extreme_price is not None else None
                    except Exception:
                        exf = None
                    if exf is None or (not np.isfinite(float(exf))) or float(exf) <= 0:
                        continue

                    if not bool(same_macro):
                        continue

                    buf = float(cfg.strategy.swing.sl_buffer_pct)
                    if str(pos.side) == "LONG":
                        new_sl = float(exf) * (1.0 - float(buf))
                        if float(new_sl) > float(pos.sl_price):
                            pos.sl_price = float(new_sl)
                    else:
                        new_sl = float(exf) * (1.0 + float(buf))
                        if float(new_sl) < float(pos.sl_price):
                            pos.sl_price = float(new_sl)

            if pos is None and pending_entry is None and bool(swing_regime_active) and (not bool(swing_first_pullback_taken)):
                evs = _events_at(int(i))
                for e in evs:
                    kind = str(e.get("kind") or "")
                    if kind not in {"pullback_weak", "pullback_medium", "pullback_strong"}:
                        continue
                    cur_sep = int(sep_ms[i])
                    if not bool((swing_regime_side == "LONG" and cur_sep == 1) or (swing_regime_side == "SHORT" and cur_sep == -1)):
                        continue
                    if swing_cross_level is None or (not np.isfinite(float(swing_cross_level))):
                        break
                    buf = float(cfg.strategy.swing.sl_buffer_pct)
                    if str(swing_regime_side) == "LONG":
                        sl0 = float(swing_cross_level) * (1.0 - float(buf))
                    else:
                        sl0 = float(swing_cross_level) * (1.0 + float(buf))

                    pending_entry = {"exec_i": int(i + 1), "side": str(swing_regime_side), "sl_price": float(sl0), "tp_price": None}
                    try:
                        extreme_price = float((e.get("meta") or {}).get("extreme_price"))
                    except Exception:
                        extreme_price = None
                    if extreme_price is not None:
                        pending_entry["extreme_price"] = float(extreme_price)
                    pending_entry["swing_mark_first_pullback"] = True
                    break

        if mode == "scalp":
            if pos is None and pending_entry is None:
                # Trend filter: strict alignment.
                side = _aligned_side(int(i))
                if str(side or "") != str(scalp_trend_side or ""):
                    scalp_trend_side = side
                    scalp_pos_run_high = None
                    scalp_neg_run_low = None

                if side in {"LONG", "SHORT"} and i - 1 >= 0:
                    prev_hist = macd_hist[int(i - 1)]
                    cur_hist = macd_hist[int(i)]
                    if np.isfinite(float(prev_hist)) and np.isfinite(float(cur_hist)):
                        # LONG: hist crosses up (<=0 -> >0), SL on last negative-run low.
                        if side == "LONG" and float(prev_hist) <= 0.0 and float(cur_hist) > 0.0:
                            if scalp_neg_run_low is not None and np.isfinite(float(scalp_neg_run_low)) and float(scalp_neg_run_low) > 0.0:
                                exf = float(scalp_neg_run_low)
                                buf = float(cfg.strategy.scalp.sl_buffer_pct)
                                sl0 = float(exf) * (1.0 - float(buf))
                                tp_pct = float(cfg.strategy.scalp.tp_pct)
                                pending_entry = {"exec_i": int(i + 1), "side": "LONG", "sl_price": float(sl0), "tp_pct": float(tp_pct), "extreme_price": float(exf)}

                        # SHORT: hist crosses down (>=0 -> <0), SL on last positive-run high.
                        if side == "SHORT" and float(prev_hist) >= 0.0 and float(cur_hist) < 0.0:
                            if scalp_pos_run_high is not None and np.isfinite(float(scalp_pos_run_high)) and float(scalp_pos_run_high) > 0.0:
                                exf = float(scalp_pos_run_high)
                                buf = float(cfg.strategy.scalp.sl_buffer_pct)
                                sl0 = float(exf) * (1.0 + float(buf))
                                tp_pct = float(cfg.strategy.scalp.tp_pct)
                                pending_entry = {"exec_i": int(i + 1), "side": "SHORT", "sl_price": float(sl0), "tp_pct": float(tp_pct), "extreme_price": float(exf)}

            # Update most recent opposite-momentum extremes (used for next sign change).
            prev_hist = macd_hist[int(i - 1)] if i - 1 >= 0 else float("nan")
            cur_hist = macd_hist[int(i)]
            if scalp_trend_side in {"LONG", "SHORT"} and np.isfinite(float(cur_hist)):
                if float(cur_hist) > 0.0:
                    if (not np.isfinite(float(prev_hist))) or float(prev_hist) <= 0.0:
                        scalp_pos_run_high = float(h_i)
                    else:
                        scalp_pos_run_high = float(h_i) if scalp_pos_run_high is None else float(max(float(scalp_pos_run_high), float(h_i)))
                elif float(cur_hist) < 0.0:
                    if (not np.isfinite(float(prev_hist))) or float(prev_hist) >= 0.0:
                        scalp_neg_run_low = float(l_i)
                    else:
                        scalp_neg_run_low = float(l_i) if scalp_neg_run_low is None else float(min(float(scalp_neg_run_low), float(l_i)))

        equity_points.append({"ts": int(ts_i), "equity": float(equity), "event": ""})

    if pos is not None:
        exit_price = float(o[int(n - 1)])
        gross = _gross_ret(side=pos.side, entry=pos.entry_price, exit_=exit_price)
        net = float(gross) - (2.0 * float(fee_rate))
        trades.append(
            TradeRecord(
                side=str(pos.side),
                entry_ts=int(pos.entry_ts),
                exit_ts=int(ts[int(n - 1)]),
                entry_price=float(pos.entry_price),
                exit_price=float(exit_price),
                exit_reason="EOD",
                gross_ret=float(gross),
                net_ret=float(net),
            )
        )
        equity += float(net)
        equity_points.append({"ts": int(ts[int(n - 1)]), "equity": float(equity), "event": "EOD"})

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    equity_df = pd.DataFrame(equity_points)
    equity_df["peak"] = equity_df["equity"].cummax()
    equity_df["dd"] = equity_df["equity"] - equity_df["peak"]

    max_dd = float(equity_df["dd"].min()) if len(equity_df) else 0.0
    equity_end = float(equity_df["equity"].iloc[-1]) if len(equity_df) else 0.0
    ratio = 0.0 if equity_end <= 0.0 else (float("inf") if max_dd == 0.0 else float(equity_end) / abs(float(max_dd)))

    wins_mask = pd.to_numeric(trades_df.get("net_ret"), errors="coerce") > 0.0 if len(trades_df) else pd.Series([], dtype=bool)
    losses_mask = pd.to_numeric(trades_df.get("net_ret"), errors="coerce") < 0.0 if len(trades_df) else pd.Series([], dtype=bool)
    n_wins = int(wins_mask.sum()) if len(trades_df) else 0
    n_losses = int(losses_mask.sum()) if len(trades_df) else 0
    winrate = float(n_wins) / float(len(trades_df)) if len(trades_df) else 0.0

    if bool(trace):
        elapsed = perf_counter() - t_start
        ev_pct = 0.0 if elapsed <= 0 else (100.0 * float(_events_time_s) / float(elapsed))
        print(
            "[triple_vwma_bt] done"
            f" elapsed={elapsed:.1f}s"
            f" trades={len(trades_df)}"
            f" events_time={_events_time_s:.1f}s"
            f" events_time_pct={ev_pct:.1f}%"
            f" events_calls={_events_calls}",
            flush=True,
        )

    return {
        "df": df,
        "trades": trades_df,
        "equity": equity_df,
        "summary": {
            "n_trades": int(len(trades_df)),
            "n_wins": int(n_wins),
            "n_losses": int(n_losses),
            "winrate": float(winrate),
            "equity_end": float(equity_end),
            "max_dd": float(max_dd),
            "ratio": float(ratio),
        },
    }
