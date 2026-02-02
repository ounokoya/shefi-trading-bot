from __future__ import annotations

import math
from typing import Dict, List, Optional

import pandas as pd


def hist_cross_up(prev_h: float, h: float) -> bool:
    if not math.isfinite(float(h)) or not math.isfinite(float(prev_h)):
        return False
    return float(prev_h) <= 0.0 and float(h) > 0.0


def hist_cross_down(prev_h: float, h: float) -> bool:
    if not math.isfinite(float(h)) or not math.isfinite(float(prev_h)):
        return False
    return float(prev_h) >= 0.0 and float(h) < 0.0


def safe_float(v: object) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def simulate_trades_from_stream(
    df: pd.DataFrame,
    *,
    series_to_col: Dict[str, str],
    mode: str,
    signal_from: str,
    selected_series: List[str],
    cci_col: str,
    cci_low: float,
    cci_high: float,
    macd_hist_col: str,
    trade_direction: str,
    min_confluence: int,
    use_fixed_stop: bool,
    stop_buffer_pct: float,
    stop_ref_series: str,
    start_i: int,
    extreme_confirm_bars: int,
    entry_require_hist_abs_growth: bool,
    entry_cci_tf_cols: List[str],
    exit_b_mode: str,
) -> List[Dict]:
    cci = pd.to_numeric(df[cci_col], errors="coerce").astype(float).to_numpy()
    macd_hist = pd.to_numeric(df[macd_hist_col], errors="coerce").astype(float).to_numpy()
    if "macd_line" in df.columns:
        macd_line = pd.to_numeric(df["macd_line"], errors="coerce").astype(float).to_numpy()
    else:
        macd_line = pd.to_numeric(pd.Series([math.nan] * int(len(df))), errors="coerce").astype(float).to_numpy()
    if "macd_signal" in df.columns:
        macd_signal = pd.to_numeric(df["macd_signal"], errors="coerce").astype(float).to_numpy()
    else:
        macd_signal = pd.to_numeric(pd.Series([math.nan] * int(len(df))), errors="coerce").astype(float).to_numpy()
    if "stoch_k" in df.columns:
        stoch_k = pd.to_numeric(df["stoch_k"], errors="coerce").astype(float).to_numpy()
    else:
        stoch_k = pd.to_numeric(pd.Series([math.nan] * int(len(df))), errors="coerce").astype(float).to_numpy()
    if "stoch_d" in df.columns:
        stoch_d = pd.to_numeric(df["stoch_d"], errors="coerce").astype(float).to_numpy()
    else:
        stoch_d = pd.to_numeric(pd.Series([math.nan] * int(len(df))), errors="coerce").astype(float).to_numpy()
    if "kvo" in df.columns:
        kvo = pd.to_numeric(df["kvo"], errors="coerce").astype(float).to_numpy()
    else:
        kvo = pd.to_numeric(pd.Series([math.nan] * int(len(df))), errors="coerce").astype(float).to_numpy()
    if "klinger_signal" in df.columns:
        klinger_signal = pd.to_numeric(df["klinger_signal"], errors="coerce").astype(float).to_numpy()
    else:
        klinger_signal = pd.to_numeric(pd.Series([math.nan] * int(len(df))), errors="coerce").astype(float).to_numpy()

    base_series = [signal_from] if str(mode) == "single" else list(selected_series)
    base_series = [s for s in base_series if s in series_to_col]
    if not base_series:
        return []

    stop_ref = str(stop_ref_series or "").strip()
    tracking_series = list(base_series)
    if bool(use_fixed_stop) and stop_ref and stop_ref in series_to_col and stop_ref not in tracking_series:
        tracking_series.append(stop_ref)

    last_confirmed: Dict[str, Dict[str, Optional[Dict]]] = {}
    for s in tracking_series:
        last_confirmed[s] = {"creux": None, "sommet": None}

    in_zone = False
    zone_type: Optional[str] = None
    zone_id = -1
    zone_start_i: Optional[int] = None

    potential: Dict[str, Optional[Dict]] = {s: None for s in tracking_series}
    potential_ready: Dict[str, Optional[Dict]] = {s: None for s in tracking_series}

    pending_bull: Optional[Dict] = None
    pending_bear: Optional[Dict] = None

    last_bull_hist_cross_i: Optional[int] = None
    last_bear_hist_cross_i: Optional[int] = None

    cci_tf_map: Dict[str, object] = {}
    for c in list(entry_cci_tf_cols or []):
        if c in df.columns:
            cci_tf_map[str(c)] = pd.to_numeric(df[str(c)], errors="coerce").astype(float).to_numpy()

    def _cci_tf_confluence_ok(i0: int, zt: str) -> bool:
        if not cci_tf_map:
            return True
        if int(i0) < 0:
            return False
        for c, arr in cci_tf_map.items():
            try:
                x = float(arr[int(i0)])
            except Exception:
                return False
            if not math.isfinite(float(x)):
                return False
            if str(zt) == "creux":
                if float(x) >= float(cci_low):
                    return False
            elif str(zt) == "sommet":
                if float(x) <= float(cci_high):
                    return False
        return True

    pos = 0
    entry: Optional[Dict] = None
    trades: List[Dict] = []
    exit_arm_i: Optional[int] = None

    def _close_position(i: int, reason: str, exit_price_override: Optional[float] = None) -> None:
        nonlocal pos, entry, trades, exit_arm_i
        if pos == 0 or entry is None:
            return
        if exit_price_override is not None:
            exit_price = safe_float(exit_price_override)
        else:
            exit_price = safe_float(df.iloc[i].get("close"))
        if exit_price is None or float(exit_price) == 0.0:
            return

        entry_price = float(entry["entry_price"])
        side = int(entry["side"])
        if side == 1:
            pct = 100.0 * ((float(exit_price) - entry_price) / entry_price)
        else:
            pct = 100.0 * ((entry_price - float(exit_price)) / entry_price)

        trade = {
            **entry,
            "exit_i": int(i),
            "exit_ts": df.iloc[i].get("ts"),
            "exit_dt": df.iloc[i].get("dt"),
            "exit_price": float(exit_price),
            "pct": float(pct),
            "exit_reason": str(reason),
        }
        trades.append(trade)
        pos = 0
        entry = None
        exit_arm_i = None

    def _cross_up(prev_a: float, a: float, prev_b: float, b: float) -> bool:
        if not (
            math.isfinite(float(prev_a))
            and math.isfinite(float(a))
            and math.isfinite(float(prev_b))
            and math.isfinite(float(b))
        ):
            return False
        return float(prev_a) <= float(prev_b) and float(a) > float(b)

    def _cross_down(prev_a: float, a: float, prev_b: float, b: float) -> bool:
        if not (
            math.isfinite(float(prev_a))
            and math.isfinite(float(a))
            and math.isfinite(float(prev_b))
            and math.isfinite(float(b))
        ):
            return False
        return float(prev_a) >= float(prev_b) and float(a) < float(b)

    def _hist_abs_growth(prev_h: float, h: float) -> bool:
        if not (math.isfinite(float(prev_h)) and math.isfinite(float(h))):
            return False
        return abs(float(h)) > abs(float(prev_h))

    def _compute_fixed_stop(side: int, signal: Dict) -> Optional[float]:
        if not bool(use_fixed_stop):
            return None
        per_series = signal.get("per_series") or {}
        price_meta = per_series.get("price")
        if not isinstance(price_meta, dict):
            return None
        prev_conf = price_meta.get("prev_confirmed")
        if not isinstance(prev_conf, dict):
            return None
        ref = safe_float(prev_conf.get("value"))
        if ref is None or float(ref) == 0.0:
            return None
        buf = float(stop_buffer_pct or 0.0)
        if int(side) == 1:
            return float(ref) * (1.0 - buf)
        return float(ref) * (1.0 + buf)

    def _open_position(i: int, side: int, signal: Dict) -> None:
        nonlocal pos, entry, exit_arm_i
        entry_price = safe_float(df.iloc[i].get("close"))
        if entry_price is None or float(entry_price) == 0.0:
            return
        stop_price = _compute_fixed_stop(int(side), signal)
        pos = int(side)
        entry = {
            "side": int(side),
            "entry_i": int(i),
            "entry_ts": df.iloc[i].get("ts"),
            "entry_dt": df.iloc[i].get("dt"),
            "entry_price": float(entry_price),
            "stop_price": stop_price,
            "signal": signal,
        }
        exit_arm_i = None

    def _apply_signal(i: int, direction: str, signal: Dict) -> None:
        nonlocal pos, pending_bull, pending_bear
        td = str(trade_direction or "both").strip().lower()

        if str(direction) == "bull":
            if td == "short":
                return
            if pos == 0:
                _open_position(i, 1, signal)
            pending_bull = None
            return

        if str(direction) == "bear":
            if td == "long":
                pending_bear = None
                return
            if pos == 0:
                _open_position(i, -1, signal)
            pending_bear = None
            return

    n = int(len(df))
    for i in range(n):
        if int(i) >= int(start_i) and pos != 0 and entry is not None:
            stop_price = safe_float(entry.get("stop_price"))
            if stop_price is not None:
                if int(pos) == 1:
                    lo = safe_float(df.iloc[i].get("low"))
                    if lo is not None and float(lo) <= float(stop_price):
                        _close_position(i, "stop", exit_price_override=float(stop_price))
                elif int(pos) == -1:
                    hi = safe_float(df.iloc[i].get("high"))
                    if hi is not None and float(hi) >= float(stop_price):
                        _close_position(i, "stop", exit_price_override=float(stop_price))

        if (
            int(i) >= int(start_i)
            and pos != 0
            and entry is not None
            and exit_arm_i is not None
            and int(i) > 0
            and int(i) > int(exit_arm_i)
        ):
            em = str(exit_b_mode or "macd").strip().lower()
            if em == "none":
                pass
            if em == "macd":
                if int(pos) == 1 and _cross_down(
                    float(macd_line[i - 1]),
                    float(macd_line[i]),
                    float(macd_signal[i - 1]),
                    float(macd_signal[i]),
                ):
                    _close_position(i, "exit_macd")
                elif int(pos) == -1 and _cross_up(
                    float(macd_line[i - 1]),
                    float(macd_line[i]),
                    float(macd_signal[i - 1]),
                    float(macd_signal[i]),
                ):
                    _close_position(i, "exit_macd")
            elif em == "stoch":
                if int(pos) == 1 and _cross_down(
                    float(stoch_k[i - 1]),
                    float(stoch_k[i]),
                    float(stoch_d[i - 1]),
                    float(stoch_d[i]),
                ):
                    _close_position(i, "exit_stoch")
                elif int(pos) == -1 and _cross_up(
                    float(stoch_k[i - 1]),
                    float(stoch_k[i]),
                    float(stoch_d[i - 1]),
                    float(stoch_d[i]),
                ):
                    _close_position(i, "exit_stoch")
            elif em == "klinger":
                if int(pos) == 1 and _cross_down(
                    float(kvo[i - 1]),
                    float(kvo[i]),
                    float(klinger_signal[i - 1]),
                    float(klinger_signal[i]),
                ):
                    _close_position(i, "exit_klinger")
                elif int(pos) == -1 and _cross_up(
                    float(kvo[i - 1]),
                    float(kvo[i]),
                    float(klinger_signal[i - 1]),
                    float(klinger_signal[i]),
                ):
                    _close_position(i, "exit_klinger")

        v = cci[i] if i < len(cci) else math.nan
        prev = cci[i - 1] if i > 0 and i - 1 < len(cci) else None

        if not in_zone:
            if (
                math.isfinite(float(v))
                and float(v) < float(cci_low)
                and (
                    prev is None
                    or (math.isfinite(float(prev)) and float(prev) >= float(cci_low))
                    or (prev is not None and not math.isfinite(float(prev)))
                )
            ):
                in_zone = True
                zone_type = "creux"
                zone_id += 1
                zone_start_i = int(i)
                for s in tracking_series:
                    potential[s] = None
            elif (
                math.isfinite(float(v))
                and float(v) > float(cci_high)
                and (
                    prev is None
                    or (math.isfinite(float(prev)) and float(prev) <= float(cci_high))
                    or (prev is not None and not math.isfinite(float(prev)))
                )
            ):
                in_zone = True
                zone_type = "sommet"
                zone_id += 1
                zone_start_i = int(i)
                for s in tracking_series:
                    potential[s] = None
        else:
            if zone_type == "creux" and math.isfinite(float(v)) and float(v) >= float(cci_low):
                struct_ok: Dict[str, bool] = {}
                per_series: Dict[str, Dict] = {}
                for s in tracking_series:
                    prev_conf = last_confirmed[s]["creux"]
                    cur_pot = potential.get(s)
                    confirmed = None
                    if cur_pot is not None:
                        confirmed = {**cur_pot, "status": "confirmed", "confirmed_i": int(i)}
                    ok = False
                    if prev_conf is not None and confirmed is not None:
                        if float(confirmed["value"]) > float(prev_conf["value"]):
                            ok = True
                    if s in base_series:
                        struct_ok[s] = bool(ok)
                    per_series[s] = {"potential": confirmed, "prev_confirmed": prev_conf}

                n_ok = sum(1 for v0 in struct_ok.values() if bool(v0))
                if int(n_ok) >= int(max(1, min_confluence)) and _cci_tf_confluence_ok(max(0, int(i) - 1), "creux"):
                    pending_bull = {
                        "zone_id": int(zone_id),
                        "zone_type": "creux",
                        "created_i": int(i),
                        "created_dt": df.iloc[i].get("dt"),
                        "extreme_start_i": int(zone_start_i) if zone_start_i is not None else int(i),
                        "mode": str(mode),
                        "series_ok": [s for s, ok in struct_ok.items() if bool(ok)],
                        "per_series": per_series,
                    }
                for s in tracking_series:
                    if potential[s] is not None:
                        last_confirmed[s]["creux"] = {**potential[s], "status": "confirmed", "confirmed_i": int(i)}
                    potential[s] = None
                    potential_ready[s] = None
                in_zone = False
                zone_type = None
                zone_start_i = None
            elif zone_type == "sommet" and math.isfinite(float(v)) and float(v) <= float(cci_high):
                struct_ok: Dict[str, bool] = {}
                per_series: Dict[str, Dict] = {}
                for s in tracking_series:
                    prev_conf = last_confirmed[s]["sommet"]
                    cur_pot = potential.get(s)
                    confirmed = None
                    if cur_pot is not None:
                        confirmed = {**cur_pot, "status": "confirmed", "confirmed_i": int(i)}
                    ok = False
                    if prev_conf is not None and confirmed is not None:
                        if float(confirmed["value"]) < float(prev_conf["value"]):
                            ok = True
                    if s in base_series:
                        struct_ok[s] = bool(ok)
                    per_series[s] = {"potential": confirmed, "prev_confirmed": prev_conf}

                n_ok = sum(1 for v0 in struct_ok.values() if bool(v0))
                if int(n_ok) >= int(max(1, min_confluence)) and _cci_tf_confluence_ok(max(0, int(i) - 1), "sommet"):
                    pending_bear = {
                        "zone_id": int(zone_id),
                        "zone_type": "sommet",
                        "created_i": int(i),
                        "created_dt": df.iloc[i].get("dt"),
                        "extreme_start_i": int(zone_start_i) if zone_start_i is not None else int(i),
                        "mode": str(mode),
                        "series_ok": [s for s, ok in struct_ok.items() if bool(ok)],
                        "per_series": per_series,
                    }
                for s in tracking_series:
                    if potential[s] is not None:
                        last_confirmed[s]["sommet"] = {**potential[s], "status": "confirmed", "confirmed_i": int(i)}
                    potential[s] = None
                    potential_ready[s] = None
                in_zone = False
                zone_type = None
                zone_start_i = None

        if in_zone and zone_type in {"creux", "sommet"}:
            struct_ok: Dict[str, bool] = {}
            per_series: Dict[str, Dict] = {}
            for s in tracking_series:
                col = series_to_col.get(s)
                if col is None:
                    continue
                vv = safe_float(df.iloc[i].get(col))
                if vv is None:
                    continue

                cur = potential.get(s)
                prev_ready = potential_ready.get(s)
                if cur is None:
                    potential[s] = {
                        "idx": int(i),
                        "value": float(vv),
                        "type": str(zone_type),
                        "serie": str(s),
                        "ts": df.iloc[i].get("ts"),
                        "dt": df.iloc[i].get("dt"),
                        "status": "potential",
                        "zone_id": int(zone_id),
                    }
                    potential_ready[s] = None
                    if (
                        int(extreme_confirm_bars) <= 0
                        and entry is not None
                        and s == stop_ref
                        and int(i) >= int(start_i)
                    ):
                        if (
                            int(pos) == 1
                            and str(zone_type) == "sommet"
                            and int(entry.get("entry_i") or 0) <= int(i)
                        ):
                            exit_arm_i = int(i)
                        elif (
                            int(pos) == -1
                            and str(zone_type) == "creux"
                            and int(entry.get("entry_i") or 0) <= int(i)
                        ):
                            exit_arm_i = int(i)
                else:
                    if zone_type == "creux" and float(vv) < float(cur["value"]):
                        potential[s] = {
                            **cur,
                            "idx": int(i),
                            "value": float(vv),
                            "ts": df.iloc[i].get("ts"),
                            "dt": df.iloc[i].get("dt"),
                        }
                        potential_ready[s] = None
                        if (
                            int(extreme_confirm_bars) <= 0
                            and entry is not None
                            and s == stop_ref
                            and int(i) >= int(start_i)
                        ):
                            if int(pos) == -1 and int(entry.get("entry_i") or 0) <= int(i):
                                exit_arm_i = int(i)
                    elif zone_type == "sommet" and float(vv) > float(cur["value"]):
                        potential[s] = {
                            **cur,
                            "idx": int(i),
                            "value": float(vv),
                            "ts": df.iloc[i].get("ts"),
                            "dt": df.iloc[i].get("dt"),
                        }
                        potential_ready[s] = None
                        if (
                            int(extreme_confirm_bars) <= 0
                            and entry is not None
                            and s == stop_ref
                            and int(i) >= int(start_i)
                        ):
                            if int(pos) == 1 and int(entry.get("entry_i") or 0) <= int(i):
                                exit_arm_i = int(i)

                pot_cur = potential.get(s)
                if pot_cur is None:
                    continue

                pot: Optional[Dict] = None
                if int(extreme_confirm_bars) <= 0:
                    pot = pot_cur
                else:
                    pot_i = int(pot_cur.get("idx") or 0)
                    if int(i) >= int(pot_i) + int(extreme_confirm_bars):
                        pot = pot_cur
                        if prev_ready is None:
                            potential_ready[s] = pot_cur
                            if entry is not None and s == stop_ref and int(i) >= int(start_i):
                                if (
                                    int(pos) == 1
                                    and str(zone_type) == "sommet"
                                    and int(entry.get("entry_i") or 0) <= int(pot_i)
                                ):
                                    exit_arm_i = int(i)
                                elif (
                                    int(pos) == -1
                                    and str(zone_type) == "creux"
                                    and int(entry.get("entry_i") or 0) <= int(pot_i)
                                ):
                                    exit_arm_i = int(i)
                    else:
                        potential_ready[s] = None

                prev_conf = last_confirmed[s][zone_type]
                ok = False
                if prev_conf is not None and pot is not None:
                    if zone_type == "creux" and float(pot["value"]) > float(prev_conf["value"]):
                        ok = True
                    elif zone_type == "sommet" and float(pot["value"]) < float(prev_conf["value"]):
                        ok = True

                if s in base_series:
                    struct_ok[s] = bool(ok)
                per_series[s] = {"potential": pot, "prev_confirmed": prev_conf}

            n_ok = sum(1 for v0 in struct_ok.values() if bool(v0))
            if int(n_ok) >= int(max(1, min_confluence)) and _cci_tf_confluence_ok(int(i), str(zone_type)):
                signal = {
                    "zone_id": int(zone_id),
                    "zone_type": str(zone_type),
                    "created_i": int(i),
                    "created_dt": df.iloc[i].get("dt"),
                    "extreme_start_i": int(zone_start_i) if zone_start_i is not None else int(i),
                    "mode": str(mode),
                    "series_ok": [s for s, ok in struct_ok.items() if bool(ok)],
                    "per_series": per_series,
                }
                if zone_type == "creux":
                    pending_bull = signal
                elif zone_type == "sommet":
                    pending_bear = signal
            else:
                if zone_type == "creux":
                    pending_bull = None
                elif zone_type == "sommet":
                    pending_bear = None

        if i <= 0 or i >= len(macd_hist):
            continue

        ok_abs = True
        if bool(entry_require_hist_abs_growth):
            ok_abs = _hist_abs_growth(float(macd_hist[i - 1]), float(macd_hist[i]))

        if hist_cross_up(float(macd_hist[i - 1]), float(macd_hist[i])) and bool(ok_abs):
            last_bull_hist_cross_i = int(i)
        elif hist_cross_down(float(macd_hist[i - 1]), float(macd_hist[i])) and bool(ok_abs):
            last_bear_hist_cross_i = int(i)

        if int(i) < int(start_i):
            continue

        if pending_bull is not None and float(macd_hist[i]) > 0.0:
            es = int(pending_bull.get("extreme_start_i") or pending_bull.get("created_i") or 0)
            if last_bull_hist_cross_i is not None and int(last_bull_hist_cross_i) >= int(es):
                _apply_signal(i, "bull", pending_bull)
        elif pending_bear is not None and float(macd_hist[i]) < 0.0:
            es = int(pending_bear.get("extreme_start_i") or pending_bear.get("created_i") or 0)
            if last_bear_hist_cross_i is not None and int(last_bear_hist_cross_i) >= int(es):
                _apply_signal(i, "bear", pending_bear)

    if int(len(df)) > 0 and pos != 0:
        _close_position(int(len(df) - 1), "eod")

    return trades
