from __future__ import annotations

from typing import Any

import pandas as pd

from libs.dca_mean_distance.math import compute_p_new, compute_p_new_short
from libs.dca_mean_distance.state import DcaState, MiniCycle
from libs.dca_mean_distance.tp import cycle_recompute_next_tp_price


def _sync_global_position_from_cycles(state: DcaState) -> None:
    size = 0.0
    w_sum = 0.0

    cycles: list[MiniCycle] = []
    if state.tp_active is not None and state.tp_active.size > 0:
        cycles.append(state.tp_active)
    for c in state.tp_bucket:
        if c.size > 0:
            cycles.append(c)

    for c in cycles:
        size += float(c.size)
        w_sum += float(c.size) * float(c.avg_open)

    state.position_size = float(size)
    state.avg_price = (w_sum / size) if size > 0 else 0.0


def run_backtest_one_side_df(*, cfg: Any, df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty:
        return {}

    if "open_time" not in df.columns:
        if "ts" in df.columns:
            df = df.copy()
            df["open_time"] = df["ts"]
        else:
            raise ValueError("Missing column: open_time (or ts)")
    if "close" not in df.columns:
        raise ValueError("Missing column: close")

    capital_usdt = float(getattr(cfg, "capital_usdt", 0.0))
    max_portions = int(getattr(cfg, "max_portions", 1) or 1)
    leverage = float(getattr(cfg, "leverage", 1.0) or 1.0)
    fee_rate = float(getattr(cfg, "fee_rate", 0.0) or 0.0)
    liquidation_threshold_pct = float(getattr(cfg, "liquidation_threshold_pct", 0.05) or 0.05)

    portion_margin = capital_usdt / float(max_portions)
    q_notional = portion_margin * leverage
    q_margin = (q_notional / leverage) if leverage > 0 else 0.0

    d_start = float(getattr(cfg, "d_start_pct", 0.0)) / 100.0
    d_step = float(getattr(cfg, "d_step_pct", 0.0)) / 100.0
    tp_frac = float(getattr(cfg, "tp_pct", 0.0)) / 100.0

    is_long = str(getattr(cfg, "side", "long") or "long").strip().lower() != "short"
    side_label = "LONG" if is_long else "SHORT"

    tp_mode = str(getattr(cfg, "tp_mode", "tp_full") or "tp_full").strip().lower()
    if tp_mode not in {"tp_full", "tp_cycles"}:
        raise ValueError(f"Unexpected cfg.tp_mode: {tp_mode}")

    need_prev_tranche_entry = bool(getattr(cfg, "macd_hist_flip_enabled", False)) and bool(
        getattr(cfg, "macd_prev_opposite_tranche_enabled", False)
    )
    need_prev_tranche_tp_partial = bool(getattr(cfg, "tp_partial_macd_hist_flip_enabled", False)) and bool(
        getattr(cfg, "tp_partial_prev_opposite_tranche_enabled", False)
    )

    state = DcaState(wallet=capital_usdt)
    equity_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    run_stats: dict[str, Any] = {
        "dca_price_triggers": 0,
        "dca_blocked_indicator": 0,
        "dca_blocked_mfi": 0,
        "dca_blocked_macd": 0,
        "dca_blocked_macd_prev_tranche": 0,
        "dca_blocked_macd_prev_cci": 0,
        "dca_blocked_macd_prev_cci_medium": 0,
        "dca_blocked_macd_prev_cci_slow": 0,
        "dca_blocked_macd_prev_mfi": 0,
        "dca_blocked_macd_prev_dmi": 0,
        "dca_blocked_margin": 0,
        "dca_executed": 0,
    }

    prev_hist_sign = 0
    current_tranche_sign = 0
    current_tranche_seen_cci = False
    current_tranche_seen_cci_medium = False
    current_tranche_seen_cci_slow = False
    current_tranche_seen_mfi = False
    current_tranche_seen_dmi = False
    current_tranche_seen_dmi_di_align = False
    current_tranche_dmi_prev_diff: float | None = None
    current_tranche_dmi_dx_was_above = False
    current_tranche_dmi_last_cross: str | None = None
    current_tranche_seen_cci_tp = False
    current_tranche_seen_cci_medium_tp = False
    current_tranche_seen_cci_slow_tp = False
    current_tranche_seen_mfi_tp = False
    current_tranche_seen_dmi_tp = False
    last_tranche_sign = 0
    last_tranche_seen_cci = False
    last_tranche_seen_cci_medium = False
    last_tranche_seen_cci_slow = False
    last_tranche_seen_mfi = False
    last_tranche_seen_dmi = False
    last_tranche_seen_cci_tp = False
    last_tranche_seen_cci_medium_tp = False
    last_tranche_seen_cci_slow_tp = False
    last_tranche_seen_mfi_tp = False
    last_tranche_seen_dmi_tp = False

    prev_stoch_k: float | None = None
    prev_stoch_d: float | None = None

    for _, row in df.iterrows():
        ts = int(row.get("open_time"))
        price = float(row.get("close"))

        cci12_val = row.get("cci12", pd.NA)
        mfi_val = row.get("mfi", pd.NA)
        macd_hist_val = row.get("macd_hist", pd.NA)
        macd_prev_tranche_cci_val = row.get("macd_prev_tranche_cci", pd.NA)
        macd_prev_tranche_cci_medium_val = row.get("macd_prev_tranche_cci_medium", pd.NA)
        macd_prev_tranche_cci_slow_val = row.get("macd_prev_tranche_cci_slow", pd.NA)
        macd_prev_tranche_mfi_val = row.get("macd_prev_tranche_mfi", pd.NA)
        tp_prev_tranche_cci_val = row.get("tp_prev_tranche_cci", pd.NA)
        tp_prev_tranche_cci_medium_val = row.get("tp_prev_tranche_cci_medium", pd.NA)
        tp_prev_tranche_cci_slow_val = row.get("tp_prev_tranche_cci_slow", pd.NA)
        tp_prev_tranche_mfi_val = row.get("tp_prev_tranche_mfi", pd.NA)
        stoch_k_val = row.get("stoch_k", pd.NA)
        stoch_d_val = row.get("stoch_d", pd.NA)

        dmi_adx_val = row.get("dmi_adx", pd.NA)
        dmi_plus_di_val = row.get("dmi_plus_di", pd.NA)
        dmi_minus_di_val = row.get("dmi_minus_di", pd.NA)
        dmi_dx_val = row.get("dmi_dx", pd.NA)
        tp_dmi_adx_val = row.get("tp_dmi_adx", pd.NA)
        tp_dmi_plus_di_val = row.get("tp_dmi_plus_di", pd.NA)
        tp_dmi_minus_di_val = row.get("tp_dmi_minus_di", pd.NA)
        tp_dmi_dx_val = row.get("tp_dmi_dx", pd.NA)

        hist_sign = 0
        if not pd.isna(macd_hist_val):
            h = float(macd_hist_val)
            if h > 0:
                hist_sign = 1
            elif h < 0:
                hist_sign = -1
            else:
                hist_sign = prev_hist_sign

        prev_hist_sign_before = int(prev_hist_sign)
        macd_flip_to_side = False
        sign_changed = False
        if prev_hist_sign != 0 and hist_sign != 0 and hist_sign != prev_hist_sign:
            sign_changed = True
            if is_long:
                macd_flip_to_side = prev_hist_sign == -1 and hist_sign == 1
            else:
                macd_flip_to_side = prev_hist_sign == 1 and hist_sign == -1

        if hist_sign != 0 and current_tranche_sign == 0:
            current_tranche_sign = hist_sign
            current_tranche_seen_cci = False
            current_tranche_seen_cci_medium = False
            current_tranche_seen_cci_slow = False
            current_tranche_seen_mfi = False
            current_tranche_seen_dmi = False
            current_tranche_seen_dmi_di_align = False
            current_tranche_dmi_prev_diff = None
            current_tranche_dmi_dx_was_above = False
            current_tranche_dmi_last_cross = None
            current_tranche_seen_cci_tp = False
            current_tranche_seen_cci_medium_tp = False
            current_tranche_seen_cci_slow_tp = False
            current_tranche_seen_mfi_tp = False
            current_tranche_seen_dmi_tp = False

        if sign_changed and current_tranche_sign != 0:
            if bool(getattr(cfg, "macd_prev_tranche_dmi_enabled", False)):
                dx_ok = True
                if bool(current_tranche_dmi_dx_was_above):
                    dx_ok = current_tranche_dmi_last_cross == "down"
                current_tranche_seen_dmi = bool(current_tranche_seen_dmi_di_align and dx_ok)

            last_tranche_sign = current_tranche_sign
            last_tranche_seen_cci = bool(current_tranche_seen_cci)
            last_tranche_seen_cci_medium = bool(current_tranche_seen_cci_medium)
            last_tranche_seen_cci_slow = bool(current_tranche_seen_cci_slow)
            last_tranche_seen_mfi = bool(current_tranche_seen_mfi)
            last_tranche_seen_dmi = bool(current_tranche_seen_dmi)
            last_tranche_seen_cci_tp = bool(current_tranche_seen_cci_tp)
            last_tranche_seen_cci_medium_tp = bool(current_tranche_seen_cci_medium_tp)
            last_tranche_seen_cci_slow_tp = bool(current_tranche_seen_cci_slow_tp)
            last_tranche_seen_mfi_tp = bool(current_tranche_seen_mfi_tp)
            last_tranche_seen_dmi_tp = bool(current_tranche_seen_dmi_tp)

            current_tranche_sign = hist_sign
            current_tranche_seen_cci = False
            current_tranche_seen_cci_medium = False
            current_tranche_seen_cci_slow = False
            current_tranche_seen_mfi = False
            current_tranche_seen_dmi = False
            current_tranche_seen_cci_tp = False
            current_tranche_seen_cci_medium_tp = False
            current_tranche_seen_cci_slow_tp = False
            current_tranche_seen_mfi_tp = False
            current_tranche_seen_dmi_tp = False

        if need_prev_tranche_entry and current_tranche_sign != 0:
            if bool(getattr(cfg, "macd_prev_tranche_cci_enabled", False)) and (not current_tranche_seen_cci):
                if not pd.isna(macd_prev_tranche_cci_val):
                    cci_f = float(macd_prev_tranche_cci_val)
                    if current_tranche_sign > 0:
                        current_tranche_seen_cci = cci_f >= abs(float(getattr(cfg, "macd_prev_tranche_cci_bull_threshold", 100.0)))
                    else:
                        current_tranche_seen_cci = cci_f <= (-abs(float(getattr(cfg, "macd_prev_tranche_cci_bear_threshold", 100.0))))

            if bool(getattr(cfg, "macd_prev_tranche_cci_medium_enabled", False)) and (not current_tranche_seen_cci_medium):
                if not pd.isna(macd_prev_tranche_cci_medium_val):
                    cci_f = float(macd_prev_tranche_cci_medium_val)
                    if current_tranche_sign > 0:
                        current_tranche_seen_cci_medium = cci_f >= abs(float(getattr(cfg, "macd_prev_tranche_cci_medium_bull_threshold", 100.0)))
                    else:
                        current_tranche_seen_cci_medium = cci_f <= (-abs(float(getattr(cfg, "macd_prev_tranche_cci_medium_bear_threshold", 100.0))))

            if bool(getattr(cfg, "macd_prev_tranche_cci_slow_enabled", False)) and (not current_tranche_seen_cci_slow):
                if not pd.isna(macd_prev_tranche_cci_slow_val):
                    cci_f = float(macd_prev_tranche_cci_slow_val)
                    if current_tranche_sign > 0:
                        current_tranche_seen_cci_slow = cci_f >= abs(float(getattr(cfg, "macd_prev_tranche_cci_slow_bull_threshold", 100.0)))
                    else:
                        current_tranche_seen_cci_slow = cci_f <= (-abs(float(getattr(cfg, "macd_prev_tranche_cci_slow_bear_threshold", 100.0))))

            if bool(getattr(cfg, "macd_prev_tranche_mfi_enabled", False)) and (not current_tranche_seen_mfi):
                if not pd.isna(macd_prev_tranche_mfi_val):
                    mfi_f = float(macd_prev_tranche_mfi_val)
                    if current_tranche_sign > 0:
                        current_tranche_seen_mfi = mfi_f >= float(getattr(cfg, "macd_prev_tranche_mfi_high_threshold", 80.0))
                    else:
                        current_tranche_seen_mfi = mfi_f <= float(getattr(cfg, "macd_prev_tranche_mfi_low_threshold", 20.0))

            if bool(getattr(cfg, "macd_prev_tranche_dmi_enabled", False)):
                if (
                    (not pd.isna(dmi_adx_val))
                    and (not pd.isna(dmi_dx_val))
                    and (not pd.isna(dmi_plus_di_val))
                    and (not pd.isna(dmi_minus_di_val))
                ):
                    adx_f = float(dmi_adx_val)
                    dx_f = float(dmi_dx_val)
                    plus_f = float(dmi_plus_di_val)
                    minus_f = float(dmi_minus_di_val)

                    di_align = (plus_f > minus_f) if current_tranche_sign > 0 else (minus_f > plus_f)
                    if di_align:
                        current_tranche_seen_dmi_di_align = True

                    diff = float(dx_f - adx_f)
                    if diff > 0:
                        current_tranche_dmi_dx_was_above = True

                    if current_tranche_dmi_prev_diff is not None:
                        if (current_tranche_dmi_prev_diff <= 0.0) and (diff > 0.0):
                            current_tranche_dmi_last_cross = "up"
                        elif (current_tranche_dmi_prev_diff >= 0.0) and (diff < 0.0):
                            current_tranche_dmi_last_cross = "down"
                    current_tranche_dmi_prev_diff = diff

                    # NB: la validation finale se fait à la fin de la tranche (au flip du MACD hist),
                    # pour respecter la contrainte "dernier cross-down".

        if need_prev_tranche_tp_partial and current_tranche_sign != 0:
            if bool(getattr(cfg, "tp_partial_prev_tranche_cci_enabled", False)) and (not current_tranche_seen_cci_tp):
                if not pd.isna(tp_prev_tranche_cci_val):
                    cci_f = float(tp_prev_tranche_cci_val)
                    if current_tranche_sign > 0:
                        current_tranche_seen_cci_tp = cci_f >= abs(float(getattr(cfg, "tp_partial_prev_tranche_cci_bull_threshold", 100.0)))
                    else:
                        current_tranche_seen_cci_tp = cci_f <= (-abs(float(getattr(cfg, "tp_partial_prev_tranche_cci_bear_threshold", 100.0))))

            if bool(getattr(cfg, "tp_partial_prev_tranche_cci_medium_enabled", False)) and (not current_tranche_seen_cci_medium_tp):
                if not pd.isna(tp_prev_tranche_cci_medium_val):
                    cci_f = float(tp_prev_tranche_cci_medium_val)
                    if current_tranche_sign > 0:
                        current_tranche_seen_cci_medium_tp = cci_f >= abs(float(getattr(cfg, "tp_partial_prev_tranche_cci_medium_bull_threshold", 100.0)))
                    else:
                        current_tranche_seen_cci_medium_tp = cci_f <= (-abs(float(getattr(cfg, "tp_partial_prev_tranche_cci_medium_bear_threshold", 100.0))))

            if bool(getattr(cfg, "tp_partial_prev_tranche_cci_slow_enabled", False)) and (not current_tranche_seen_cci_slow_tp):
                if not pd.isna(tp_prev_tranche_cci_slow_val):
                    cci_f = float(tp_prev_tranche_cci_slow_val)
                    if current_tranche_sign > 0:
                        current_tranche_seen_cci_slow_tp = cci_f >= abs(float(getattr(cfg, "tp_partial_prev_tranche_cci_slow_bull_threshold", 100.0)))
                    else:
                        current_tranche_seen_cci_slow_tp = cci_f <= (-abs(float(getattr(cfg, "tp_partial_prev_tranche_cci_slow_bear_threshold", 100.0))))

            if bool(getattr(cfg, "tp_partial_prev_tranche_mfi_enabled", False)) and (not current_tranche_seen_mfi_tp):
                if not pd.isna(tp_prev_tranche_mfi_val):
                    mfi_f = float(tp_prev_tranche_mfi_val)
                    if current_tranche_sign > 0:
                        current_tranche_seen_mfi_tp = mfi_f >= float(getattr(cfg, "tp_partial_prev_tranche_mfi_high_threshold", 80.0))
                    else:
                        current_tranche_seen_mfi_tp = mfi_f <= float(getattr(cfg, "tp_partial_prev_tranche_mfi_low_threshold", 20.0))

            if bool(getattr(cfg, "tp_partial_prev_tranche_dmi_enabled", False)) and (not current_tranche_seen_dmi_tp):
                if (not pd.isna(tp_dmi_dx_val)) and (not pd.isna(tp_dmi_plus_di_val)) and (not pd.isna(tp_dmi_minus_di_val)):
                    dx_f = float(tp_dmi_dx_val)
                    plus_f = float(tp_dmi_plus_di_val)
                    minus_f = float(tp_dmi_minus_di_val)
                    di_max = max(plus_f, minus_f)
                    di_min = min(plus_f, minus_f)
                    dx_ok = (dx_f > di_max) or (dx_f < di_min)
                    di_align = (plus_f > minus_f) if current_tranche_sign > 0 else (minus_f > plus_f)
                    current_tranche_seen_dmi_tp = bool(dx_ok and di_align)

        prev_tranche_ok_dbg: bool | None = None
        if need_prev_tranche_entry:
            expected_prev_sign = -1 if is_long else 1
            prev_sign_ok = last_tranche_sign == expected_prev_sign
            prev_tranche_cci_ok = True
            if bool(getattr(cfg, "macd_prev_tranche_cci_enabled", False)):
                prev_tranche_cci_ok = prev_tranche_cci_ok and bool(last_tranche_seen_cci)
            if bool(getattr(cfg, "macd_prev_tranche_cci_medium_enabled", False)):
                prev_tranche_cci_ok = prev_tranche_cci_ok and bool(last_tranche_seen_cci_medium)
            if bool(getattr(cfg, "macd_prev_tranche_cci_slow_enabled", False)):
                prev_tranche_cci_ok = prev_tranche_cci_ok and bool(last_tranche_seen_cci_slow)
            prev_tranche_mfi_ok = (not bool(getattr(cfg, "macd_prev_tranche_mfi_enabled", False))) or bool(last_tranche_seen_mfi)
            prev_tranche_dmi_ok = (not bool(getattr(cfg, "macd_prev_tranche_dmi_enabled", False))) or bool(last_tranche_seen_dmi)
            prev_tranche_ok_dbg = bool(prev_sign_ok and prev_tranche_cci_ok and prev_tranche_mfi_ok and prev_tranche_dmi_ok)

        tp_prev_tranche_ok_dbg: bool | None = None
        if need_prev_tranche_tp_partial:
            expected_prev_sign = 1 if is_long else -1
            prev_sign_ok = last_tranche_sign == expected_prev_sign
            prev_tranche_cci_ok = True
            if bool(getattr(cfg, "tp_partial_prev_tranche_cci_enabled", False)):
                prev_tranche_cci_ok = prev_tranche_cci_ok and bool(last_tranche_seen_cci_tp)
            if bool(getattr(cfg, "tp_partial_prev_tranche_cci_medium_enabled", False)):
                prev_tranche_cci_ok = prev_tranche_cci_ok and bool(last_tranche_seen_cci_medium_tp)
            if bool(getattr(cfg, "tp_partial_prev_tranche_cci_slow_enabled", False)):
                prev_tranche_cci_ok = prev_tranche_cci_ok and bool(last_tranche_seen_cci_slow_tp)
            prev_tranche_mfi_ok = (not bool(getattr(cfg, "tp_partial_prev_tranche_mfi_enabled", False))) or bool(last_tranche_seen_mfi_tp)
            prev_tranche_dmi_ok = (not bool(getattr(cfg, "tp_partial_prev_tranche_dmi_enabled", False))) or bool(last_tranche_seen_dmi_tp)
            tp_prev_tranche_ok_dbg = bool(prev_sign_ok and prev_tranche_cci_ok and prev_tranche_mfi_ok and prev_tranche_dmi_ok)

        if hist_sign != 0:
            prev_hist_sign = hist_sign

        if tp_mode == "tp_cycles":
            _sync_global_position_from_cycles(state)

        pnl_unrealized = 0.0
        if state.position_size > 0 and state.avg_price > 0:
            if is_long:
                pnl_unrealized = (price - state.avg_price) * state.position_size
            else:
                pnl_unrealized = (state.avg_price - price) * state.position_size
        equity = state.wallet + pnl_unrealized

        wallet_free = state.wallet - state.margin_invested
        margin_total = state.wallet + max(0.0, pnl_unrealized)
        loss_unrealized = max(0.0, -pnl_unrealized)
        liq_limit = (1.0 - liquidation_threshold_pct) * margin_total

        if (state.position_size > 0.0) and (loss_unrealized >= liq_limit):
            state.is_liquidated = True
            state.liquidation_reason = (
                f"LIQUIDATION price={price:.6f} loss={loss_unrealized:.2f} limit={liq_limit:.2f} "
                f"wallet={state.wallet:.2f} margin_invested={state.margin_invested:.2f}"
            )
            notional = state.position_size * price
            trades.append(
                {
                    "timestamp": ts,
                    "price": price,
                    "macd_hist": (None if pd.isna(macd_hist_val) else float(macd_hist_val)),
                    "macd_hist_sign": int(hist_sign),
                    "macd_tranche_sign": int(current_tranche_sign),
                    "macd_prev_tranche_sign": int(last_tranche_sign),
                    "macd_flip_to_side": bool(macd_flip_to_side),
                    "macd_prev_tranche_ok": (None if prev_tranche_ok_dbg is None else bool(prev_tranche_ok_dbg)),
                    "cci12": (None if pd.isna(cci12_val) else float(cci12_val)),
                    "mfi": (None if pd.isna(mfi_val) else float(mfi_val)),
                    "side": side_label,
                    "type": "LIQUIDATION",
                    "qty": state.position_size,
                    "qty_usdt": notional,
                    "pnl_realized": 0.0,
                    "fee": 0.0,
                }
            )
            equity_rows.append(
                {
                    "timestamp": ts,
                    "price": price,
                    "wallet": state.wallet,
                    "wallet_free": wallet_free,
                    "margin_invested": state.margin_invested,
                    "equity": equity,
                    "position_size": state.position_size,
                    "avg_price": state.avg_price,
                    "pnl_unrealized": pnl_unrealized,
                    "is_liquidated": True,
                    "liquidation_reason": state.liquidation_reason,
                }
            )
            break

        can_start_new_cycle = True
        max_cycles_cfg = int(getattr(cfg, "max_cycles", 0) or 0)
        if max_cycles_cfg and state.cycles_completed >= max_cycles_cfg:
            can_start_new_cycle = False
        cooldown_minutes = int(getattr(cfg, "reentry_cooldown_minutes", 0) or 0)
        if cooldown_minutes and state.last_exit_ts:
            cooldown_ms = cooldown_minutes * 60_000
            if (ts - int(state.last_exit_ts)) < cooldown_ms:
                can_start_new_cycle = False

        if tp_mode == "tp_full":
            if state.position_size > 0 and state.avg_price > 0 and tp_frac > 0:
                if is_long:
                    tp_price = state.avg_price * (1.0 + tp_frac)
                    should_tp = price >= tp_price
                else:
                    tp_price = state.avg_price * (1.0 - tp_frac)
                    should_tp = price <= tp_price

                if should_tp:
                    notional = state.position_size * price
                    fee = notional * fee_rate
                    if is_long:
                        pnl_realized = (price - state.avg_price) * state.position_size
                    else:
                        pnl_realized = (state.avg_price - price) * state.position_size

                    state.wallet += pnl_realized
                    state.wallet -= fee

                    trades.append(
                        {
                            "timestamp": ts,
                            "price": price,
                            "macd_hist": (None if pd.isna(macd_hist_val) else float(macd_hist_val)),
                            "macd_hist_sign": int(hist_sign),
                            "macd_tranche_sign": int(current_tranche_sign),
                            "macd_prev_tranche_sign": int(last_tranche_sign),
                            "macd_flip_to_side": bool(macd_flip_to_side),
                            "macd_prev_tranche_ok": (None if prev_tranche_ok_dbg is None else bool(prev_tranche_ok_dbg)),
                            "cci12": (None if pd.isna(cci12_val) else float(cci12_val)),
                            "mfi": (None if pd.isna(mfi_val) else float(mfi_val)),
                            "side": side_label,
                            "type": "TP_FULL",
                            "qty": state.position_size,
                            "qty_usdt": notional,
                            "pnl_realized": pnl_realized,
                            "fee": fee,
                        }
                    )

                    state.position_size = 0.0
                    state.avg_price = 0.0
                    state.current_d_index = 0
                    state.next_target_price = 0.0
                    state.margin_invested = 0.0
                    state.portions_used = 0
                    state.cycles_completed += 1
                    state.last_exit_ts = ts

            if state.position_size <= 0 and state.portions_used == 0 and can_start_new_cycle:
                qty = q_notional / price if price > 0 else 0.0
                if qty > 0:
                    fee = q_notional * fee_rate
                    wallet_free2 = state.wallet - state.margin_invested
                    required_free = q_margin + fee
                    if wallet_free2 < required_free:
                        qty = 0.0

                if qty > 0:
                    new_size = state.position_size + qty
                    new_avg = (state.position_size * state.avg_price + q_notional) / new_size
                    state.position_size = new_size
                    state.avg_price = new_avg
                    state.wallet -= fee
                    state.margin_invested += q_margin
                    state.portions_used = 1

                    trades.append(
                        {
                            "timestamp": ts,
                            "price": price,
                            "macd_hist": (None if pd.isna(macd_hist_val) else float(macd_hist_val)),
                            "macd_hist_sign": int(hist_sign),
                            "macd_tranche_sign": int(current_tranche_sign),
                            "macd_prev_tranche_sign": int(last_tranche_sign),
                            "macd_flip_to_side": bool(macd_flip_to_side),
                            "macd_prev_tranche_ok": (None if prev_tranche_ok_dbg is None else bool(prev_tranche_ok_dbg)),
                            "cci12": (None if pd.isna(cci12_val) else float(cci12_val)),
                            "mfi": (None if pd.isna(mfi_val) else float(mfi_val)),
                            "side": side_label,
                            "type": "BASE",
                            "qty": qty,
                            "qty_usdt": q_notional,
                            "pnl_realized": 0.0,
                            "fee": fee,
                            "margin_cost": q_margin,
                        }
                    )

                    state.current_d_index = 1
                    d = d_start
                    if is_long:
                        state.next_target_price = compute_p_new(state.position_size, state.avg_price, q_notional, d)
                    else:
                        state.next_target_price = compute_p_new_short(state.position_size, state.avg_price, q_notional, d)

            if state.position_size > 0 and state.portions_used < max_portions:
                if state.current_d_index > 0 and state.next_target_price > 0.0:
                    price_ok = (price <= state.next_target_price) if is_long else (price >= state.next_target_price)

                    dca_signal = True
                    prev_tranche_ok = True
                    if bool(getattr(cfg, "macd_hist_flip_enabled", False)):
                        if need_prev_tranche_entry:
                            prev_tranche_ok = bool(prev_tranche_ok_dbg) if prev_tranche_ok_dbg is not None else False
                        dca_signal = bool(macd_flip_to_side and prev_tranche_ok)

                    should_dca = bool(dca_signal and price_ok)
                    if should_dca:
                        run_stats["dca_price_triggers"] += 1

                    cci_ok = True
                    if bool(getattr(cfg, "cci12_enabled", False)):
                        if pd.isna(cci12_val):
                            cci_ok = False
                        else:
                            cci_f = float(cci12_val)
                            if is_long:
                                cci_ok = cci_f <= (-abs(float(getattr(cfg, "cci_long_threshold", 100.0))))
                            else:
                                cci_ok = cci_f >= (abs(float(getattr(cfg, "cci_short_threshold", 100.0))))

                    mfi_ok = True
                    if bool(getattr(cfg, "mfi_enabled", False)):
                        if pd.isna(mfi_val):
                            mfi_ok = False
                        else:
                            mfi_f = float(mfi_val)
                            if is_long:
                                mfi_ok = mfi_f <= float(getattr(cfg, "mfi_long_threshold", 20.0))
                            else:
                                mfi_ok = mfi_f >= float(getattr(cfg, "mfi_short_threshold", 80.0))

                    if should_dca and (not cci_ok):
                        run_stats["dca_blocked_indicator"] += 1
                    if should_dca and (not mfi_ok):
                        run_stats["dca_blocked_mfi"] += 1

                    if bool(getattr(cfg, "macd_hist_flip_enabled", False)) and bool(macd_flip_to_side) and (not prev_tranche_ok):
                        run_stats["dca_blocked_macd"] += 1
                        if need_prev_tranche_entry:
                            run_stats["dca_blocked_macd_prev_tranche"] += 1

                    if should_dca and cci_ok and mfi_ok:
                        qty = q_notional / price if price > 0 else 0.0
                        if qty > 0:
                            fee = q_notional * fee_rate
                            wallet_free2 = state.wallet - state.margin_invested
                            required_free = q_margin + fee
                            if wallet_free2 < required_free:
                                run_stats["dca_blocked_margin"] += 1
                                qty = 0.0

                        if qty > 0:
                            new_size = state.position_size + qty
                            new_avg = (state.position_size * state.avg_price + q_notional) / new_size
                            state.position_size = new_size
                            state.avg_price = new_avg
                            state.wallet -= fee
                            state.margin_invested += q_margin
                            state.portions_used += 1

                            trades.append(
                                {
                                    "timestamp": ts,
                                    "price": price,
                                    "macd_hist": (None if pd.isna(macd_hist_val) else float(macd_hist_val)),
                                    "macd_hist_sign": int(hist_sign),
                                    "macd_tranche_sign": int(current_tranche_sign),
                                    "macd_prev_tranche_sign": int(last_tranche_sign),
                                    "macd_flip_to_side": bool(macd_flip_to_side),
                                    "macd_prev_tranche_ok": (None if prev_tranche_ok_dbg is None else bool(prev_tranche_ok_dbg)),
                                    "cci12": (None if pd.isna(cci12_val) else float(cci12_val)),
                                    "mfi": (None if pd.isna(mfi_val) else float(mfi_val)),
                                    "side": side_label,
                                    "type": "DCA",
                                    "qty": qty,
                                    "qty_usdt": q_notional,
                                    "pnl_realized": 0.0,
                                    "fee": fee,
                                    "margin_cost": q_margin,
                                }
                            )
                            run_stats["dca_executed"] += 1

                            state.current_d_index += 1
                            d = d_start + (state.current_d_index - 1) * d_step
                            if is_long:
                                state.next_target_price = compute_p_new(state.position_size, state.avg_price, q_notional, d)
                            else:
                                state.next_target_price = compute_p_new_short(state.position_size, state.avg_price, q_notional, d)

        if tp_mode == "tp_cycles":
            entry_signal_raw = True
            if bool(getattr(cfg, "macd_hist_flip_enabled", False)):
                prev_ok = True
                if need_prev_tranche_entry:
                    prev_ok = bool(prev_tranche_ok_dbg) if prev_tranche_ok_dbg is not None else False
                entry_signal_raw = bool(macd_flip_to_side and prev_ok)

            macd_flip_away_from_side = False
            if prev_hist_sign_before != 0 and hist_sign != 0 and hist_sign != prev_hist_sign_before:
                if is_long:
                    macd_flip_away_from_side = prev_hist_sign_before == 1 and hist_sign == -1
                else:
                    macd_flip_away_from_side = prev_hist_sign_before == -1 and hist_sign == 1

            tp_partial_mode = str(getattr(cfg, "tp_partial_mode", "macd_hist_flip") or "macd_hist_flip").strip().lower()
            if tp_partial_mode in {"stoch", "stoch_cross", "stoch_kd_cross"}:
                tp_partial_mode = "stoch_cross"

            tp_partial_signal_raw = False
            if bool(getattr(cfg, "tp_partial_macd_hist_flip_enabled", False)):
                if tp_partial_mode == "macd_hist_flip":
                    tp_partial_signal_raw = bool(macd_flip_away_from_side)
                    if need_prev_tranche_tp_partial:
                        tp_ok = bool(tp_prev_tranche_ok_dbg) if tp_prev_tranche_ok_dbg is not None else False
                        tp_partial_signal_raw = bool(tp_partial_signal_raw and tp_ok)
                elif tp_partial_mode == "stoch_cross":
                    cross_ok = False
                    if (prev_stoch_k is not None) and (prev_stoch_d is not None) and (not pd.isna(stoch_k_val)) and (not pd.isna(stoch_d_val)):
                        k0 = float(prev_stoch_k)
                        d0 = float(prev_stoch_d)
                        k1 = float(stoch_k_val)
                        d1 = float(stoch_d_val)

                        min_k = getattr(cfg, "tp_partial_stoch_min_k", None)
                        max_k = getattr(cfg, "tp_partial_stoch_max_k", None)
                        if (min_k is not None) and (k1 < float(min_k)):
                            cross_ok = False
                        elif (max_k is not None) and (k1 > float(max_k)):
                            cross_ok = False
                        else:
                            if is_long:
                                cross_ok = bool(k0 >= d0 and k1 < d1)
                            else:
                                cross_ok = bool(k0 <= d0 and k1 > d1)

                    # Les filtres TP partial sont évalués au moment du signal (pas par tranche entière),
                    # et sont basés sur les colonnes tp_prev_tranche_* (déjà calculées sur la série).
                    filters_ok = True
                    if bool(getattr(cfg, "tp_partial_prev_tranche_cci_enabled", False)):
                        if pd.isna(tp_prev_tranche_cci_val):
                            filters_ok = False
                        else:
                            cci_f = float(tp_prev_tranche_cci_val)
                            if current_tranche_sign > 0:
                                filters_ok = filters_ok and (cci_f >= abs(float(getattr(cfg, "tp_partial_prev_tranche_cci_bull_threshold", 100.0))))
                            else:
                                filters_ok = filters_ok and (cci_f <= (-abs(float(getattr(cfg, "tp_partial_prev_tranche_cci_bear_threshold", 100.0)))) )
                    if bool(getattr(cfg, "tp_partial_prev_tranche_cci_medium_enabled", False)):
                        if pd.isna(tp_prev_tranche_cci_medium_val):
                            filters_ok = False
                        else:
                            cci_f = float(tp_prev_tranche_cci_medium_val)
                            if current_tranche_sign > 0:
                                filters_ok = filters_ok and (cci_f >= abs(float(getattr(cfg, "tp_partial_prev_tranche_cci_medium_bull_threshold", 100.0))))
                            else:
                                filters_ok = filters_ok and (cci_f <= (-abs(float(getattr(cfg, "tp_partial_prev_tranche_cci_medium_bear_threshold", 100.0)))) )
                    if bool(getattr(cfg, "tp_partial_prev_tranche_cci_slow_enabled", False)):
                        if pd.isna(tp_prev_tranche_cci_slow_val):
                            filters_ok = False
                        else:
                            cci_f = float(tp_prev_tranche_cci_slow_val)
                            if current_tranche_sign > 0:
                                filters_ok = filters_ok and (cci_f >= abs(float(getattr(cfg, "tp_partial_prev_tranche_cci_slow_bull_threshold", 100.0))))
                            else:
                                filters_ok = filters_ok and (cci_f <= (-abs(float(getattr(cfg, "tp_partial_prev_tranche_cci_slow_bear_threshold", 100.0)))) )

                    if bool(getattr(cfg, "tp_partial_prev_tranche_mfi_enabled", False)):
                        if pd.isna(tp_prev_tranche_mfi_val):
                            filters_ok = False
                        else:
                            mfi_f = float(tp_prev_tranche_mfi_val)
                            if current_tranche_sign > 0:
                                filters_ok = filters_ok and (mfi_f >= float(getattr(cfg, "tp_partial_prev_tranche_mfi_high_threshold", 80.0)))
                            else:
                                filters_ok = filters_ok and (mfi_f <= float(getattr(cfg, "tp_partial_prev_tranche_mfi_low_threshold", 20.0)))

                    if bool(getattr(cfg, "tp_partial_prev_tranche_dmi_enabled", False)):
                        if pd.isna(tp_dmi_dx_val) or pd.isna(tp_dmi_adx_val):
                            filters_ok = False
                        else:
                            dx_f = float(tp_dmi_dx_val)
                            adx_f = float(tp_dmi_adx_val)
                            filters_ok = filters_ok and (dx_f > adx_f)

                    tp_partial_signal_raw = bool(cross_ok and filters_ok)

            if not pd.isna(stoch_k_val):
                prev_stoch_k = float(stoch_k_val)
            if not pd.isna(stoch_d_val):
                prev_stoch_d = float(stoch_d_val)

            entry_signal = bool(entry_signal_raw and (not state.prev_entry_signal_raw))
            tp_partial_signal = bool(tp_partial_signal_raw and (not state.prev_tp_partial_signal_raw))
            state.prev_entry_signal_raw = bool(entry_signal_raw)
            state.prev_tp_partial_signal_raw = bool(tp_partial_signal_raw)

            for c in state.tp_bucket:
                if c.next_tp_price > 0:
                    if is_long:
                        if price >= c.next_tp_price:
                            c.tp_reached = True
                    else:
                        if price <= c.next_tp_price:
                            c.tp_reached = True

            if state.tp_active is not None and state.tp_active.next_tp_price > 0:
                c = state.tp_active
                if is_long:
                    if price >= c.next_tp_price:
                        c.tp_reached = True
                else:
                    if price <= c.next_tp_price:
                        c.tp_reached = True

            if tp_partial_signal:
                state.tp_phase = "TP"
                if state.tp_active is not None and state.tp_active.size > 0:
                    state.tp_bucket.append(state.tp_active)
                    state.tp_active = None

            if state.tp_phase == "TP" and tp_partial_signal and state.tp_bucket:
                close_ratio = float(getattr(cfg, "tp_close_ratio", 0.5))
                if close_ratio <= 0:
                    close_ratio = 0.0
                if close_ratio > 1.0:
                    close_ratio = 1.0

                new_bucket: list[MiniCycle] = []
                for c in reversed(state.tp_bucket):
                    eligible = bool(c.next_tp_price > 0 and ((price >= c.next_tp_price) if is_long else (price <= c.next_tp_price)))
                    if (not eligible) or c.size <= 0 or close_ratio <= 0:
                        new_bucket.append(c)
                        continue

                    close_qty = float(c.size) * close_ratio
                    if close_qty <= 0:
                        new_bucket.append(c)
                        continue

                    notional = close_qty * price
                    fee = notional * fee_rate
                    if is_long:
                        pnl_realized = (price - c.avg_open) * close_qty
                    else:
                        pnl_realized = (c.avg_open - price) * close_qty

                    state.wallet += pnl_realized
                    state.wallet -= fee

                    margin_release = notional / leverage if leverage > 0 else 0.0
                    state.margin_invested = max(0.0, float(state.margin_invested) - float(margin_release))

                    if c.closed_qty <= 0:
                        c.avg_close = float(price)
                        c.closed_qty = float(close_qty)
                    else:
                        new_closed_qty = float(c.closed_qty) + float(close_qty)
                        c.avg_close = (float(c.avg_close) * float(c.closed_qty) + float(price) * float(close_qty)) / new_closed_qty
                        c.closed_qty = new_closed_qty

                    c.size = max(0.0, float(c.size) - float(close_qty))
                    c.tp_index += 1
                    c.tp_reached = False

                    cycle_recompute_next_tp_price(
                        c,
                        is_long=is_long,
                        tp_d_start_pct=float(getattr(cfg, "tp_d_start_pct", 0.0)),
                        tp_d_step_pct=float(getattr(cfg, "tp_d_step_pct", 0.0)),
                    )

                    trades.append(
                        {
                            "timestamp": ts,
                            "price": price,
                            "macd_hist": (None if pd.isna(macd_hist_val) else float(macd_hist_val)),
                            "macd_hist_sign": int(hist_sign),
                            "macd_tranche_sign": int(current_tranche_sign),
                            "macd_prev_tranche_sign": int(last_tranche_sign),
                            "macd_flip_to_side": bool(macd_flip_to_side),
                            "macd_prev_tranche_ok": (None if prev_tranche_ok_dbg is None else bool(prev_tranche_ok_dbg)),
                            "side": side_label,
                            "type": "TP_PARTIAL",
                            "cycle_id": int(c.cycle_id),
                            "qty": float(close_qty),
                            "qty_usdt": float(notional),
                            "pnl_realized": float(pnl_realized),
                            "fee": float(fee),
                        }
                    )

                    if c.size <= 0:
                        state.cycles_completed += 1
                        state.last_exit_ts = ts
                    else:
                        new_bucket.append(c)

                state.tp_bucket = list(reversed(new_bucket))

            used_portions_f = (state.margin_invested / portion_margin) if portion_margin > 0 else 0.0
            state.portions_used = int(max(0, round(used_portions_f)))

            has_active = bool(state.tp_active is not None and state.tp_active.size > 0)

            if state.tp_phase == "TP":
                if entry_signal and can_start_new_cycle and (used_portions_f + 1.0) <= float(max_portions):
                    allow_base = True
                    if state.tp_bucket:
                        n = float(getattr(cfg, "tp_new_cycle_min_distance_pct", 0.0))
                        if n > 0:
                            if is_long:
                                ref = min(float(c.avg_open) for c in state.tp_bucket if c.avg_open > 0)
                                allow_base = price <= ref * (1.0 - n / 100.0)
                            else:
                                ref = max(float(c.avg_open) for c in state.tp_bucket if c.avg_open > 0)
                                allow_base = price >= ref * (1.0 + n / 100.0)
                    if allow_base and price > 0:
                        qty = q_notional / price
                        fee = q_notional * fee_rate
                        wallet_free2 = state.wallet - state.margin_invested
                        required_free = q_margin + fee
                        if wallet_free2 >= required_free and qty > 0:
                            cycle_id = int(state.tp_next_cycle_id)
                            state.tp_next_cycle_id += 1
                            c = MiniCycle(cycle_id=cycle_id, created_ts=int(ts), size=float(qty), avg_open=float(price))
                            c.current_d_index = 1
                            d = d_start
                            if is_long:
                                c.next_target_price = compute_p_new(c.size, c.avg_open, q_notional, d)
                            else:
                                c.next_target_price = compute_p_new_short(c.size, c.avg_open, q_notional, d)
                            cycle_recompute_next_tp_price(
                                c,
                                is_long=is_long,
                                tp_d_start_pct=float(getattr(cfg, "tp_d_start_pct", 0.0)),
                                tp_d_step_pct=float(getattr(cfg, "tp_d_step_pct", 0.0)),
                            )

                            state.tp_active = c
                            state.wallet -= fee
                            state.margin_invested += q_margin
                            state.tp_phase = "OPEN"

                            trades.append(
                                {
                                    "timestamp": ts,
                                    "price": price,
                                    "macd_hist": (None if pd.isna(macd_hist_val) else float(macd_hist_val)),
                                    "macd_hist_sign": int(hist_sign),
                                    "macd_tranche_sign": int(current_tranche_sign),
                                    "macd_prev_tranche_sign": int(last_tranche_sign),
                                    "macd_flip_to_side": bool(macd_flip_to_side),
                                    "macd_prev_tranche_ok": (None if prev_tranche_ok_dbg is None else bool(prev_tranche_ok_dbg)),
                                    "side": side_label,
                                    "type": "BASE",
                                    "cycle_id": cycle_id,
                                    "qty": float(qty),
                                    "qty_usdt": float(q_notional),
                                    "pnl_realized": 0.0,
                                    "fee": float(fee),
                                    "margin_cost": float(q_margin),
                                }
                            )

            if (state.tp_phase == "OPEN") and (not (state.tp_active is not None and state.tp_active.size > 0)) and can_start_new_cycle and (used_portions_f + 1.0) <= float(max_portions):
                allow_base = True
                if state.tp_bucket:
                    allow_base = bool(entry_signal)
                    n = float(getattr(cfg, "tp_new_cycle_min_distance_pct", 0.0))
                    if allow_base and n > 0:
                        if is_long:
                            ref = min(float(c.avg_open) for c in state.tp_bucket if c.avg_open > 0)
                            allow_base = price <= ref * (1.0 - n / 100.0)
                        else:
                            ref = max(float(c.avg_open) for c in state.tp_bucket if c.avg_open > 0)
                            allow_base = price >= ref * (1.0 + n / 100.0)

                if allow_base and price > 0:
                    qty = q_notional / price
                    fee = q_notional * fee_rate
                    wallet_free2 = state.wallet - state.margin_invested
                    required_free = q_margin + fee
                    if wallet_free2 >= required_free and qty > 0:
                        cycle_id = int(state.tp_next_cycle_id)
                        state.tp_next_cycle_id += 1
                        c = MiniCycle(cycle_id=cycle_id, created_ts=int(ts), size=float(qty), avg_open=float(price))
                        c.current_d_index = 1
                        d = d_start
                        if is_long:
                            c.next_target_price = compute_p_new(c.size, c.avg_open, q_notional, d)
                        else:
                            c.next_target_price = compute_p_new_short(c.size, c.avg_open, q_notional, d)
                        cycle_recompute_next_tp_price(
                            c,
                            is_long=is_long,
                            tp_d_start_pct=float(getattr(cfg, "tp_d_start_pct", 0.0)),
                            tp_d_step_pct=float(getattr(cfg, "tp_d_step_pct", 0.0)),
                        )

                        state.tp_active = c
                        state.wallet -= fee
                        state.margin_invested += q_margin

                        trades.append(
                            {
                                "timestamp": ts,
                                "price": price,
                                "macd_hist": (None if pd.isna(macd_hist_val) else float(macd_hist_val)),
                                "macd_hist_sign": int(hist_sign),
                                "macd_tranche_sign": int(current_tranche_sign),
                                "macd_prev_tranche_sign": int(last_tranche_sign),
                                "macd_flip_to_side": bool(macd_flip_to_side),
                                "macd_prev_tranche_ok": (None if prev_tranche_ok_dbg is None else bool(prev_tranche_ok_dbg)),
                                "side": side_label,
                                "type": "BASE",
                                "cycle_id": cycle_id,
                                "qty": float(qty),
                                "qty_usdt": float(q_notional),
                                "pnl_realized": 0.0,
                                "fee": float(fee),
                                "margin_cost": float(q_margin),
                            }
                        )

            used_portions_f = (state.margin_invested / portion_margin) if portion_margin > 0 else 0.0
            if (
                state.tp_phase == "OPEN"
                and state.tp_active is not None
                and state.tp_active.size > 0
                and used_portions_f < float(max_portions)
            ):
                c = state.tp_active
                if c.current_d_index > 0 and c.next_target_price > 0.0:
                    price_ok = (price <= c.next_target_price) if is_long else (price >= c.next_target_price)

                    dca_signal = True
                    prev_tranche_ok = True
                    if bool(getattr(cfg, "macd_hist_flip_enabled", False)):
                        if need_prev_tranche_entry:
                            prev_tranche_ok = bool(prev_tranche_ok_dbg) if prev_tranche_ok_dbg is not None else False
                        dca_signal = bool(macd_flip_to_side and prev_tranche_ok)

                    should_dca = bool(dca_signal and price_ok)
                    if should_dca:
                        run_stats["dca_price_triggers"] += 1

                    cci_ok = True
                    if bool(getattr(cfg, "cci12_enabled", False)):
                        if pd.isna(cci12_val):
                            cci_ok = False
                        else:
                            cci_f = float(cci12_val)
                            if is_long:
                                cci_ok = cci_f <= (-abs(float(getattr(cfg, "cci_long_threshold", 100.0))))
                            else:
                                cci_ok = cci_f >= (abs(float(getattr(cfg, "cci_short_threshold", 100.0))))

                    mfi_ok = True
                    if bool(getattr(cfg, "mfi_enabled", False)):
                        if pd.isna(mfi_val):
                            mfi_ok = False
                        else:
                            mfi_f = float(mfi_val)
                            if is_long:
                                mfi_ok = mfi_f <= float(getattr(cfg, "mfi_long_threshold", 20.0))
                            else:
                                mfi_ok = mfi_f >= float(getattr(cfg, "mfi_short_threshold", 80.0))

                    if should_dca and (not cci_ok):
                        run_stats["dca_blocked_indicator"] += 1
                    if should_dca and (not mfi_ok):
                        run_stats["dca_blocked_mfi"] += 1

                    if bool(getattr(cfg, "macd_hist_flip_enabled", False)) and bool(macd_flip_to_side) and (not prev_tranche_ok):
                        run_stats["dca_blocked_macd"] += 1
                        if need_prev_tranche_entry:
                            run_stats["dca_blocked_macd_prev_tranche"] += 1

                    if should_dca and cci_ok and mfi_ok:
                        qty = q_notional / price if price > 0 else 0.0
                        if qty > 0:
                            fee = q_notional * fee_rate
                            wallet_free2 = state.wallet - state.margin_invested
                            required_free = q_margin + fee
                            if wallet_free2 < required_free:
                                run_stats["dca_blocked_margin"] += 1
                                qty = 0.0

                        if qty > 0:
                            new_size = c.size + qty
                            new_avg = (c.size * c.avg_open + q_notional) / new_size
                            c.size = new_size
                            c.avg_open = new_avg
                            state.wallet -= fee
                            state.margin_invested += q_margin

                            trades.append(
                                {
                                    "timestamp": ts,
                                    "price": price,
                                    "macd_hist": (None if pd.isna(macd_hist_val) else float(macd_hist_val)),
                                    "macd_hist_sign": int(hist_sign),
                                    "macd_tranche_sign": int(current_tranche_sign),
                                    "macd_prev_tranche_sign": int(last_tranche_sign),
                                    "macd_flip_to_side": bool(macd_flip_to_side),
                                    "macd_prev_tranche_ok": (None if prev_tranche_ok_dbg is None else bool(prev_tranche_ok_dbg)),
                                    "side": side_label,
                                    "type": "DCA",
                                    "cycle_id": int(c.cycle_id),
                                    "qty": float(qty),
                                    "qty_usdt": float(q_notional),
                                    "pnl_realized": 0.0,
                                    "fee": float(fee),
                                    "margin_cost": float(q_margin),
                                }
                            )
                            run_stats["dca_executed"] += 1

                            c.current_d_index += 1
                            d = d_start + (c.current_d_index - 1) * d_step
                            if is_long:
                                c.next_target_price = compute_p_new(c.size, c.avg_open, q_notional, d)
                            else:
                                c.next_target_price = compute_p_new_short(c.size, c.avg_open, q_notional, d)

            _sync_global_position_from_cycles(state)

        pnl_unrealized = 0.0
        if state.position_size > 0 and state.avg_price > 0:
            if is_long:
                pnl_unrealized = (price - state.avg_price) * state.position_size
            else:
                pnl_unrealized = (state.avg_price - price) * state.position_size
        equity = state.wallet + pnl_unrealized
        wallet_free = state.wallet - state.margin_invested

        equity_rows.append(
            {
                "timestamp": ts,
                "price": price,
                "macd_hist": (None if pd.isna(macd_hist_val) else float(macd_hist_val)),
                "macd_hist_sign": int(hist_sign),
                "macd_tranche_sign": int(current_tranche_sign),
                "macd_prev_tranche_sign": int(last_tranche_sign),
                "macd_flip_to_side": bool(macd_flip_to_side),
                "macd_prev_tranche_ok": (None if prev_tranche_ok_dbg is None else bool(prev_tranche_ok_dbg)),
                "wallet": state.wallet,
                "wallet_free": wallet_free,
                "margin_invested": state.margin_invested,
                "equity": equity,
                "position_size": state.position_size,
                "avg_price": state.avg_price,
                "pnl_unrealized": pnl_unrealized,
                "portions_used": state.portions_used,
                "cycles_completed": state.cycles_completed,
                "current_d_index": state.current_d_index,
                "next_target_price": state.next_target_price,
                "is_liquidated": bool(state.is_liquidated),
                "liquidation_reason": state.liquidation_reason if state.is_liquidated else "",
                "tp_phase": state.tp_phase,
                "tp_bucket_size": len(state.tp_bucket),
                "tp_active": bool(state.tp_active is not None and state.tp_active.size > 0),
            }
        )

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trades)
    return {"equity": equity_df, "trades": trades_df, "stats": run_stats}
