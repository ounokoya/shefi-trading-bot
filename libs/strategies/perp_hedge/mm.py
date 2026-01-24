from dataclasses import dataclass, field
from typing import List, Optional, Dict
from .models import AccountState, SignalIntent, MMAction, PositionSide, ActionType

@dataclass
class MoneyManagerConfig:
    mode: str = "gap_balance"
    capital_usdt: float = 1000.0
    initial_long_pct: float = 0.0
    initial_short_pct: float = 0.0
    initial_long_usdt: float = 100.0
    initial_short_usdt: float = 100.0
    max_initial_invest_pct: float = 0.0
    max_initial_invest_usdt: float = 400.0
    gap_mode: str = "leveq"
    max_notional_gap_pct: float = 1.0
    min_leg_notional_usdt: float = 0.0
    near_liquidation_risk_ratio: float = 0.85
    gap_threshold_pct: float = 0.03
    min_step_usdt: float = 25.0
    max_step_usdt: float = 150.0
    cooldown_bars: int = 3
    leverage: float = 10.0
    pnl_gate_after_increase_pct: float = 0.0
    pnl_gate_after_decrease_pct: float = 0.0

class MoneyManager:
    def __init__(self, config: MoneyManagerConfig):
        self.config = config
        self.last_action_bar: Dict[PositionSide, int] = {
            PositionSide.LONG: -9999,
            PositionSide.SHORT: -9999
        }

        self._total_invest_ref_usdt: Dict[PositionSide, float] = {
            PositionSide.LONG: 0.0,
            PositionSide.SHORT: 0.0,
        }

        self._pnl_gate_ref_invested: Dict[PositionSide, float] = {
            PositionSide.LONG: 0.0,
            PositionSide.SHORT: 0.0,
        }
        self._pnl_gate_required_pct: Dict[PositionSide, float] = {
            PositionSide.LONG: 0.0,
            PositionSide.SHORT: 0.0,
        }

    def _pnl_gate_allows_side(self, side: PositionSide, state: AccountState) -> bool:
        req_pct = float(self._pnl_gate_required_pct.get(side, 0.0) or 0.0)
        if req_pct <= 0.0:
            return True

        ref_inv = float(self._pnl_gate_ref_invested.get(side, 0.0) or 0.0)
        if ref_inv <= 0.0:
            self._pnl_gate_required_pct[side] = 0.0
            self._pnl_gate_ref_invested[side] = 0.0
            return True

        pnl_side = float(state.pnl_unrealized_long) if side == PositionSide.LONG else float(state.pnl_unrealized_short)
        if pnl_side >= (ref_inv * req_pct):
            self._pnl_gate_required_pct[side] = 0.0
            self._pnl_gate_ref_invested[side] = 0.0
            return True

        return False

    def _update_pnl_gate_after_action(self, side: PositionSide, action_type: ActionType, qty_usdt: float, state: AccountState) -> None:
        qty_usdt = max(0.0, float(qty_usdt))
        if qty_usdt <= 0.0:
            return

        if action_type == ActionType.INCREASE:
            req_pct = float(getattr(self.config, "pnl_gate_after_increase_pct", 0.0) or 0.0)
        else:
            req_pct = float(getattr(self.config, "pnl_gate_after_decrease_pct", 0.0) or 0.0)

        if req_pct <= 0.0:
            return

        leverage = max(1e-12, float(getattr(self.config, "leverage", 10.0) or 10.0))

        if side == PositionSide.LONG:
            inv = float(state.long_invested)
            notional = max(0.0, float(state.notional_long))
        else:
            inv = float(state.short_invested)
            notional = max(0.0, float(state.notional_short))

        if action_type == ActionType.INCREASE:
            inv_new = max(0.0, inv + (qty_usdt / leverage))
        else:
            if notional <= 0.0:
                inv_new = 0.0
            else:
                frac = min(1.0, qty_usdt / notional)
                inv_new = max(0.0, inv * (1.0 - frac))

        self._pnl_gate_ref_invested[side] = float(inv_new)
        self._pnl_gate_required_pct[side] = float(req_pct)

    def compute_actions(self, timestamp: int, bar_index: int, state: AccountState, intentions: List[SignalIntent]) -> List[MMAction]:
        actions = []

        pnl_total = float(state.pnl_unrealized_long) + float(state.pnl_unrealized_short)
        loss_unrealized = max(0.0, -pnl_total)
        margin_total = max(0.0, float(state.margin_total))
        risk_ratio = (loss_unrealized / margin_total) if margin_total > 0 else 0.0
        near_liquidation_risk_ratio = float(getattr(self.config, "near_liquidation_risk_ratio", 1.0))

        notional_long = max(0.0, float(state.notional_long))
        notional_short = max(0.0, float(state.notional_short))
        notional_total = notional_long + notional_short
        notional_diff = notional_long - notional_short
        notional_gap_pct = (abs(notional_diff) / notional_total) if notional_total > 0 else 0.0
        max_notional_gap_pct = float(getattr(self.config, "max_notional_gap_pct", 1.0))

        def _improves_notional_balance(side: PositionSide, action_type: ActionType, qty_usdt: float) -> bool:
            qty_usdt = max(0.0, float(qty_usdt))
            if qty_usdt <= 0.0:
                return False

            diff = notional_diff
            if action_type == ActionType.INCREASE:
                diff_new = diff + qty_usdt if side == PositionSide.LONG else diff - qty_usdt
            else:
                diff_new = diff - qty_usdt if side == PositionSide.LONG else diff + qty_usdt

            return abs(diff_new) < abs(diff)

        def _notional_gap_pct_for(n_long: float, n_short: float) -> float:
            n_long = max(0.0, float(n_long))
            n_short = max(0.0, float(n_short))
            total = n_long + n_short
            return (abs(n_long - n_short) / total) if total > 0 else 0.0

        def _apply_notional_action(n_long: float, n_short: float, side: PositionSide, action_type: ActionType, qty_usdt: float) -> tuple[float, float]:
            qty_usdt = max(0.0, float(qty_usdt))
            if qty_usdt <= 0.0:
                return n_long, n_short

            if action_type == ActionType.INCREASE:
                if side == PositionSide.LONG:
                    return n_long + qty_usdt, n_short
                return n_long, n_short + qty_usdt

            if side == PositionSide.LONG:
                return max(0.0, n_long - qty_usdt), n_short
            return n_long, max(0.0, n_short - qty_usdt)

        def _passes_notional_safety(actions_to_apply: List[tuple[PositionSide, ActionType, float]]) -> bool:
            if max_notional_gap_pct >= 1.0:
                return True

            n_long = notional_long
            n_short = notional_short
            for side, action_type, qty in actions_to_apply:
                n_long, n_short = _apply_notional_action(n_long, n_short, side, action_type, qty)

            new_gap_pct = _notional_gap_pct_for(n_long, n_short)
            eps = 1e-12

            # If we're below the cap, do not allow an action set that would cross above it.
            if notional_gap_pct <= max_notional_gap_pct + eps:
                return new_gap_pct <= max_notional_gap_pct + eps

            # If we're already above the cap (legacy state), only allow actions that improve balance.
            return new_gap_pct < notional_gap_pct - eps

        can_act_long_any = (bar_index - self.last_action_bar[PositionSide.LONG]) > self.config.cooldown_bars
        can_act_short_any = (bar_index - self.last_action_bar[PositionSide.SHORT]) > self.config.cooldown_bars

        can_act_long = can_act_long_any and self._pnl_gate_allows_side(PositionSide.LONG, state)
        can_act_short = can_act_short_any and self._pnl_gate_allows_side(PositionSide.SHORT, state)

        can_decrease_long_intent = can_act_long and (SignalIntent.LONG_CAN_DECREASE in intentions)
        can_decrease_short_intent = can_act_short and (SignalIntent.SHORT_CAN_DECREASE in intentions)

        if risk_ratio >= near_liquidation_risk_ratio:
            min_step = max(0.0, float(self.config.min_step_usdt))
            max_step = float(self.config.max_step_usdt)
            min_leg_notional_usdt = max(0.0, float(getattr(self.config, "min_leg_notional_usdt", 0.0)))

            def _simulate_risk_ratio_after_decrease(side: PositionSide, qty_usdt: float) -> float:
                qty_usdt = max(0.0, float(qty_usdt))
                if qty_usdt <= 0.0:
                    return risk_ratio

                price = float(state.current_price)
                if price <= 0:
                    return risk_ratio

                if side == PositionSide.LONG:
                    old_size = float(state.long_size)
                    entry = float(state.long_entry_price)
                    pnl_side = float(state.pnl_unrealized_long)
                else:
                    old_size = float(state.short_size)
                    entry = float(state.short_entry_price)
                    pnl_side = float(state.pnl_unrealized_short)

                if old_size <= 0:
                    return risk_ratio

                qty_change = qty_usdt / price
                fraction = min(1.0, qty_change / old_size) if old_size > 0 else 0.0
                if fraction <= 0.0:
                    return risk_ratio

                close_qty = old_size * fraction
                if side == PositionSide.LONG:
                    realized_pnl = (price - entry) * close_qty
                    pnl_long_new = pnl_side * (1.0 - fraction)
                    pnl_short_new = float(state.pnl_unrealized_short)
                else:
                    realized_pnl = (entry - price) * close_qty
                    pnl_short_new = pnl_side * (1.0 - fraction)
                    pnl_long_new = float(state.pnl_unrealized_long)

                wallet_new = float(state.wallet_balance) + realized_pnl
                pnl_total_new = pnl_long_new + pnl_short_new
                margin_total_new = wallet_new + max(0.0, pnl_total_new)
                loss_unrealized_new = max(0.0, -pnl_total_new)
                if margin_total_new <= 0:
                    return 1.0
                return loss_unrealized_new / margin_total_new

            candidate_sides: List[PositionSide] = []
            if can_decrease_long_intent:
                candidate_sides.append(PositionSide.LONG)
            if can_decrease_short_intent:
                candidate_sides.append(PositionSide.SHORT)

            if not candidate_sides:
                if can_act_long_any:
                    candidate_sides.append(PositionSide.LONG)
                if can_act_short_any:
                    candidate_sides.append(PositionSide.SHORT)

            best_side: Optional[PositionSide] = None
            best_qty: float = 0.0
            best_risk: float = risk_ratio

            for side in candidate_sides:
                side_notional = notional_long if side == PositionSide.LONG else notional_short
                remainable = max(0.0, float(side_notional) - min_leg_notional_usdt)
                if remainable < min_step:
                    continue

                qty = min(remainable, max_step) if max_step > 0 else remainable
                if qty < min_step:
                    continue

                if not _passes_notional_safety([(side, ActionType.DECREASE, qty)]):
                    continue

                rr = _simulate_risk_ratio_after_decrease(side, qty)
                if rr < best_risk - 1e-12:
                    best_risk = rr
                    best_side = side
                    best_qty = qty

            if best_side is not None and best_qty >= min_step:
                actions.append(MMAction(
                    side=best_side,
                    action_type=ActionType.DECREASE,
                    qty_usdt=best_qty,
                    reason=f"Near liquidation: risk={risk_ratio:.2%} >= {near_liquidation_risk_ratio:.2%}, Decreasing to reduce risk"
                ))
                self.last_action_bar[best_side] = bar_index
                self._update_pnl_gate_after_action(best_side, ActionType.DECREASE, best_qty, state)
                return actions

        mm_mode = (getattr(self.config, 'mode', 'gap_balance') or 'gap_balance').lower().strip()
        if mm_mode == 'event_target':
            cap = max(0.0, float(getattr(self.config, 'capital_usdt', 0.0) or 0.0))
            leverage = max(1e-12, float(getattr(self.config, 'leverage', 10.0) or 10.0))
            target_pct = max(0.0, float(getattr(self.config, 'max_initial_invest_pct', 0.0) or 0.0))
            if target_pct > 0.0:
                max_target_notional = cap * target_pct * leverage
            else:
                max_target_notional = max(0.0, float(getattr(self.config, 'max_initial_invest_usdt', 0.0) or 0.0)) * leverage

            profit_threshold_pct = max(0.0, float(getattr(self.config, 'gap_threshold_pct', 0.0) or 0.0))
            min_leg_notional_usdt = max(0.0, float(getattr(self.config, 'min_leg_notional_usdt', 0.0) or 0.0))

            can_act_long = ((bar_index - self.last_action_bar[PositionSide.LONG]) > int(getattr(self.config, 'cooldown_bars', 0) or 0))
            can_act_short = ((bar_index - self.last_action_bar[PositionSide.SHORT]) > int(getattr(self.config, 'cooldown_bars', 0) or 0))

            force_close_long = (SignalIntent.LONG_FORCE_CLOSE in intentions)
            force_close_short = (SignalIntent.SHORT_FORCE_CLOSE in intentions)

            want_htf_add_long = (SignalIntent.LONG_HTF_ADD in intentions)
            want_htf_add_short = (SignalIntent.SHORT_HTF_ADD in intentions)
            want_ltf_dip_long = (SignalIntent.LONG_LTF_DIP in intentions)
            want_ltf_dip_short = (SignalIntent.SHORT_LTF_DIP in intentions)

            want_tp_long = (SignalIntent.LONG_CAN_DECREASE in intentions)
            want_tp_short = (SignalIntent.SHORT_CAN_DECREASE in intentions)

            def _pnl_side(side: PositionSide) -> float:
                return float(state.pnl_unrealized_long) if side == PositionSide.LONG else float(state.pnl_unrealized_short)

            def _notional_side(side: PositionSide) -> float:
                return notional_long if side == PositionSide.LONG else notional_short

            def _set_ref(side: PositionSide, ref: float) -> None:
                self._total_invest_ref_usdt[side] = max(0.0, float(ref))

            # HTF cross sets the reference target for the locked side.
            if want_htf_add_long:
                _set_ref(PositionSide.LONG, max_target_notional)
            if want_htf_add_short:
                _set_ref(PositionSide.SHORT, max_target_notional)

            def _try_force_close(side: PositionSide) -> Optional[MMAction]:
                if side == PositionSide.LONG:
                    if not (can_act_long and force_close_long):
                        return None
                else:
                    if not (can_act_short and force_close_short):
                        return None

                n = _notional_side(side)
                if n <= 0.0:
                    _set_ref(side, 0.0)
                    return None

                if not _passes_notional_safety([(side, ActionType.DECREASE, n)]):
                    return None

                _set_ref(side, 0.0)
                return MMAction(side=side, action_type=ActionType.DECREASE, qty_usdt=float(n), reason=f'EventTarget: force close {side.name.lower()}')

            def _try_tp(side: PositionSide) -> Optional[MMAction]:
                if side == PositionSide.LONG:
                    if not (can_act_long and want_tp_long):
                        return None
                else:
                    if not (can_act_short and want_tp_short):
                        return None

                pnl = _pnl_side(side)
                if pnl <= 0.0:
                    return None

                ref = max(0.0, float(self._total_invest_ref_usdt.get(side, 0.0) or 0.0))
                if ref <= 0.0:
                    ref = _notional_side(side)
                if ref <= 0.0:
                    return None

                if pnl < (ref * profit_threshold_pct):
                    return None

                n = _notional_side(side)
                remainable = max(0.0, float(n) - min_leg_notional_usdt)
                if remainable <= 0.0:
                    return None

                qty = min(float(pnl), float(remainable))
                if qty <= 0.0:
                    return None

                if not _passes_notional_safety([(side, ActionType.DECREASE, qty)]):
                    return None

                # Update reference to expected remaining notional after the partial TP.
                _set_ref(side, max(0.0, n - qty))
                return MMAction(side=side, action_type=ActionType.DECREASE, qty_usdt=float(qty), reason=f'EventTarget: tp {side.name.lower()} pnl={pnl:.2f} ref={ref:.2f}')

            def _try_ltf_dip(side: PositionSide) -> Optional[MMAction]:
                if risk_ratio >= near_liquidation_risk_ratio:
                    return None

                if side == PositionSide.LONG:
                    if not (can_act_long and want_ltf_dip_long):
                        return None
                else:
                    if not (can_act_short and want_ltf_dip_short):
                        return None

                pnl = _pnl_side(side)
                if pnl >= 0.0:
                    return None

                ref = max(0.0, float(self._total_invest_ref_usdt.get(side, 0.0) or 0.0))
                if ref <= 0.0:
                    return None

                n = _notional_side(side)
                missing = max(0.0, ref - n)
                if missing <= 0.0:
                    return None

                qty = min(abs(float(pnl)), missing)
                if qty <= 0.0:
                    return None

                if not _passes_notional_safety([(side, ActionType.INCREASE, qty)]):
                    return None

                return MMAction(side=side, action_type=ActionType.INCREASE, qty_usdt=float(qty), reason=f'EventTarget: dip {side.name.lower()} pnl={pnl:.2f} ref={ref:.2f}')

            def _try_htf_fill(side: PositionSide) -> Optional[MMAction]:
                if risk_ratio >= near_liquidation_risk_ratio:
                    return None

                if side == PositionSide.LONG:
                    if not (can_act_long and want_htf_add_long):
                        return None
                else:
                    if not (can_act_short and want_htf_add_short):
                        return None

                ref = max_target_notional
                if ref <= 0.0:
                    return None

                n = _notional_side(side)
                missing = max(0.0, ref - n)
                if missing <= 0.0:
                    return None

                if not _passes_notional_safety([(side, ActionType.INCREASE, missing)]):
                    return None

                return MMAction(side=side, action_type=ActionType.INCREASE, qty_usdt=float(missing), reason=f'EventTarget: htf fill {side.name.lower()} ref={ref:.2f}')

            # Priority: force close, then TP, then dip, then HTF fill.
            cand: List[MMAction] = []
            for s in (PositionSide.LONG, PositionSide.SHORT):
                a = _try_force_close(s)
                if a is not None:
                    cand.append(a)
            for s in (PositionSide.LONG, PositionSide.SHORT):
                a = _try_tp(s)
                if a is not None:
                    cand.append(a)
            for s in (PositionSide.LONG, PositionSide.SHORT):
                a = _try_ltf_dip(s)
                if a is not None:
                    cand.append(a)
            for s in (PositionSide.LONG, PositionSide.SHORT):
                a = _try_htf_fill(s)
                if a is not None:
                    cand.append(a)

            if not cand:
                return []

            for a in cand:
                actions.append(a)
                self.last_action_bar[a.side] = bar_index
            return actions
        if mm_mode == 'signal_direct':
            min_step = max(0.0, float(self.config.min_step_usdt))
            max_step = float(self.config.max_step_usdt)
            min_leg_notional_usdt = max(0.0, float(getattr(self.config, "min_leg_notional_usdt", 0.0)))
            leverage = max(1e-12, float(getattr(self.config, 'leverage', 10.0) or 10.0))

            max_margin = float(getattr(self.config, 'max_initial_invest_usdt', 0.0) or 0.0)
            room_margin = max(0.0, max_margin - float(state.margin_invested))
            room_notional = room_margin * leverage

            force_close_long = (SignalIntent.LONG_FORCE_CLOSE in intentions)
            force_close_short = (SignalIntent.SHORT_FORCE_CLOSE in intentions)

            def _try_decrease(side: PositionSide) -> Optional[MMAction]:
                if side == PositionSide.LONG:
                    if force_close_long and notional_long > 0.0:
                        qty = float(notional_long)
                        return MMAction(
                            side=side,
                            action_type=ActionType.DECREASE,
                            qty_usdt=qty,
                            reason='SignalDirect: force close long'
                        )
                    if not (can_act_long and (SignalIntent.LONG_CAN_DECREASE in intentions)):
                        return None
                    side_notional = notional_long
                else:
                    if force_close_short and notional_short > 0.0:
                        qty = float(notional_short)
                        return MMAction(
                            side=side,
                            action_type=ActionType.DECREASE,
                            qty_usdt=qty,
                            reason='SignalDirect: force close short'
                        )
                    if not (can_act_short and (SignalIntent.SHORT_CAN_DECREASE in intentions)):
                        return None
                    side_notional = notional_short

                remainable = max(0.0, float(side_notional) - min_leg_notional_usdt)
                if remainable < min_step:
                    return None

                qty = remainable
                if max_step > 0:
                    qty = min(qty, max_step)
                if qty < min_step:
                    return None

                if not _passes_notional_safety([(side, ActionType.DECREASE, qty)]):
                    return None

                return MMAction(
                    side=side,
                    action_type=ActionType.DECREASE,
                    qty_usdt=float(qty),
                    reason='SignalDirect: decrease allowed by brain'
                )

            def _try_increase(side: PositionSide) -> Optional[MMAction]:
                if risk_ratio >= near_liquidation_risk_ratio:
                    return None

                if room_notional < min_step:
                    return None

                if side == PositionSide.LONG:
                    if not (can_act_long and (SignalIntent.LONG_CAN_INCREASE in intentions)):
                        return None
                else:
                    if not (can_act_short and (SignalIntent.SHORT_CAN_INCREASE in intentions)):
                        return None

                qty = room_notional
                if max_step > 0:
                    qty = min(qty, max_step)
                if qty < min_step:
                    return None

                if not _passes_notional_safety([(side, ActionType.INCREASE, qty)]):
                    return None

                return MMAction(
                    side=side,
                    action_type=ActionType.INCREASE,
                    qty_usdt=float(qty),
                    reason='SignalDirect: increase allowed by brain'
                )

            # Priority: DECREASE then INCREASE (per your TP-first preference)
            cand: List[MMAction] = []
            for s in (PositionSide.LONG, PositionSide.SHORT):
                a = _try_decrease(s)
                if a is not None:
                    cand.append(a)
            for s in (PositionSide.LONG, PositionSide.SHORT):
                a = _try_increase(s)
                if a is not None:
                    cand.append(a)

            if not cand:
                return []

            # Apply and update trackers
            for a in cand:
                actions.append(a)
                self.last_action_bar[a.side] = bar_index
                self._update_pnl_gate_after_action(a.side, a.action_type, a.qty_usdt, state)

            return actions
        
        gap_mode = (getattr(self.config, "gap_mode", "notional") or "notional").lower().strip()
        if gap_mode == "leveq":
            equity_long = state.long_invested + state.pnl_unrealized_long
            equity_short = state.short_invested + state.pnl_unrealized_short

            val_long = abs(float(equity_long)) * self.config.leverage
            val_short = abs(float(equity_short)) * self.config.leverage
            gap_mode_label = "LevEq"
        else:
            val_long = max(0.0, float(state.notional_long))
            val_short = max(0.0, float(state.notional_short))
            gap_mode_label = "Notional"
        
        total_val = val_long + val_short
        
        if total_val == 0:
            return [] 

        gap_usdt = abs(val_long - val_short)
        gap_pct = gap_usdt / total_val
        
        # 2. Check Threshold
        if gap_pct <= self.config.gap_threshold_pct:
            return [] # No rebalancing needed
            
        # 3. Determine Step Size (Notional USDT)
        min_step = max(0.0, float(self.config.min_step_usdt))
        max_step = float(self.config.max_step_usdt)
        if gap_usdt < min_step:
            return []

        step_usdt = gap_usdt
        if max_step > 0:
            step_usdt = min(step_usdt, max_step)
        
        # 4. Determine Direction based on Leveraged Equity Value
        long_is_weak = val_long < val_short
        weak_side = PositionSide.LONG if long_is_weak else PositionSide.SHORT
        strong_side = PositionSide.SHORT if long_is_weak else PositionSide.LONG
        
        # Check cooldowns
        can_act_weak = ((bar_index - self.last_action_bar[weak_side]) > self.config.cooldown_bars) and self._pnl_gate_allows_side(weak_side, state)
        can_act_strong = ((bar_index - self.last_action_bar[strong_side]) > self.config.cooldown_bars) and self._pnl_gate_allows_side(strong_side, state)
        
        # Check Intents
        can_increase_weak = False
        if weak_side == PositionSide.LONG and SignalIntent.LONG_CAN_INCREASE in intentions:
            can_increase_weak = True
        elif weak_side == PositionSide.SHORT and SignalIntent.SHORT_CAN_INCREASE in intentions:
            can_increase_weak = True
            
        can_decrease_strong = False
        if strong_side == PositionSide.LONG and SignalIntent.LONG_CAN_DECREASE in intentions:
            can_decrease_strong = True
        elif strong_side == PositionSide.SHORT and SignalIntent.SHORT_CAN_DECREASE in intentions:
            can_decrease_strong = True

        if risk_ratio >= near_liquidation_risk_ratio:
            can_increase_weak = False

        # Check Caps for increasing
        # max_initial_invest_usdt is interpreted as MAX USED MARGIN (Cost Basis) allowed.
        # step_margin = step_notional / leverage
        step_margin = step_usdt / self.config.leverage
        has_capital_room_full = (state.margin_invested + step_margin) <= self.config.max_initial_invest_usdt
        
        action_generated = None
        
        # Priority 1: Increase Weak Side
        strong_notional = val_long if strong_side == PositionSide.LONG else val_short
        strong_actual_notional = state.notional_long if strong_side == PositionSide.LONG else state.notional_short

        min_leg_notional_usdt = max(0.0, float(getattr(self.config, "min_leg_notional_usdt", 0.0)))
        strong_remainable = max(0.0, float(strong_actual_notional) - min_leg_notional_usdt)

        can_do_increase_full = can_act_weak and can_increase_weak and has_capital_room_full
        can_do_decrease = can_act_strong and can_decrease_strong and (strong_remainable >= min_step)

        if can_do_decrease and can_act_weak and can_increase_weak:
            split_step = step_usdt / 2.0
            if split_step >= min_step:
                inc_margin = split_step / self.config.leverage
                has_capital_room_split = (state.margin_invested + inc_margin) <= self.config.max_initial_invest_usdt
                if not has_capital_room_split:
                    split_step = 0.0

            if split_step >= min_step:
                dec_qty = min(split_step, strong_remainable)
                inc_qty = split_step

                if not _passes_notional_safety([(strong_side, ActionType.DECREASE, dec_qty), (weak_side, ActionType.INCREASE, inc_qty)]):
                    if _passes_notional_safety([(strong_side, ActionType.DECREASE, dec_qty)]):
                        inc_qty = 0.0
                    elif _passes_notional_safety([(weak_side, ActionType.INCREASE, inc_qty)]):
                        dec_qty = 0.0
                    else:
                        dec_qty = 0.0
                        inc_qty = 0.0

                if max_notional_gap_pct < 1.0 and notional_gap_pct > max_notional_gap_pct:
                    if not _improves_notional_balance(strong_side, ActionType.DECREASE, dec_qty):
                        dec_qty = 0.0
                    if not _improves_notional_balance(weak_side, ActionType.INCREASE, inc_qty):
                        inc_qty = 0.0

                if dec_qty < min_step and inc_qty < min_step:
                    return []

                if dec_qty >= min_step:
                    actions.append(MMAction(
                        side=strong_side,
                        action_type=ActionType.DECREASE,
                        qty_usdt=dec_qty,
                        reason=f"Gap {gap_pct:.2%} ({gap_mode_label}) > {self.config.gap_threshold_pct:.2%}, Decreasing Strong"
                    ))

                if inc_qty >= min_step:
                    actions.append(MMAction(
                        side=weak_side,
                        action_type=ActionType.INCREASE,
                        qty_usdt=inc_qty,
                        reason=f"Gap {gap_pct:.2%} ({gap_mode_label}) > {self.config.gap_threshold_pct:.2%}, Increasing Weak (risk={risk_ratio:.2%})"
                    ))

                if actions:
                    self.last_action_bar[strong_side] = bar_index
                    self.last_action_bar[weak_side] = bar_index

                    for a in actions:
                        self._update_pnl_gate_after_action(a.side, a.action_type, a.qty_usdt, state)
                    return actions

        if can_do_decrease:
            dec_qty = min(step_usdt, strong_remainable)
            if max_notional_gap_pct < 1.0 and notional_gap_pct > max_notional_gap_pct:
                if not _improves_notional_balance(strong_side, ActionType.DECREASE, dec_qty):
                    dec_qty = 0.0

            if dec_qty > 0.0 and not _passes_notional_safety([(strong_side, ActionType.DECREASE, dec_qty)]):
                dec_qty = 0.0

            if dec_qty < min_step:
                dec_qty = 0.0

            if dec_qty > 0.0:
                action_generated = MMAction(
                    side=strong_side,
                    action_type=ActionType.DECREASE,
                    qty_usdt=dec_qty,
                    reason=f"Gap {gap_pct:.2%} ({gap_mode_label}) > {self.config.gap_threshold_pct:.2%}, Decreasing Strong"
                )
                self.last_action_bar[strong_side] = bar_index

        elif can_do_increase_full:
            inc_qty = step_usdt
            if max_notional_gap_pct < 1.0 and notional_gap_pct > max_notional_gap_pct:
                if not _improves_notional_balance(weak_side, ActionType.INCREASE, inc_qty):
                    inc_qty = 0.0

            if inc_qty > 0.0 and not _passes_notional_safety([(weak_side, ActionType.INCREASE, inc_qty)]):
                inc_qty = 0.0

            if inc_qty >= min_step:
                action_generated = MMAction(
                    side=weak_side,
                    action_type=ActionType.INCREASE,
                    qty_usdt=inc_qty,
                    reason=f"Gap {gap_pct:.2%} ({gap_mode_label}) > {self.config.gap_threshold_pct:.2%}, Increasing Weak (risk={risk_ratio:.2%})"
                )
                self.last_action_bar[weak_side] = bar_index

        if action_generated:
            actions.append(action_generated)
            self._update_pnl_gate_after_action(action_generated.side, action_generated.action_type, action_generated.qty_usdt, state)

        return actions
