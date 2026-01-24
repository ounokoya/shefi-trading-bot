import pandas as pd
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from .models import AccountState, PositionSide, ActionType, Position, MMAction
from .brain import BaseBrain
from .mm import MoneyManager, MoneyManagerConfig

@dataclass
class BacktestConfig:
    symbol: str
    start_date: str
    end_date: str
    timeframe: str = "5m"
    leverage: float = 10.0
    fee_rate: float = 0.0015
    liquidation_threshold_pct: float = 0.05
    
    # MM Config
    mm_config: MoneyManagerConfig = field(default_factory=MoneyManagerConfig)

class BacktestEngine:
    def __init__(self, config: BacktestConfig, brain: BaseBrain, df: pd.DataFrame):
        self.config = config
        self.brain = brain
        self.df = df.reset_index(drop=True)
        self.mm = MoneyManager(config.mm_config)
        
        # Init State
        # Initial positions are opened at the START of the backtest logic?
        # Spec says: "initial_long_usdt", "initial_short_usdt".
        # We will initialize them on the first tick.
        
        self.wallet_balance = config.mm_config.capital_usdt
        self.positions = {
            PositionSide.LONG: Position(PositionSide.LONG, 0.0, 0.0, 0.0),
            PositionSide.SHORT: Position(PositionSide.SHORT, 0.0, 0.0, 0.0)
        }
        
        # History
        self.equity_history = []
        self.trades_history = []
        self.margin_history = []
        
        self.is_liquidated = False
        self.liquidation_reason = ""

        self._last_intentions = []
        self._last_actions = []
        
        # Precompute initial positions flag
        self.initialized_positions = False

    def run(self):
        logging.info("Starting Backtest...")
        
        for i, row in self.df.iterrows():
            if self.is_liquidated:
                break
                
            timestamp = int(row['open_time']) # Assumes standard column names
            price = float(row['close'])
            
            # 1. Update State (PnL, Liquidations)
            self._update_state(price)
            if self.is_liquidated:
                self._last_intentions = []
                self._last_actions = []
                self._log_step(timestamp, price)
                break
                
            # 2. Initialize positions if needed
            if not self.initialized_positions:
                self._open_initial_positions(price)
                self.initialized_positions = True
                self._update_state(price) # Re-update to reflect spread/fees if any (ignored here for simplicity)
            
            # 3. Get Brain Intentions
            current_state = self._get_snapshot_state(price)
            intentions = self.brain.get_intentions(timestamp, current_state)
            self._last_intentions = intentions
            
            # 4. Get MM Actions
            actions = self.mm.compute_actions(timestamp, i, current_state, intentions)
            self._last_actions = actions
            
            # 5. Execute Actions
            for action in actions:
                self._execute_action(action, price, timestamp)
                
            # 6. Log Step
            self._log_step(timestamp, price)
            
        logging.info(f"Backtest Finished. Liquidated: {self.is_liquidated}")

    def _update_state(self, price: float):
        # Calculate PnL Unrealized
        # Long: (current - entry) * size
        # Short: (entry - current) * size
        
        long_pos = self.positions[PositionSide.LONG]
        short_pos = self.positions[PositionSide.SHORT]
        
        pnl_long = (price - long_pos.entry_price) * long_pos.size if long_pos.size > 0 else 0
        pnl_short = (short_pos.entry_price - price) * short_pos.size if short_pos.size > 0 else 0
        
        total_pnl = pnl_long + pnl_short
        
        # Check Liquidation
        # margin_total = wallet_balance + max(0, pnl_total) -- Wait, spec says max(0, pnl_unrealized) is usually total positive pnl?
        # Spec: "margin_total = wallet_balance + max(0, pnl_non_realise)" where pnl_non_realise is NET.
        # Spec: "loss_unrealized = max(0, -pnl_non_realise)"
        
        margin_total = self.wallet_balance + max(0.0, total_pnl)
        loss_unrealized = max(0.0, -total_pnl)
        
        # Threshold: liquidate if loss >= (1 - threshold) * margin_total
        limit = (1.0 - self.config.liquidation_threshold_pct) * margin_total
        
        if loss_unrealized >= limit:
            self.is_liquidated = True
            
            # Detailed Liquidation Log
            invested_long = long_pos.invested_amount
            invested_short = short_pos.invested_amount
            total_invested = invested_long + invested_short
            wallet_free = self.wallet_balance # Wallet balance is technically "free" margin buffer + realized pnl - fees
            
            detail_msg = (
                f"\n!!! LIQUIDATION TRIGGERED !!!\n"
                f"Price: {price:.4f}\n"
                f"Unrealized Loss: {loss_unrealized:.2f} >= Limit: {limit:.2f} (Threshold: {self.config.liquidation_threshold_pct*100}%)\n"
                f"Total Margin (Wallet + Positive PnL): {margin_total:.2f}\n"
                f"--------------------------------------------------\n"
                f"Wallet Balance: {self.wallet_balance:.2f}\n"
                f"Total Invested Margin: {total_invested:.2f} (Long: {invested_long:.2f}, Short: {invested_short:.2f})\n"
                f"PnL Long: {pnl_long:.2f} (Size: {long_pos.size:.4f}, Entry: {long_pos.entry_price:.4f})\n"
                f"PnL Short: {pnl_short:.2f} (Size: {short_pos.size:.4f}, Entry: {short_pos.entry_price:.4f})\n"
                f"Net PnL Unrealized: {total_pnl:.2f}\n"
                f"--------------------------------------------------"
            )
            
            # Add Recent Trades History
            recent_trades = self.trades_history[-15:]
            trades_msg = "\n\nLast 15 Trades:\n" + "-"*50 + "\n"
            for t in recent_trades:
                ts_str = pd.to_datetime(t['timestamp'], unit='ms').strftime('%Y-%m-%d %H:%M')
                trades_msg += f"[{ts_str}] {t['side']} {t['type']} | Qty: {t['qty']:.4f} (${t['qty_usdt']:.1f}) @ {t['price']:.4f} | Reason: {t['reason']}\n"
            
            self.liquidation_reason = detail_msg + trades_msg
            logging.warning(self.liquidation_reason)

    def _get_snapshot_state(self, price: float) -> AccountState:
        long_pos = self.positions[PositionSide.LONG]
        short_pos = self.positions[PositionSide.SHORT]
        
        pnl_long = (price - long_pos.entry_price) * long_pos.size if long_pos.size > 0 else 0
        pnl_short = (short_pos.entry_price - price) * short_pos.size if short_pos.size > 0 else 0
        
        # Calculate Margin Invested from actual tracked amounts
        margin_invested = long_pos.invested_amount + short_pos.invested_amount
        
        return AccountState(
            wallet_balance=self.wallet_balance,
            margin_invested=margin_invested,
            long_size=long_pos.size,
            long_entry_price=long_pos.entry_price,
            long_invested=long_pos.invested_amount,
            short_size=short_pos.size,
            short_entry_price=short_pos.entry_price,
            short_invested=short_pos.invested_amount,
            current_price=price,
            pnl_unrealized_long=pnl_long,
            pnl_unrealized_short=pnl_short
        )

    def _open_initial_positions(self, price: float):
        cap = max(0.0, float(getattr(self.config.mm_config, 'capital_usdt', 0.0) or 0.0))
        init_long_pct = max(0.0, float(getattr(self.config.mm_config, 'initial_long_pct', 0.0) or 0.0))
        init_short_pct = max(0.0, float(getattr(self.config.mm_config, 'initial_short_pct', 0.0) or 0.0))

        init_long_usdt = 0.0
        if init_long_pct > 0.0:
            init_long_usdt = cap * init_long_pct * float(self.config.leverage)
        else:
            init_long_usdt = float(getattr(self.config.mm_config, 'initial_long_usdt', 0.0) or 0.0)

        init_short_usdt = 0.0
        if init_short_pct > 0.0:
            init_short_usdt = cap * init_short_pct * float(self.config.leverage)
        else:
            init_short_usdt = float(getattr(self.config.mm_config, 'initial_short_usdt', 0.0) or 0.0)

        # Initial Long
        if init_long_usdt > 0:
            qty = init_long_usdt / price
            cost = init_long_usdt / self.config.leverage
            fee = init_long_usdt * self.config.fee_rate
            
            self.positions[PositionSide.LONG] = Position(PositionSide.LONG, qty, price, cost)
            self.wallet_balance -= fee
            
        # Initial Short
        if init_short_usdt > 0:
            qty = init_short_usdt / price
            cost = init_short_usdt / self.config.leverage
            fee = init_short_usdt * self.config.fee_rate
            
            self.positions[PositionSide.SHORT] = Position(PositionSide.SHORT, qty, price, cost)
            self.wallet_balance -= fee
            
        logging.info(f"Initialized positions at {price}. Wallet: {self.wallet_balance:.2f}")

    def _execute_action(self, action: MMAction, price: float, timestamp: int):
        pos = self.positions[action.side]
        qty_change = action.qty_usdt / price
        # Cost of this specific action (margin added)
        action_cost = action.qty_usdt / self.config.leverage
        fee = action.qty_usdt * self.config.fee_rate
        
        # Record Trade
        trade_record = {
            'timestamp': timestamp,
            'price': price,
            'side': action.side.name,
            'type': action.action_type.name,
            'qty_usdt': action.qty_usdt,
            'qty': qty_change,
            'fee': fee,
            'reason': action.reason
        }
        
        if action.action_type == ActionType.INCREASE:
            # Weighted Average Price update
            new_size = pos.size + qty_change
            if new_size > 0:
                new_entry = ((pos.size * pos.entry_price) + (qty_change * price)) / new_size
                pos.entry_price = new_entry
                pos.size = new_size
                pos.invested_amount += action_cost # Add margin
            self.wallet_balance -= fee
            
        elif action.action_type == ActionType.DECREASE:
            # Realize PnL on portion
            if pos.size <= 0: return
            
            fraction = min(1.0, qty_change / pos.size)
            close_qty = pos.size * fraction
            
            # Reduce invested amount proportionally
            invested_removed = pos.invested_amount * fraction
            pos.invested_amount -= invested_removed
            
            # PnL Realized
            if action.side == PositionSide.LONG:
                pnl = (price - pos.entry_price) * close_qty
            else:
                pnl = (pos.entry_price - price) * close_qty
                
            self.wallet_balance += pnl
            self.wallet_balance -= fee
            
            pos.size -= close_qty
            # Entry price doesn't change on reduction
            
            trade_record['pnl_realized'] = pnl
            
        self.trades_history.append(trade_record)

    def _format_intentions(self, intentions) -> str:
        try:
            return "|".join([i.name for i in intentions])
        except Exception:
            return ""

    def _format_actions(self, actions) -> str:
        try:
            parts = []
            for a in actions:
                parts.append(f"{a.side.name}:{a.action_type.name}:{float(a.qty_usdt):.2f}")
            return "|".join(parts)
        except Exception:
            return ""

    def _get_brain_indicators(self, timestamp: int) -> Dict[str, float]:
        out: Dict[str, float] = {}

        try:
            if hasattr(self.brain, '_lock_dir_map'):
                v = getattr(self.brain, '_lock_dir_map', {}).get(timestamp, None)
                if v is not None:
                    out['brain_lock_dir'] = int(v)

            if hasattr(self.brain, '_htf_cross_dir_map'):
                v = getattr(self.brain, '_htf_cross_dir_map', {}).get(timestamp, None)
                if v is not None:
                    out['brain_htf_cross_dir'] = int(v)

            if hasattr(self.brain, '_ltf_cross_dir_map'):
                v = getattr(self.brain, '_ltf_cross_dir_map', {}).get(timestamp, None)
                if v is not None:
                    out['brain_ltf_cross_dir'] = int(v)

            if hasattr(self.brain, '_dip_done_map'):
                v = getattr(self.brain, '_dip_done_map', {}).get(timestamp, None)
                if v is not None:
                    out['brain_dip_done'] = int(bool(v))

            if hasattr(self.brain, '_long_extreme_entry_map'):
                v = getattr(self.brain, '_long_extreme_entry_map', {}).get(timestamp, None)
                if v is not None:
                    out['brain_long_extreme_entry'] = int(bool(v))

            if hasattr(self.brain, '_short_extreme_entry_map'):
                v = getattr(self.brain, '_short_extreme_entry_map', {}).get(timestamp, None)
                if v is not None:
                    out['brain_short_extreme_entry'] = int(bool(v))

            if hasattr(self.brain, '_cci_map'):
                v = getattr(self.brain, '_cci_map', {}).get(timestamp, None)
                if v is not None:
                    out['cci'] = float(v)
            elif hasattr(self.brain, 'source_df') and hasattr(self.brain, 'cci_series'):
                try:
                    m = dict(zip(getattr(self.brain, 'source_df')['ts'], getattr(self.brain, 'cci_series')))
                    v = m.get(timestamp, None)
                    if v is not None:
                        out['cci'] = float(v)
                except Exception:
                    pass

            if hasattr(self.brain, 'filter_mode'):
                out['brain_filter_mode'] = str(getattr(self.brain, 'filter_mode'))

            if hasattr(self.brain, '_kvo_map'):
                v = getattr(self.brain, '_kvo_map', {}).get(timestamp, None)
                if v is not None:
                    out['kvo'] = float(v)
            elif hasattr(self.brain, 'source_df') and hasattr(self.brain, 'kvo_series'):
                try:
                    m = dict(zip(getattr(self.brain, 'source_df')['ts'], getattr(self.brain, 'kvo_series')))
                    v = m.get(timestamp, None)
                    if v is not None:
                        out['kvo'] = float(v)
                except Exception:
                    pass

            if hasattr(self.brain, '_kvo_signal_map'):
                v = getattr(self.brain, '_kvo_signal_map', {}).get(timestamp, None)
                if v is not None:
                    out['kvo_signal'] = float(v)
            elif hasattr(self.brain, 'source_df') and hasattr(self.brain, 'kvo_signal_series'):
                try:
                    m = dict(zip(getattr(self.brain, 'source_df')['ts'], getattr(self.brain, 'kvo_signal_series')))
                    v = m.get(timestamp, None)
                    if v is not None:
                        out['kvo_signal'] = float(v)
                except Exception:
                    pass
        except Exception:
            return out

        return out

    def _log_step(self, timestamp: int, price: float):
        state = self._get_snapshot_state(price)

        long_pos = self.positions[PositionSide.LONG]
        short_pos = self.positions[PositionSide.SHORT]

        wallet_free = state.wallet_balance - state.margin_invested

        notional_total = state.notional_long + state.notional_short
        notional_gap_usdt = abs(state.notional_long - state.notional_short)
        notional_gap_pct = (notional_gap_usdt / notional_total) if notional_total > 0 else 0.0

        gap_mode = (getattr(self.config.mm_config, 'gap_mode', 'notional') or 'notional').lower().strip()
        if gap_mode == 'leveq':
            equity_long = state.long_invested + state.pnl_unrealized_long
            equity_short = state.short_invested + state.pnl_unrealized_short
            mm_val_long = abs(float(equity_long)) * self.config.mm_config.leverage
            mm_val_short = abs(float(equity_short)) * self.config.mm_config.leverage
        else:
            mm_val_long = max(0.0, float(state.notional_long))
            mm_val_short = max(0.0, float(state.notional_short))
        mm_total_val = mm_val_long + mm_val_short
        mm_gap_usdt = abs(mm_val_long - mm_val_short)
        mm_gap_pct = (mm_gap_usdt / mm_total_val) if mm_total_val > 0 else 0.0

        brain_ind = self._get_brain_indicators(timestamp)

        mm_ref_long = float(getattr(getattr(self, 'mm', None), '_pnl_gate_ref_invested', {}).get(PositionSide.LONG, 0.0) or 0.0)
        mm_ref_short = float(getattr(getattr(self, 'mm', None), '_pnl_gate_ref_invested', {}).get(PositionSide.SHORT, 0.0) or 0.0)
        mm_req_long = float(getattr(getattr(self, 'mm', None), '_pnl_gate_required_pct', {}).get(PositionSide.LONG, 0.0) or 0.0)
        mm_req_short = float(getattr(getattr(self, 'mm', None), '_pnl_gate_required_pct', {}).get(PositionSide.SHORT, 0.0) or 0.0)

        mm_gate_long_allows = True if (mm_req_long <= 0.0 or mm_ref_long <= 0.0) else (float(state.pnl_unrealized_long) >= (mm_ref_long * mm_req_long))
        mm_gate_short_allows = True if (mm_req_short <= 0.0 or mm_ref_short <= 0.0) else (float(state.pnl_unrealized_short) >= (mm_ref_short * mm_req_short))

        self.equity_history.append({
            'timestamp': timestamp,
            'price': price,
            'wallet_balance': state.wallet_balance,
            'wallet_free': wallet_free,
            'margin_invested': state.margin_invested,
            'equity': state.wallet_balance + state.pnl_unrealized_long + state.pnl_unrealized_short,
            'margin_total': state.margin_total,
            'long_notional': state.notional_long,
            'short_notional': state.notional_short,
            'notional_gap_usdt': notional_gap_usdt,
            'notional_gap_pct': notional_gap_pct,
            'long_equity': state.equity_long,
            'short_equity': state.equity_short,
            'pnl_long': state.pnl_unrealized_long,
            'pnl_short': state.pnl_unrealized_short,
            'mm_pnl_gate_long_ref_invested': mm_ref_long,
            'mm_pnl_gate_short_ref_invested': mm_ref_short,
            'mm_pnl_gate_long_req_pct': mm_req_long,
            'mm_pnl_gate_short_req_pct': mm_req_short,
            'mm_pnl_gate_long_allows': bool(mm_gate_long_allows),
            'mm_pnl_gate_short_allows': bool(mm_gate_short_allows),
            'long_size': long_pos.size,
            'short_size': short_pos.size,
            'long_entry_price': long_pos.entry_price,
            'short_entry_price': short_pos.entry_price,
            'long_invested': long_pos.invested_amount,
            'short_invested': short_pos.invested_amount,
            'mm_val_long': mm_val_long,
            'mm_val_short': mm_val_short,
            'mm_gap_usdt': mm_gap_usdt,
            'mm_gap_pct': mm_gap_pct,
            'intentions': self._format_intentions(getattr(self, '_last_intentions', [])),
            'actions': self._format_actions(getattr(self, '_last_actions', [])),
            'is_liquidated': self.is_liquidated
        })

        if brain_ind:
            self.equity_history[-1].update(brain_ind)
