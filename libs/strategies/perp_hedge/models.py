from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

class SignalIntent(Enum):
    LONG_CAN_INCREASE = auto()
    LONG_CAN_DECREASE = auto()
    SHORT_CAN_INCREASE = auto()
    SHORT_CAN_DECREASE = auto()
    LONG_FORCE_CLOSE = auto()
    SHORT_FORCE_CLOSE = auto()
    LONG_HTF_ADD = auto()
    SHORT_HTF_ADD = auto()
    LONG_LTF_DIP = auto()
    SHORT_LTF_DIP = auto()

class PositionSide(Enum):
    LONG = auto()
    SHORT = auto()

@dataclass
class Position:
    side: PositionSide
    size: float  # Quantity in base asset (e.g. BTC)
    entry_price: float
    invested_amount: float = 0.0 # Marge investie (Cost basis)

@dataclass
class AccountState:
    wallet_balance: float  # Marge non investie + Marge investie (fixed state)
    margin_invested: float # Used margin
    
    # Positions state
    long_size: float
    long_entry_price: float
    long_invested: float # Added to state
    
    short_size: float
    short_entry_price: float
    short_invested: float # Added to state
    
    # Real-time metrics (updated by engine before passing to MM)
    current_price: float = 0.0
    pnl_unrealized_long: float = 0.0
    pnl_unrealized_short: float = 0.0
    
    @property
    def equity_long(self) -> float:
        return self.long_invested + self.pnl_unrealized_long

    @property
    def equity_short(self) -> float:
        return self.short_invested + self.pnl_unrealized_short

    @property
    def notional_long(self) -> float:
        return self.long_size * self.current_price
        
    @property
    def notional_short(self) -> float:
        return self.short_size * self.current_price
    
    @property
    def margin_total(self) -> float:
        # wallet_balance + max(0, pnl_unrealized_total)
        # Note: pnl_unrealized_total = pnl_long + pnl_short
        pnl_total = self.pnl_unrealized_long + self.pnl_unrealized_short
        return self.wallet_balance + max(0.0, pnl_total)

class ActionType(Enum):
    INCREASE = auto() # Add to position
    DECREASE = auto() # Reduce position

@dataclass
class MMAction:
    side: PositionSide
    action_type: ActionType
    qty_usdt: float # Notional amount to trade
    reason: str = ""
