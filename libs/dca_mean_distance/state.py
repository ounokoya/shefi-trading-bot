from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MiniCycle:
    cycle_id: int
    created_ts: int
    size: float = 0.0
    avg_open: float = 0.0
    avg_close: float = 0.0
    closed_qty: float = 0.0
    tp_index: int = 0
    next_tp_price: float = 0.0
    tp_reached: bool = False
    current_d_index: int = 0
    next_target_price: float = 0.0


@dataclass
class DcaState:
    wallet: float
    position_size: float = 0.0
    avg_price: float = 0.0
    margin_invested: float = 0.0
    portions_used: int = 0
    current_d_index: int = 0
    next_target_price: float = 0.0
    cycles_completed: int = 0
    last_exit_ts: int = 0
    is_liquidated: bool = False
    liquidation_reason: str = ""

    tp_phase: str = "OPEN"
    tp_active: Optional[MiniCycle] = None
    tp_bucket: list[MiniCycle] = field(default_factory=list)
    tp_next_cycle_id: int = 1
    prev_entry_signal_raw: bool = False
    prev_tp_partial_signal_raw: bool = False
