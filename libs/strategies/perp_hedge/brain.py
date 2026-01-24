from abc import ABC, abstractmethod
from typing import List
from .models import AccountState, SignalIntent

class BaseBrain(ABC):
    @abstractmethod
    def get_intentions(self, timestamp: int, account_state: AccountState) -> List[SignalIntent]:
        pass

class DummyBrain(BaseBrain):
    """
    A Brain that always returns all permissions (Always Yes).
    Used to test the Money Manager's pure balancing logic.
    """
    def get_intentions(self, timestamp: int, account_state: AccountState) -> List[SignalIntent]:
        return [
            SignalIntent.LONG_CAN_INCREASE,
            SignalIntent.LONG_CAN_DECREASE,
            SignalIntent.SHORT_CAN_INCREASE,
            SignalIntent.SHORT_CAN_DECREASE
        ]
