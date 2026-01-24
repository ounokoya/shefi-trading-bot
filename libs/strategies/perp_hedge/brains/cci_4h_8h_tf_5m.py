import pandas as pd
import numpy as np
from typing import List, Dict
from ..models import AccountState, SignalIntent
from ..brain import BaseBrain
from libs.indicators.momentum.cci_tv import cci_tv
from libs.indicators.volume.klinger_oscillator_tv import klinger_oscillator_tv

class CCI4h8hTf5mBrain(BaseBrain):
    """
    Double CCI Brain (4H + 8H on 5m Data):
    - CCI Fast: 4H (Length 48)
    - CCI Slow: 8H (Length 96)
    - Optional Klinger Filter.
    """
    def __init__(self, source_df: pd.DataFrame, source_tf: str = "5m", filter_mode: str = "none"):
        self.source_df = source_df.copy()
        self.source_tf = source_tf
        self.filter_mode = filter_mode
        
        # Precompute CCIs
        self.cci_fast_series = self._compute_cci(length=48)
        self.cci_slow_series = self._compute_cci(length=96)
        
        # Precompute Klinger
        if self.filter_mode in ['kvo', 'signal']:
            self._compute_klinger()
        
    def _compute_cci(self, length: int) -> pd.Series:
        high = self.source_df['high'].tolist()
        low = self.source_df['low'].tolist()
        close = self.source_df['close'].tolist()
        
        cci_values = cci_tv(high, low, close, length)
        return pd.Series(cci_values, index=self.source_df.index).fillna(0)
        
    def _compute_klinger(self):
        high = self.source_df['high'].tolist()
        low = self.source_df['low'].tolist()
        close = self.source_df['close'].tolist()
        volume = self.source_df['volume'].tolist()
        
        kvo, kvo_sig = klinger_oscillator_tv(high, low, close, volume)
        
        self.kvo_series = pd.Series(kvo, index=self.source_df.index).fillna(0)
        self.kvo_signal_series = pd.Series(kvo_sig, index=self.source_df.index).fillna(0)

    def get_intentions(self, timestamp: int, account_state: AccountState) -> List[SignalIntent]:
        if not hasattr(self, '_cci_map_fast'):
             self._cci_map_fast = dict(zip(self.source_df['ts'], self.cci_fast_series))
             self._cci_map_slow = dict(zip(self.source_df['ts'], self.cci_slow_series))
             
        fast_val = self._cci_map_fast.get(timestamp, None)
        slow_val = self._cci_map_slow.get(timestamp, None)
        
        if fast_val is None or slow_val is None:
            return []
            
        if pd.isna(fast_val) or pd.isna(slow_val):
            return []
            
        # Klinger Filter
        allow_long = True
        allow_short = True
        
        if self.filter_mode != 'none':
            if not hasattr(self, '_kvo_map'):
                 self._kvo_map = dict(zip(self.source_df['ts'], self.kvo_series))
                 self._kvo_signal_map = dict(zip(self.source_df['ts'], self.kvo_signal_series))
            
            k_val = self._kvo_map.get(timestamp, 0) if self.filter_mode == 'kvo' else self._kvo_signal_map.get(timestamp, 0)
            
            if k_val >= 0: allow_long = False
            if k_val <= 0: allow_short = False
        
        intentions = []
        
        # Consensus: Both must be extreme
        
        # Oversold -> Buy Long / Close Short
        if fast_val < -100 and slow_val < -100:
            if allow_long:
                intentions.append(SignalIntent.LONG_CAN_INCREASE)
            intentions.append(SignalIntent.SHORT_CAN_DECREASE)
            
        # Overbought -> Sell Short / Close Long
        elif fast_val > 100 and slow_val > 100:
            if allow_short:
                intentions.append(SignalIntent.SHORT_CAN_INCREASE)
            intentions.append(SignalIntent.LONG_CAN_DECREASE)
            
        return intentions
