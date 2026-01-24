import pandas as pd
import numpy as np
from typing import List, Dict
from ..models import AccountState, SignalIntent
from ..brain import BaseBrain
from libs.indicators.momentum.cci_tv import cci_tv
from libs.indicators.volume.klinger_oscillator_tv import klinger_oscillator_tv

class CCI1hTf5mBrain(BaseBrain):
    """
    CCI 1H Brain (on 5m Data):
    - Calculates CCI(12) on the source timeframe (5m).
    - Optional Klinger Filter:
      - filter_mode='kvo': Require KVO < 0 for Long, KVO > 0 for Short.
      - filter_mode='signal': Require KVO Signal < 0 for Long, KVO Signal > 0 for Short.
    """
    def __init__(self, source_df: pd.DataFrame, source_tf: str = "5m", cci_length: int = 12, filter_mode: str = "none"):
        self.source_df = source_df.copy() # Work on a copy
        self.source_tf = source_tf
        self.cci_length = 12
        self.filter_mode = filter_mode
        
        # Precompute CCI using official module
        self.cci_series = self._compute_cci()
        
        # Precompute Klinger if needed using official module
        if self.filter_mode in ['kvo', 'signal']:
            self._compute_klinger()
        
    def _compute_cci(self) -> pd.Series:
        # Convert to lists for the TV module
        high = self.source_df['high'].tolist()
        low = self.source_df['low'].tolist()
        close = self.source_df['close'].tolist()
        
        # Call cci_tv
        cci_values = cci_tv(high, low, close, self.cci_length)
        
        # Convert back to Series
        return pd.Series(cci_values, index=self.source_df.index).fillna(0)
        
    def _compute_klinger(self):
        high = self.source_df['high'].tolist()
        low = self.source_df['low'].tolist()
        close = self.source_df['close'].tolist()
        volume = self.source_df['volume'].tolist()
        
        # Call klinger_oscillator_tv
        kvo, kvo_sig = klinger_oscillator_tv(high, low, close, volume)
        
        self.kvo_series = pd.Series(kvo, index=self.source_df.index).fillna(0)
        self.kvo_signal_series = pd.Series(kvo_sig, index=self.source_df.index).fillna(0)

    def get_intentions(self, timestamp: int, account_state: AccountState) -> List[SignalIntent]:
        # Map timestamp to CCI value
        if not hasattr(self, '_cci_map'):
             self._cci_map = dict(zip(self.source_df['ts'].astype(int), self.cci_series))
        
        cci_val = self._cci_map.get(timestamp, None)
        
        if cci_val is None or pd.isna(cci_val):
            return []
            
        # Klinger Filter
        allow_long = True
        allow_short = True
        
        if self.filter_mode != 'none':
            if not hasattr(self, '_kvo_map'):
                 self._kvo_map = dict(zip(self.source_df['ts'].astype(int), self.kvo_series))
                 self._kvo_signal_map = dict(zip(self.source_df['ts'].astype(int), self.kvo_signal_series))
            
            k_val = self._kvo_map.get(timestamp, 0) if self.filter_mode == 'kvo' else self._kvo_signal_map.get(timestamp, 0)
            
            # Rule: Pour CCI negatif (Long), Klinger doit etre negatif
            if k_val >= 0: allow_long = False
            
            # Rule: Pour CCI positif (Short), Klinger doit etre positif
            if k_val <= 0: allow_short = False
        
        intentions = []
        
        # Oversold (CCI < -100) -> Long OK (Rebound), Short Close OK
        if cci_val < -100:
            if allow_long:
                intentions.append(SignalIntent.LONG_CAN_INCREASE)
            intentions.append(SignalIntent.SHORT_CAN_DECREASE) # Closing is always allowed? Assuming yes for now.
            
        # Overbought (CCI > 100) -> Short OK (Drop), Long Close OK
        elif cci_val > 100:
            if allow_short:
                intentions.append(SignalIntent.SHORT_CAN_INCREASE)
            intentions.append(SignalIntent.LONG_CAN_DECREASE)
            
        return intentions
