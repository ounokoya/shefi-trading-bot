import pandas as pd
import numpy as np
from typing import List
from ..models import AccountState, SignalIntent
from ..brain import BaseBrain
from libs.indicators.momentum.cci_tv import cci_tv
from libs.indicators.volume.klinger_oscillator_tv import klinger_oscillator_tv

class CCIDouble8h1dTf15mBrain(BaseBrain):
    """
    Double CCI Brain (8H + 1D on 15m Data):
    - Resamples 5m source data to 15m.
    - CCI Fast: 8H (Length 32 on 15m)
    - CCI Slow: 1D (Length 96 on 15m)
    - Optional Klinger Filter.
    """
    def __init__(self, source_df: pd.DataFrame, source_tf: str = "5m", filter_mode: str = "none"):
        self.source_df = source_df.copy()
        self.filter_mode = filter_mode
        
        # 1. Resample to 15m
        self.df_15m = self._resample(self.source_df, "15min")
        
        # 2. Compute CCIs on 15m
        self.cci_fast_series = self._compute_cci(self.df_15m, length=32)
        self.cci_slow_series = self._compute_cci(self.df_15m, length=96)
        
        # 3. Compute Klinger on 15m if needed
        if self.filter_mode in ['kvo', 'signal']:
            self._compute_klinger_15m()
            
        # 4. Align back
        self.cci_fast_map = self._create_map(self.source_df, self.cci_fast_series)
        self.cci_slow_map = self._create_map(self.source_df, self.cci_slow_series)
        
        if self.filter_mode != 'none':
            self.kvo_map = self._create_map(self.source_df, self.kvo_series_15m)
            self.kvo_signal_map = self._create_map(self.source_df, self.kvo_signal_series_15m)

    def _resample(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        df = df.copy()
        if 'dt' not in df.columns:
            df['dt'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('dt', inplace=True)
        
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        resampled = df.resample(tf).agg(agg_dict).dropna()
        return resampled

    def _compute_cci(self, df: pd.DataFrame, length: int) -> pd.Series:
        high = df['high'].tolist()
        low = df['low'].tolist()
        close = df['close'].tolist()
        
        cci_values = cci_tv(high, low, close, length)
        return pd.Series(cci_values, index=df.index).fillna(0)
        
    def _compute_klinger_15m(self):
        high = self.df_15m['high'].tolist()
        low = self.df_15m['low'].tolist()
        close = self.df_15m['close'].tolist()
        volume = self.df_15m['volume'].tolist()
        
        kvo, kvo_sig = klinger_oscillator_tv(high, low, close, volume)
        
        self.kvo_series_15m = pd.Series(kvo, index=self.df_15m.index).fillna(0)
        self.kvo_signal_series_15m = pd.Series(kvo_sig, index=self.df_15m.index).fillna(0)

    def _create_map(self, source_df: pd.DataFrame, series_15m: pd.Series) -> dict:
        shifted = series_15m.shift(1)
        source_dt_idx = pd.to_datetime(source_df['ts'], unit='ms')
        aligned = shifted.reindex(source_dt_idx, method='ffill')
        return dict(zip(source_df['ts'], aligned.values))

    def get_intentions(self, timestamp: int, account_state: AccountState) -> List[SignalIntent]:
        fast_val = self.cci_fast_map.get(timestamp, 0.0)
        slow_val = self.cci_slow_map.get(timestamp, 0.0)
        
        if pd.isna(fast_val) or pd.isna(slow_val): return []
        
        # Klinger Filter
        allow_long = True
        allow_short = True
        
        if self.filter_mode != 'none':
            k_val = self.kvo_map.get(timestamp, 0) if self.filter_mode == 'kvo' else self.kvo_signal_map.get(timestamp, 0)
            if k_val >= 0: allow_long = False
            if k_val <= 0: allow_short = False
        
        intentions = []
        # Consensus
        if fast_val < -100 and slow_val < -100:
            if allow_long:
                intentions.append(SignalIntent.LONG_CAN_INCREASE)
            intentions.append(SignalIntent.SHORT_CAN_DECREASE)
        elif fast_val > 100 and slow_val > 100:
            if allow_short:
                intentions.append(SignalIntent.SHORT_CAN_INCREASE)
            intentions.append(SignalIntent.LONG_CAN_DECREASE)
            
        return intentions
