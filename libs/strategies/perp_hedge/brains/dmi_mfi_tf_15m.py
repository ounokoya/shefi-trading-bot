import pandas as pd
from typing import List

from ..models import AccountState, SignalIntent
from ..brain import BaseBrain
from libs.indicators.momentum.dmi_tv import dmi_tv
from libs.indicators.volume.mfi_tv import mfi_tv


class DMIMFITf15mBrain(BaseBrain):
    def __init__(
        self,
        source_df: pd.DataFrame,
        source_tf: str = "15m",
        dmi_period: int = 14,
        adx_smoothing: int = 6,
        maturity_mode: str = "di_max",
        adx_min_threshold: float = 20.0,
        mfi_period: int = 14,
        strict_tf: bool = True,
    ):
        self.source_df = source_df.copy()
        self.source_tf = source_tf
        self.dmi_period = int(dmi_period)
        self.adx_smoothing = int(adx_smoothing)
        self.maturity_mode = str(maturity_mode or "di_max").strip().lower()
        self.adx_min_threshold = float(adx_min_threshold)
        self.mfi_period = int(mfi_period)

        tf_norm = (self.source_tf or "").lower().strip()
        if strict_tf and tf_norm not in {"15m", "15min"}:
            raise ValueError(f"DMIMFITf15mBrain requires source_tf='15m' (got {self.source_tf!r})")

        if self.maturity_mode not in {"di_max", "adx_threshold"}:
            raise ValueError(
                f"DMIMFITf15mBrain: invalid maturity_mode={self.maturity_mode!r} (expected 'di_max' or 'adx_threshold')"
            )

        if tf_norm in {"15m", "15min"}:
            df = self.source_df.copy()
            if 'dt' not in df.columns:
                df['dt'] = pd.to_datetime(df['ts'], unit='ms')
            df.set_index('dt', inplace=True)
            self.df_15m = df[['open', 'high', 'low', 'close', 'volume']].dropna()
        else:
            self.df_15m = self._resample(self.source_df, "15min")
        self._compute_indicators_15m()

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

    def _compute_indicators_15m(self) -> None:
        high = self.df_15m['high'].tolist()
        low = self.df_15m['low'].tolist()
        close = self.df_15m['close'].tolist()
        volume = self.df_15m['volume'].tolist()

        adx, plus_di, minus_di = dmi_tv(high, low, close, period=self.dmi_period, adx_smoothing=self.adx_smoothing)
        mfi = mfi_tv(high, low, close, volume, period=self.mfi_period)

        df = self.df_15m.copy()
        df['adx'] = pd.Series(adx, index=df.index)
        df['plus_di'] = pd.Series(plus_di, index=df.index)
        df['minus_di'] = pd.Series(minus_di, index=df.index)
        di_sum = df['plus_di'] + df['minus_di']
        df['dx'] = (df['plus_di'] - df['minus_di']).abs() / di_sum.replace(0.0, pd.NA) * 100.0
        df['mfi'] = pd.Series(mfi, index=df.index)

        self.df_15m_ind = df

    def get_intentions(self, timestamp: int, account_state: AccountState) -> List[SignalIntent]:
        dt = pd.to_datetime(int(timestamp), unit='ms')

        if dt.minute % 15 != 0 or dt.second != 0:
            return []

        bar15 = dt.floor('15min') - pd.Timedelta(minutes=15)
        prev15 = bar15 - pd.Timedelta(minutes=15)

        if bar15 not in self.df_15m_ind.index or prev15 not in self.df_15m_ind.index:
            return []

        row = self.df_15m_ind.loc[bar15]
        prev = self.df_15m_ind.loc[prev15]

        adx_now = float(row['adx'])
        plus_di_now = float(row['plus_di'])
        minus_di_now = float(row['minus_di'])
        dx_now = float(row['dx'])
        mfi_now = float(row['mfi'])

        adx_prev = float(prev['adx'])
        dx_prev = float(prev['dx'])

        if pd.isna(adx_now) or pd.isna(plus_di_now) or pd.isna(minus_di_now) or pd.isna(dx_now) or pd.isna(mfi_now):
            return []
        if pd.isna(adx_prev) or pd.isna(dx_prev):
            return []

        if self.maturity_mode == "di_max":
            is_mature = adx_now > max(plus_di_now, minus_di_now)
        else:
            is_mature = adx_now >= self.adx_min_threshold
        if not is_mature:
            return []

        dx_cross_under = (dx_prev > adx_prev) and (dx_now <= adx_now)
        if not dx_cross_under:
            return []

        intentions: List[SignalIntent] = []

        if plus_di_now > minus_di_now and mfi_now > 50.0:
            intentions.append(SignalIntent.LONG_CAN_DECREASE)
            intentions.append(SignalIntent.SHORT_CAN_INCREASE)
        elif minus_di_now > plus_di_now and mfi_now < 50.0:
            intentions.append(SignalIntent.SHORT_CAN_DECREASE)
            intentions.append(SignalIntent.LONG_CAN_INCREASE)

        return intentions
