import pandas as pd
import numpy as np
from typing import List, Dict

from ..models import AccountState, SignalIntent
from ..brain import BaseBrain
from libs.indicators.moving_averages.vwma_tv import vwma_tv
from libs.indicators.momentum.dmi_tv import dmi_tv
from libs.indicators.volume.mfi_tv import mfi_tv
from libs.indicators.momentum.cci_tv import cci_tv


class VWMADxMfiCciMtfBrain(BaseBrain):
    def __init__(
        self,
        source_df: pd.DataFrame,
        source_tf: str,
        higher_tf: str = "1h",
        htf_close_opposite: bool = False,
        vwma_fast_ltf: int = 12,
        vwma_slow_ltf: int = 72,
        vwma_fast_htf: int = 12,
        vwma_slow_htf: int = 72,
        dmi_period: int = 14,
        adx_smoothing: int = 6,
        mfi_period: int = 14,
        cci_period: int = 20,
        mfi_high: float = 60.0,
        mfi_low: float = 40.0,
        cci_high: float = 80.0,
        cci_low: float = -80.0,
        ltf_gate_mode: str = "increase_only",
    ):
        self.source_df = source_df.copy()
        self.source_tf = str(source_tf or "").strip()
        self.higher_tf = str(higher_tf or "").strip()
        self.htf_close_opposite = bool(htf_close_opposite)

        self.vwma_fast_ltf = int(vwma_fast_ltf)
        self.vwma_slow_ltf = int(vwma_slow_ltf)
        self.vwma_fast_htf = int(vwma_fast_htf)
        self.vwma_slow_htf = int(vwma_slow_htf)

        self.dmi_period = int(dmi_period)
        self.adx_smoothing = int(adx_smoothing)
        self.mfi_period = int(mfi_period)
        self.cci_period = int(cci_period)

        self.mfi_high = float(mfi_high)
        self.mfi_low = float(mfi_low)
        self.cci_high = float(cci_high)
        self.cci_low = float(cci_low)

        self.ltf_gate_mode = str(ltf_gate_mode or "both").strip().lower()
        if self.ltf_gate_mode not in {"both", "increase_only", "none"}:
            raise ValueError(
                f"VWMADxMfiCciMtfBrain: invalid ltf_gate_mode={self.ltf_gate_mode!r} (expected 'both', 'increase_only', or 'none')"
            )

        self._build_maps()

    def _tf_to_pandas_rule(self, tf: str) -> str:
        tf_norm = str(tf or "").strip().lower()
        if tf_norm.endswith("min"):
            return tf_norm
        if tf_norm.endswith("m") and tf_norm[:-1].isdigit():
            return f"{int(tf_norm[:-1])}min"
        if tf_norm.endswith("h") and tf_norm[:-1].isdigit():
            return f"{int(tf_norm[:-1])}h"
        if tf_norm.endswith("d") and tf_norm[:-1].isdigit():
            return f"{int(tf_norm[:-1])}d"
        return tf_norm

    def _resample_ohlcv(self, df: pd.DataFrame, tf_rule: str, time_col: str) -> pd.DataFrame:
        df = df.copy()
        if "dt" not in df.columns:
            df["dt"] = pd.to_datetime(df[time_col], unit="ms")
        df = df.set_index("dt")
        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        return df.resample(tf_rule).agg(agg_dict).dropna()

    def _compute_cross_dir(self, fast: pd.Series, slow: pd.Series) -> pd.Series:
        diff = fast - slow
        prev = diff.shift(1)
        bull = (prev <= 0.0) & (diff > 0.0)
        bear = (prev >= 0.0) & (diff < 0.0)
        out = pd.Series(0, index=diff.index, dtype=int)
        out[bull] = 1
        out[bear] = -1
        return out

    def _dt_to_ts(self, dt: pd.Timestamp) -> int:
        return int(pd.Timestamp(dt).value // 1_000_000)

    def _build_maps(self) -> None:
        time_col = "open_time" if "open_time" in self.source_df.columns else "ts"

        df_ltf = self.source_df.copy()
        if "dt" not in df_ltf.columns:
            df_ltf["dt"] = pd.to_datetime(df_ltf[time_col], unit="ms")
        df_ltf = df_ltf.set_index("dt")
        df_ltf = df_ltf[["open", "high", "low", "close", "volume"]].dropna()

        close_ltf = df_ltf["close"].tolist()
        vol_ltf = df_ltf["volume"].tolist()
        high_ltf = df_ltf["high"].tolist()
        low_ltf = df_ltf["low"].tolist()

        vwma_fast_ltf = pd.Series(vwma_tv(close_ltf, vol_ltf, self.vwma_fast_ltf), index=df_ltf.index)
        vwma_slow_ltf = pd.Series(vwma_tv(close_ltf, vol_ltf, self.vwma_slow_ltf), index=df_ltf.index)
        ltf_cross_dir = self._compute_cross_dir(vwma_fast_ltf, vwma_slow_ltf)

        adx, plus_di, minus_di = dmi_tv(
            high_ltf,
            low_ltf,
            close_ltf,
            period=self.dmi_period,
            adx_smoothing=self.adx_smoothing,
        )
        mfi = mfi_tv(high_ltf, low_ltf, close_ltf, vol_ltf, period=self.mfi_period)
        cci = cci_tv(high_ltf, low_ltf, close_ltf, self.cci_period)

        df_ltf_ind = df_ltf.copy()
        df_ltf_ind["adx"] = pd.Series(adx, index=df_ltf.index)
        df_ltf_ind["plus_di"] = pd.Series(plus_di, index=df_ltf.index)
        df_ltf_ind["minus_di"] = pd.Series(minus_di, index=df_ltf.index)
        di_sum = df_ltf_ind["plus_di"] + df_ltf_ind["minus_di"]
        df_ltf_ind["dx"] = (df_ltf_ind["plus_di"] - df_ltf_ind["minus_di"]).abs() / di_sum.replace(0.0, pd.NA) * 100.0
        df_ltf_ind["mfi"] = pd.Series(mfi, index=df_ltf.index)
        df_ltf_ind["cci"] = pd.Series(cci, index=df_ltf.index)

        tf_rule = self._tf_to_pandas_rule(self.higher_tf)
        df_htf = self._resample_ohlcv(self.source_df, tf_rule, time_col=time_col)

        close_htf = df_htf["close"].tolist()
        vol_htf = df_htf["volume"].tolist()

        vwma_fast_htf = pd.Series(vwma_tv(close_htf, vol_htf, self.vwma_fast_htf), index=df_htf.index)
        vwma_slow_htf = pd.Series(vwma_tv(close_htf, vol_htf, self.vwma_slow_htf), index=df_htf.index)
        htf_cross_dir = self._compute_cross_dir(vwma_fast_htf, vwma_slow_htf)

        lock_dir_htf = htf_cross_dir.replace(0, np.nan).ffill().fillna(0).astype(int)

        source_ts = self.source_df[time_col].astype(int).tolist()
        source_dt = pd.to_datetime(source_ts, unit="ms")

        lock_dir_active = lock_dir_htf.shift(1).reindex(source_dt, method="ffill").fillna(0).astype(int)

        ltf_cross_event = ltf_cross_dir.shift(1)
        ltf_cross_aligned = ltf_cross_event.reindex(source_dt).fillna(0).astype(int)

        htf_cross_event = htf_cross_dir.shift(1)
        htf_cross_event_dt = {dt: int(v) for dt, v in htf_cross_event.items() if pd.notna(v)}
        htf_cross_aligned = pd.Series(0, index=source_dt, dtype=int)
        for i, dt in enumerate(source_dt):
            if dt in htf_cross_event_dt:
                htf_cross_aligned.iat[i] = int(htf_cross_event_dt[dt])

        ltf_ind_shifted = df_ltf_ind.shift(1)
        ltf_ind_aligned = ltf_ind_shifted.reindex(source_dt, method="ffill")

        plus_di_s = pd.to_numeric(ltf_ind_aligned["plus_di"], errors="coerce")
        minus_di_s = pd.to_numeric(ltf_ind_aligned["minus_di"], errors="coerce")
        dx_s = pd.to_numeric(ltf_ind_aligned["dx"], errors="coerce")
        mfi_s = pd.to_numeric(ltf_ind_aligned["mfi"], errors="coerce")
        cci_s = pd.to_numeric(ltf_ind_aligned["cci"], errors="coerce")

        di_sup_s = pd.concat([plus_di_s, minus_di_s], axis=1).max(axis=1)
        dx_gt_di_sup_s = (dx_s > di_sup_s)

        long_extreme_s = (dx_gt_di_sup_s & (mfi_s >= float(self.mfi_high)) & (cci_s >= float(self.cci_high)))
        short_extreme_s = (dx_gt_di_sup_s & (mfi_s <= float(self.mfi_low)) & (cci_s <= float(self.cci_low)))

        long_extreme_s = long_extreme_s.fillna(False).astype(bool)
        short_extreme_s = short_extreme_s.fillna(False).astype(bool)

        prev_long_extreme_s = long_extreme_s.shift(1, fill_value=False)
        prev_short_extreme_s = short_extreme_s.shift(1, fill_value=False)

        long_extreme_entry_s = (long_extreme_s & ~prev_long_extreme_s).astype(bool)
        short_extreme_entry_s = (short_extreme_s & ~prev_short_extreme_s).astype(bool)

        ltf_cross_ok_s = ((ltf_cross_aligned != 0) & (ltf_cross_aligned == lock_dir_active)).fillna(False).astype(bool)
        regime_id_s = (htf_cross_aligned != 0).astype(int).cumsum()
        dip_seen_s = ltf_cross_ok_s.groupby(regime_id_s).cummax().astype(bool)
        dip_done_s = dip_seen_s.groupby(regime_id_s).shift(1).astype("boolean").fillna(False).astype(bool)

        self._lock_dir_map = dict(zip(source_ts, lock_dir_active.tolist()))
        self._ltf_cross_dir_map = dict(zip(source_ts, ltf_cross_aligned.tolist()))
        self._htf_cross_dir_map = dict(zip(source_ts, htf_cross_aligned.tolist()))

        self._dip_done_map = dict(zip(source_ts, dip_done_s.tolist()))
        self._long_extreme_entry_map = dict(zip(source_ts, long_extreme_entry_s.tolist()))
        self._short_extreme_entry_map = dict(zip(source_ts, short_extreme_entry_s.tolist()))

        self._dx_map = dict(zip(source_ts, ltf_ind_aligned["dx"].tolist()))
        self._plus_di_map = dict(zip(source_ts, ltf_ind_aligned["plus_di"].tolist()))
        self._minus_di_map = dict(zip(source_ts, ltf_ind_aligned["minus_di"].tolist()))
        self._mfi_map = dict(zip(source_ts, ltf_ind_aligned["mfi"].tolist()))
        self._cci_map = dict(zip(source_ts, ltf_ind_aligned["cci"].tolist()))

    def get_intentions(self, timestamp: int, account_state: AccountState) -> List[SignalIntent]:
        lock_dir = int(self._lock_dir_map.get(int(timestamp), 0) or 0)
        if lock_dir == 0:
            return []

        ltf_cross_dir = int(self._ltf_cross_dir_map.get(int(timestamp), 0) or 0)
        htf_cross_dir = int(self._htf_cross_dir_map.get(int(timestamp), 0) or 0)

        dx = self._dx_map.get(int(timestamp), None)
        plus_di = self._plus_di_map.get(int(timestamp), None)
        minus_di = self._minus_di_map.get(int(timestamp), None)
        mfi = self._mfi_map.get(int(timestamp), None)
        cci = self._cci_map.get(int(timestamp), None)

        dip_done = bool(getattr(self, "_dip_done_map", {}).get(int(timestamp), False))
        long_extreme_entry = bool(getattr(self, "_long_extreme_entry_map", {}).get(int(timestamp), False))
        short_extreme_entry = bool(getattr(self, "_short_extreme_entry_map", {}).get(int(timestamp), False))

        if dx is None or plus_di is None or minus_di is None or mfi is None or cci is None:
            return []
        if pd.isna(dx) or pd.isna(plus_di) or pd.isna(minus_di) or pd.isna(mfi) or pd.isna(cci):
            return []

        di_sup = max(float(plus_di), float(minus_di))
        dx_gt_di_sup = float(dx) > di_sup

        ltf_cross_ok = (self.ltf_gate_mode != "none") and (ltf_cross_dir == lock_dir)

        intentions: List[SignalIntent] = []

        if self.htf_close_opposite and (htf_cross_dir == lock_dir) and (htf_cross_dir != 0):
            if lock_dir > 0:
                intentions.append(SignalIntent.SHORT_FORCE_CLOSE)
            else:
                intentions.append(SignalIntent.LONG_FORCE_CLOSE)

        if lock_dir > 0:
            can_decrease = bool(dip_done and long_extreme_entry)

            if can_decrease:
                intentions.append(SignalIntent.LONG_CAN_DECREASE)
                return intentions

            can_increase = False
            if htf_cross_dir == lock_dir:
                can_increase = True
            elif self.ltf_gate_mode != "none" and ltf_cross_ok:
                can_increase = True

            if can_increase:
                if htf_cross_dir == lock_dir:
                    intentions.append(SignalIntent.LONG_HTF_ADD)
                elif self.ltf_gate_mode != "none" and ltf_cross_ok:
                    intentions.append(SignalIntent.LONG_LTF_DIP)
                intentions.append(SignalIntent.LONG_CAN_INCREASE)

        else:
            can_decrease = bool(dip_done and short_extreme_entry)

            if can_decrease:
                intentions.append(SignalIntent.SHORT_CAN_DECREASE)
                return intentions

            can_increase = False
            if htf_cross_dir == lock_dir:
                can_increase = True
            elif self.ltf_gate_mode != "none" and ltf_cross_ok:
                can_increase = True

            if can_increase:
                if htf_cross_dir == lock_dir:
                    intentions.append(SignalIntent.SHORT_HTF_ADD)
                elif self.ltf_gate_mode != "none" and ltf_cross_ok:
                    intentions.append(SignalIntent.SHORT_LTF_DIP)
                intentions.append(SignalIntent.SHORT_CAN_INCREASE)

        return intentions
