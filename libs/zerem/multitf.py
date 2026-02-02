from __future__ import annotations

import math

import pandas as pd

from libs.indicators.momentum.cci_tv import cci_tv
from libs.zerem.timeframes import indicator_params_for_tf, tf_to_minutes, tf_to_pandas_rule


def ensure_cci_tf_column(df: pd.DataFrame, *, base_tf: str, target_tf: str) -> str:
    col = f"cci_tf_{str(target_tf)}"
    if col in df.columns:
        return col
    if "dt" not in df.columns:
        raise ValueError("Missing dt column")

    if tf_to_minutes(str(target_tf)) < tf_to_minutes(str(base_tf)):
        raise ValueError(f"target_tf must be >= base_tf: base_tf={base_tf} target_tf={target_tf}")

    rule = tf_to_pandas_rule(str(target_tf))
    tmp = df[["dt", "open", "high", "low", "close", "volume"]].copy()
    tmp["dt"] = pd.to_datetime(tmp["dt"], utc=True, errors="coerce")
    tmp = tmp.dropna(subset=["dt"]).sort_values("dt")
    tmp = tmp.set_index("dt")

    res = tmp.resample(rule, label="right", closed="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    res = res.dropna(subset=["high", "low", "close"]).reset_index()
    if res.empty:
        df[col] = math.nan
        return col

    tf_params = indicator_params_for_tf(str(target_tf))
    high = res["high"].astype(float).tolist()
    low = res["low"].astype(float).tolist()
    close = res["close"].astype(float).tolist()
    res[col] = cci_tv(high, low, close, period=int(tf_params["cci_period"]))
    res = res[["dt", col]].sort_values("dt")

    base_dt = df[["dt"]].copy()
    base_dt["dt"] = pd.to_datetime(base_dt["dt"], utc=True, errors="coerce")
    base_dt = base_dt.sort_values("dt")

    aligned = pd.merge_asof(base_dt, res, on="dt", direction="backward")
    df[col] = aligned[col].ffill().to_numpy()
    return col
