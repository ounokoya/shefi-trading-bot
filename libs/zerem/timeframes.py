from __future__ import annotations

from typing import Dict

import pandas as pd


def indicator_params_for_tf(timeframe: str) -> Dict[str, int]:
    tf = str(timeframe or "").strip().lower()
    if tf == "5m":
        return {
            "cci_period": 96,
            "stoch_k_period": 96,
            "stoch_k_smooth": 2,
            "stoch_d_period": 3,
        }
    if tf == "15m":
        return {
            "cci_period": 96,
            "stoch_k_period": 96,
            "stoch_k_smooth": 2,
            "stoch_d_period": 3,
        }
    if tf == "1h":
        return {
            "cci_period": 168,
            "stoch_k_period": 168,
            "stoch_k_smooth": 2,
            "stoch_d_period": 3,
        }
    if tf == "2h":
        return {
            "cci_period": 84,
            "stoch_k_period": 84,
            "stoch_k_smooth": 2,
            "stoch_d_period": 3,
        }
    if tf == "6h":
        return {
            "cci_period": 120,
            "stoch_k_period": 120,
            "stoch_k_smooth": 2,
            "stoch_d_period": 3,
        }
    if tf == "1d":
        return {
            "cci_period": 90,
            "stoch_k_period": 90,
            "stoch_k_smooth": 2,
            "stoch_d_period": 3,
        }
    return {
        "cci_period": 20,
        "stoch_k_period": 14,
        "stoch_k_smooth": 3,
        "stoch_d_period": 3,
    }


def tf_to_minutes(timeframe: str) -> int:
    tf = str(timeframe or "").strip().lower()
    if not tf:
        return 0

    if tf.endswith("min") and tf[:-3].isdigit():
        return int(tf[:-3])

    unit = tf[-1:]
    if unit not in {"m", "h", "d", "w"}:
        if tf.isdigit():
            return int(tf)
        return 0

    n_raw = tf[:-1]
    if not n_raw.isdigit():
        return 0

    n = int(n_raw)
    if unit == "m":
        return int(n)
    if unit == "h":
        return int(n) * 60
    if unit == "d":
        return int(n) * 60 * 24
    if unit == "w":
        return int(n) * 60 * 24 * 7
    return 0


def tf_to_pandas_rule(timeframe: str) -> str:
    tf = str(timeframe).strip().lower()
    unit = tf[-1:]
    try:
        value = int(tf[:-1])
    except ValueError:
        value = 1

    if unit == "m":
        return f"{int(value)}min"
    if unit == "h":
        return f"{int(value)}h"
    if unit == "d":
        return f"{int(value)}D"
    if unit == "w":
        return f"{int(value)}W"
    return f"{int(value)}min"


def tf_to_timedelta(timeframe: str, n: int) -> pd.Timedelta:
    tf = str(timeframe or "").strip().lower()
    if not tf:
        return pd.Timedelta(minutes=0)

    unit = tf[-1:]
    try:
        value = int(tf[:-1])
    except ValueError:
        value = 1

    minutes = int(value)
    if unit == "h":
        minutes = int(value) * 60
    elif unit == "d":
        minutes = int(value) * 60 * 24
    elif unit == "w":
        minutes = int(value) * 60 * 24 * 7
    return pd.Timedelta(minutes=int(minutes) * int(n))
