from __future__ import annotations

import pandas as pd


def validate_ts_sorted_unique_df(df: pd.DataFrame, *, ts_col: str = "ts") -> None:
    if ts_col not in df.columns:
        raise ValueError(f"Missing ts column: {ts_col}")

    ts = pd.to_numeric(df[ts_col], errors="coerce")
    if ts.isna().any():
        raise ValueError(f"Column {ts_col} contains NaN after numeric coercion")

    if not ts.is_monotonic_increasing:
        raise ValueError(f"Column {ts_col} is not sorted ascending")

    if not ts.is_unique:
        dup = ts[ts.duplicated()].iloc[:10].tolist()
        raise ValueError(f"Column {ts_col} contains duplicates (sample): {dup}")
