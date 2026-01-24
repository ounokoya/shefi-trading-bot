from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def validate_ohlcv_numeric_df(df: pd.DataFrame, *, cols: Sequence[str]) -> None:
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
        s = pd.to_numeric(df[c], errors="coerce")
        if s.isna().any():
            raise ValueError(f"Column {c} contains NaN after numeric coercion")
