from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def validate_required_columns_df(df: pd.DataFrame, *, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
