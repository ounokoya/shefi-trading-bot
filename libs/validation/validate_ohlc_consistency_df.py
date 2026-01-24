from __future__ import annotations

import pandas as pd


def validate_ohlc_consistency_df(
    df: pd.DataFrame,
    *,
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> None:
    o = pd.to_numeric(df[open_col], errors="coerce")
    h = pd.to_numeric(df[high_col], errors="coerce")
    l = pd.to_numeric(df[low_col], errors="coerce")
    c = pd.to_numeric(df[close_col], errors="coerce")

    bad = (
        (h < l)
        | (h < o)
        | (h < c)
        | (l > o)
        | (l > c)
    )

    if bad.any():
        idx = df.index[bad].tolist()[:10]
        raise ValueError(f"OHLC consistency failed for {int(bad.sum())} rows (sample idx): {idx}")
