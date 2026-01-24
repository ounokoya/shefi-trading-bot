from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from libs.features.rolling_quantile_features_series import rolling_quantile_features_series


def add_rolling_quantile_features_df(
    df: pd.DataFrame,
    *,
    cols: Sequence[str],
    window: int,
    quantiles: Iterable[float],
) -> pd.DataFrame:
    out = df.copy()
    missing = [str(c) for c in cols if str(c) not in out.columns]
    if missing:
        raise ValueError(f"Missing cols for rolling quantiles: {missing}")
    for c in cols:
        feats = rolling_quantile_features_series(
            out[c],
            window=window,
            quantiles=quantiles,
            prefix=str(c),
        )
        out = out.join(feats)
    return out
