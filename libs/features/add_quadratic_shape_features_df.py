from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from libs.features.quadratic_shape_features_series import quadratic_shape_features_series


def add_quadratic_shape_features_df(
    df: pd.DataFrame,
    *,
    cols: Sequence[str],
    windows: Iterable[int],
) -> pd.DataFrame:
    out = df.copy()
    missing = [str(c) for c in cols if str(c) not in out.columns]
    if missing:
        raise ValueError(f"Missing cols for quadratic shape features: {missing}")

    for c in cols:
        for w in windows:
            feats = quadratic_shape_features_series(out[str(c)], window=int(w), prefix=str(c))
            out = out.join(feats)

    return out
