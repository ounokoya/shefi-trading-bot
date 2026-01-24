from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def rolling_quantile_features_series(
    s: pd.Series,
    *,
    window: int,
    quantiles: Iterable[float],
    prefix: str,
) -> pd.DataFrame:
    q = list(quantiles)
    if window <= 0:
        raise ValueError("window must be > 0")
    if not q:
        raise ValueError("quantiles must be non-empty")
    if any((p <= 0.0 or p >= 1.0) for p in q):
        raise ValueError("quantiles must be in (0,1)")
    if any(q[i] >= q[i + 1] for i in range(len(q) - 1)):
        raise ValueError("quantiles must be strictly increasing")

    x = pd.to_numeric(s, errors="coerce").astype(float).to_numpy()
    n = int(x.shape[0])

    amp = np.full(n, np.nan, dtype=float)
    pos = np.full(n, np.nan, dtype=float)
    pct = np.full(n, np.nan, dtype=float)
    q_vals = np.full((len(q), n), np.nan, dtype=float)
    dist = np.full((len(q), n), np.nan, dtype=float)
    bin_idx = np.full(n, np.nan, dtype=float)

    for i in range(window - 1, n):
        w = x[i - window + 1 : i + 1]
        if np.isnan(w).any() or math.isnan(x[i]):
            continue

        w_min = float(np.min(w))
        w_max = float(np.max(w))
        w_amp = w_max - w_min
        amp[i] = w_amp

        if w_amp != 0.0:
            pos[i] = (float(x[i]) - w_min) / w_amp

        q_i = np.quantile(w, q)
        q_vals[:, i] = q_i

        if w_amp != 0.0:
            dist[:, i] = (float(x[i]) - q_i) / w_amp

        v = float(x[i])
        c_lt = float(np.sum(w < v))
        c_eq = float(np.sum(w == v))
        pct_i = (c_lt + 0.5 * c_eq) / float(window)
        pct[i] = pct_i
        bin_idx[i] = float(np.searchsorted(q, pct_i, side="right"))

    suffix = f"L{window}"
    out: dict[str, np.ndarray] = {
        f"{prefix}_amp_{suffix}": amp,
        f"{prefix}_pos_{suffix}": pos,
        f"{prefix}_pct_{suffix}": pct,
        f"{prefix}_bin_{suffix}": bin_idx,
    }

    for j, p in enumerate(q):
        p_int = int(round(p * 100.0))
        out[f"{prefix}_q{p_int}_{suffix}"] = q_vals[j]
        out[f"{prefix}_dist_q{p_int}_{suffix}"] = dist[j]

    return pd.DataFrame(out, index=s.index)
