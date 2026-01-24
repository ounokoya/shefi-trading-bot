from __future__ import annotations

import math
from typing import List, Sequence

from libs.indicators.common.is_bad import _is_bad


def obv_tv(close: Sequence[float], volume: Sequence[float]) -> List[float]:
    """On-Balance Volume (OBV) conforme TradingView.

    Règles TradingView (support):
    - Si close[i] > close[i-1]  => obv[i] = obv[i-1] + volume[i]
    - Si close[i] < close[i-1]  => obv[i] = obv[i-1] - volume[i]
    - Si close[i] == close[i-1] => obv[i] = obv[i-1]

    Notes d'implémentation:
    - Initialisation: obv[0] = 0.0 (choix standard pour éviter un décalage arbitraire).
    - Propagation NaN: si une valeur nécessaire est NaN/inf, on produit NaN.
    """

    n = len(close)
    if n != len(volume):
        raise ValueError("close and volume must have same length")

    out = [math.nan] * n
    if n == 0:
        return out

    c0 = float(close[0])
    out[0] = 0.0 if (not _is_bad(c0)) else math.nan

    for i in range(1, n):
        c = float(close[i])
        pc = float(close[i - 1])
        v = float(volume[i])
        prev = out[i - 1]

        if _is_bad(c) or _is_bad(pc) or _is_bad(v) or _is_bad(prev):
            out[i] = math.nan
            continue

        if c > pc:
            out[i] = prev + v
        elif c < pc:
            out[i] = prev - v
        else:
            out[i] = prev

    return out
