from __future__ import annotations

import math
from typing import List, Sequence


def vwma_tv(close: Sequence[float], volume: Sequence[float], period: int) -> List[float]:
    n = len(close)
    if n != len(volume):
        raise ValueError("close and volume must have same length")
    out = [math.nan] * n
    if period <= 0 or n == 0 or period > n:
        return out

    for i in range(period - 1, n):
        sum_wp = 0.0
        sum_v = 0.0
        for j in range(i - period + 1, i + 1):
            c = float(close[j])
            v = float(volume[j])
            sum_wp += c * v
            sum_v += v
        out[i] = (sum_wp / sum_v) if sum_v != 0 else math.nan

    return out
