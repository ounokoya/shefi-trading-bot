from __future__ import annotations

import math
from typing import List, Sequence

from libs.indicators.moving_averages.rma_tv import rma_tv


def atr_tv(high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int) -> List[float]:
    n = len(high)
    if n != len(low) or n != len(close):
        raise ValueError("high, low, close must have same length")

    tr = [math.nan] * n
    for i in range(n):
        h = float(high[i])
        l = float(low[i])
        if i == 0:
            tr[i] = h - l
        else:
            pc = float(close[i - 1])
            r1 = h - l
            r2 = abs(h - pc)
            r3 = abs(l - pc)
            tr[i] = max(r1, r2, r3)

    return rma_tv(tr, period)
