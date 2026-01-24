from __future__ import annotations

import math
from typing import List, Sequence

from libs.indicators.common.is_bad import _is_bad
from libs.indicators.moving_averages.sma_tv import sma_tv


def ema_tv(src: Sequence[float], period: int) -> List[float]:
    n = len(src)
    out = [math.nan] * n
    if period <= 0 or n == 0 or period > n:
        return out

    sma = sma_tv(src, period)
    alpha = 2.0 / (float(period) + 1.0)

    seeded = False
    for i in range(period - 1, n):
        if not seeded:
            seed = sma[i]
            if not _is_bad(seed):
                out[i] = seed
                seeded = True
            else:
                out[i] = math.nan
            continue

        prev = out[i - 1]
        v = float(src[i])
        if _is_bad(prev) or _is_bad(v):
            out[i] = math.nan
            seeded = False
            continue

        out[i] = alpha * v + (1.0 - alpha) * prev

    return out
