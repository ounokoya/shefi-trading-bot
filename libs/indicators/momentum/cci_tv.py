from __future__ import annotations

import math
from typing import List, Sequence

from libs.indicators.common.is_bad import _is_bad
from libs.indicators.moving_averages.sma_tv import sma_tv


def cci_tv(high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int) -> List[float]:
    n = len(high)
    if n != len(low) or n != len(close):
        raise ValueError("high, low, close must have same length")

    tp = [(float(high[i]) + float(low[i]) + float(close[i])) / 3.0 for i in range(n)]
    sma = sma_tv(tp, period)

    mean_dev = [math.nan] * n
    for i in range(n):
        if i < period - 1:
            mean_dev[i] = math.nan
            continue
        s = 0.0
        for j in range(i - period + 1, i + 1):
            s += abs(tp[j] - sma[i])
        mean_dev[i] = s / float(period)

    out = [math.nan] * n
    constant = 0.015
    for i in range(n):
        if i < period - 1:
            out[i] = math.nan
        elif mean_dev[i] == 0 or _is_bad(mean_dev[i]) or _is_bad(sma[i]):
            out[i] = 0.0 if (not _is_bad(mean_dev[i]) and mean_dev[i] == 0) else math.nan
        else:
            out[i] = (tp[i] - sma[i]) / (constant * mean_dev[i])

    return out
