from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from libs.indicators.common.is_bad import _is_bad
from libs.indicators.moving_averages.ema_tv import ema_tv


def macd_tv(
    prices: Sequence[float],
    fast_period: int,
    slow_period: int,
    signal_period: int,
) -> Tuple[List[float], List[float], List[float]]:
    n = len(prices)
    if n == 0:
        return [], [], []

    fast = ema_tv(prices, fast_period)
    slow = ema_tv(prices, slow_period)

    macd_line = [math.nan] * n
    for i in range(n):
        if not _is_bad(fast[i]) and not _is_bad(slow[i]):
            macd_line[i] = fast[i] - slow[i]
        else:
            macd_line[i] = math.nan

    signal = ema_tv(macd_line, signal_period)

    hist = [math.nan] * n
    for i in range(n):
        if not _is_bad(macd_line[i]) and not _is_bad(signal[i]):
            hist[i] = macd_line[i] - signal[i]
        else:
            hist[i] = math.nan

    return macd_line, signal, hist
