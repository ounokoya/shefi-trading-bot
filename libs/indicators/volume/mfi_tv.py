from __future__ import annotations

import math
from typing import List, Sequence


def mfi_tv(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    volume: Sequence[float],
    period: int,
) -> List[float]:
    n = len(high)
    if n != len(low) or n != len(close) or n != len(volume):
        raise ValueError("high, low, close, volume must have same length")

    tp = [(float(high[i]) + float(low[i]) + float(close[i])) / 3.0 for i in range(n)]
    raw_mf = [tp[i] * float(volume[i]) for i in range(n)]

    pos = [0.0] * n
    neg = [0.0] * n
    for i in range(n):
        if i == 0:
            pos[i] = 0.0
            neg[i] = 0.0
            continue
        if tp[i] > tp[i - 1]:
            pos[i] = raw_mf[i]
            neg[i] = 0.0
        elif tp[i] < tp[i - 1]:
            pos[i] = 0.0
            neg[i] = raw_mf[i]
        else:
            pos[i] = 0.0
            neg[i] = 0.0

    out = [math.nan] * n
    if period <= 0 or n == 0:
        return out

    for i in range(period, n):
        sum_pos = 0.0
        sum_neg = 0.0
        valid = 0
        for j in range(i - period + 1, i + 1):
            sum_pos += pos[j]
            sum_neg += neg[j]
            valid += 1
        if valid != period:
            continue

        if sum_pos > 0 and sum_neg == 0:
            out[i] = 100.0
        elif sum_pos == 0 and sum_neg > 0:
            out[i] = 0.0
        elif sum_pos == 0 and sum_neg == 0:
            out[i] = 50.0
        else:
            ratio = sum_pos / sum_neg
            out[i] = 100.0 - (100.0 / (1.0 + ratio))

    return out
