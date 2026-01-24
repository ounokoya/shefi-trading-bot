from __future__ import annotations

import math
from typing import List, Sequence

from libs.indicators.common.is_bad import _is_bad


def sma_tv(src: Sequence[float], period: int) -> List[float]:
    n = len(src)
    out = [math.nan] * n
    if n == 0 or period <= 0 or period > n:
        return out

    s = 0.0
    count = 0
    for i in range(n):
        v = float(src[i])
        if _is_bad(v):
            s = 0.0
            count = 0
            out[i] = math.nan
            continue

        s += v
        count += 1

        if i >= period:
            old = float(src[i - period])
            if not _is_bad(old):
                s -= old
                count -= 1

        if i >= period - 1 and count == period:
            out[i] = s / float(period)
        else:
            out[i] = math.nan

    return out
