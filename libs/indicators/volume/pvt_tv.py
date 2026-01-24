from __future__ import annotations

import math
from typing import List, Sequence

from libs.indicators.common.is_bad import _is_bad


def pvt_tv(close: Sequence[float], volume: Sequence[float]) -> List[float]:
    n = len(close)
    if n != len(volume):
        raise ValueError("close and volume must have same length")

    out = [math.nan] * n
    if n == 0:
        return out

    c0 = float(close[0])
    out[0] = 0.0 if not _is_bad(c0) else math.nan

    for i in range(1, n):
        c = float(close[i])
        pc = float(close[i - 1])
        v = float(volume[i])
        prev = out[i - 1]

        if _is_bad(c) or _is_bad(pc) or _is_bad(v) or _is_bad(prev) or pc == 0.0:
            out[i] = math.nan
            continue

        out[i] = prev + (v * (c - pc) / pc)

    return out
