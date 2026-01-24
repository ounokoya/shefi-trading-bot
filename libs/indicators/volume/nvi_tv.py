from __future__ import annotations

import math
from typing import List, Sequence

from libs.indicators.common.is_bad import _is_bad


def nvi_tv(close: Sequence[float], volume: Sequence[float], *, start: float = 1000.0) -> List[float]:
    n = len(close)
    if n != len(volume):
        raise ValueError("close and volume must have same length")

    out = [math.nan] * n
    if n == 0:
        return out

    c0 = float(close[0])
    out[0] = float(start) if not _is_bad(c0) else math.nan

    for i in range(1, n):
        c = float(close[i])
        pc = float(close[i - 1])
        v = float(volume[i])
        pv = float(volume[i - 1])
        prev = out[i - 1]

        if _is_bad(c) or _is_bad(pc) or _is_bad(v) or _is_bad(pv) or _is_bad(prev) or pc == 0.0:
            out[i] = math.nan
            continue

        if v < pv:
            out[i] = prev + (prev * (c - pc) / pc)
        else:
            out[i] = prev

    return out
