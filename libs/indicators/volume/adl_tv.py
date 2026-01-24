from __future__ import annotations

import math
from typing import List, Sequence

from libs.indicators.common.is_bad import _is_bad


def adl_tv(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    volume: Sequence[float],
) -> List[float]:
    """Accumulation/Distribution Line (ADL) conforme TradingView.

    Source (TradingView Support):
    Accumulation/Distribution = ((Close – Low) – (High – Close)) / (High – Low) * Volume

    Décomposition:
    - Money Flow Multiplier (MFM) = ((close - low) - (high - close)) / (high - low)
                               = (2*close - high - low) / (high - low)
    - Money Flow Volume (MFV) = MFM * volume
    - ADL = cumulative sum(MFV)

    Cas limite TradingView/pratique:
    - Si high == low, on force MFM = 0.0 (évite division par zéro).
    - La ligne est cumulative depuis le début (pas de période).
    """

    n = len(high)
    if n != len(low) or n != len(close) or n != len(volume):
        raise ValueError("high, low, close, volume must have same length")

    out = [math.nan] * n
    if n == 0:
        return out

    running = 0.0
    for i in range(n):
        h = float(high[i])
        l = float(low[i])
        c = float(close[i])
        v = float(volume[i])

        if _is_bad(h) or _is_bad(l) or _is_bad(c) or _is_bad(v):
            running = math.nan
            out[i] = math.nan
            continue

        denom = h - l
        if denom == 0.0:
            mfm = 0.0
        else:
            mfm = ((2.0 * c) - h - l) / denom

        mfv = mfm * v

        if _is_bad(running) or _is_bad(mfv):
            running = math.nan
            out[i] = math.nan
        else:
            running += mfv
            out[i] = running

    return out
