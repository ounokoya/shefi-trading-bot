from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from libs.indicators.common.is_bad import _is_bad
from libs.indicators.moving_averages.ema_tv import ema_tv


def klinger_oscillator_tv(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    volume: Sequence[float],
    *,
    fast: int = 34,
    slow: int = 55,
    signal: int = 13,
    vf_use_abs_temp: bool = True,
) -> Tuple[List[float], List[float]]:
    n = len(high)
    if n != len(low) or n != len(close) or n != len(volume):
        raise ValueError("high, low, close, volume must have same length")

    ko = [math.nan] * n
    sig = [math.nan] * n
    if n == 0:
        return ko, sig

    if int(fast) <= 0 or int(slow) <= 0 or int(signal) <= 0:
        return ko, sig

    dm = [math.nan] * n
    trend = [0] * n
    cm = [math.nan] * n
    vf = [math.nan] * n

    h0 = float(high[0])
    l0 = float(low[0])
    c0 = float(close[0])
    v0 = float(volume[0])

    if (not _is_bad(h0)) and (not _is_bad(l0)) and (not _is_bad(c0)) and (not _is_bad(v0)):
        dm0 = h0 - l0
        dm[0] = dm0
        trend[0] = 0
        cm[0] = dm0
        vf[0] = 0.0

    for i in range(1, n):
        h = float(high[i])
        l = float(low[i])
        c = float(close[i])
        v = float(volume[i])
        ph = float(high[i - 1])
        pl = float(low[i - 1])
        pc = float(close[i - 1])

        if _is_bad(h) or _is_bad(l) or _is_bad(c) or _is_bad(v) or _is_bad(ph) or _is_bad(pl) or _is_bad(pc):
            dm[i] = math.nan
            cm[i] = math.nan
            vf[i] = math.nan
            trend[i] = 0
            continue

        s0 = h + l + c
        s1 = ph + pl + pc
        t = 1 if float(s0) > float(s1) else -1
        trend[i] = int(t)

        dmi = h - l
        dm[i] = float(dmi)

        prev_cm = cm[i - 1]
        prev_dm = dm[i - 1]
        prev_trend = int(trend[i - 1])

        if _is_bad(prev_cm):
            prev_cm = float(dmi)
        if _is_bad(prev_dm):
            prev_dm = float(dmi)
        if prev_trend == 0:
            prev_trend = int(t)

        if int(t) == int(prev_trend):
            cmi = float(prev_cm) + float(dmi)
        else:
            cmi = float(prev_dm) + float(dmi)
        cm[i] = float(cmi)

        if _is_bad(cm[i]) or float(cm[i]) == 0.0:
            temp = -2.0
        else:
            raw = 2.0 * ((float(dmi) / float(cmi)) - 1.0)
            temp = abs(raw) if bool(vf_use_abs_temp) else raw

        vf[i] = float(v) * float(t) * float(temp) * 100.0

    ema_fast = ema_tv(vf, int(fast))
    ema_slow = ema_tv(vf, int(slow))

    for i in range(n):
        a = float(ema_fast[i])
        b = float(ema_slow[i])
        if _is_bad(a) or _is_bad(b):
            ko[i] = math.nan
        else:
            ko[i] = a - b

    sig = ema_tv(ko, int(signal))

    return ko, sig
