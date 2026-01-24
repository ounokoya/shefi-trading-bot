from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from libs.indicators.common.is_bad import _is_bad
from libs.indicators.moving_averages.rma_tv import rma_tv


def dmi_tv(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    period: int,
    adx_smoothing: int | None = None,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Calcul DMI (Directional Movement Index) conforme TradingView
    
    Formule exacte TradingView:
    1. Calcul True Range (TR)
    2. Calcul +DM et -DM
    3. Lissage avec Wilder (RMA)
    4. Calcul +DI et -DI
    5. Calcul ADX = RMA(|+DI - -DI| / (+DI + -DI) * 100)
    
    Args:
        high: Série des prix hauts
        low: Série des prix bas
        close: Série des prix de clôture
        period: DI Length (défaut TradingView: 14)
        adx_smoothing: ADX Smoothing (défaut TradingView: 14)
    
    Returns:
        Tuple (adx, plus_di, minus_di)
    """
    n = len(high)
    if period <= 0 or period > n:
        return [math.nan] * n, [math.nan] * n, [math.nan] * n

    adx_period = period if adx_smoothing is None else int(adx_smoothing)
    if adx_period <= 0 or adx_period > n:
        return [math.nan] * n, [math.nan] * n, [math.nan] * n

    # Calcul True Range
    tr = [math.nan] * n
    for i in range(n):
        if i == 0:
            tr[i] = float(high[i]) - float(low[i])
        else:
            hl = float(high[i]) - float(low[i])
            hc = abs(float(high[i]) - float(close[i-1]))
            lc = abs(float(low[i]) - float(close[i-1]))
            tr[i] = max(hl, hc, lc)

    # Calcul +DM et -DM
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n
    for i in range(1, n):
        up_move = float(high[i]) - float(high[i-1])
        down_move = float(low[i-1]) - float(low[i])
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0.0
            
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0.0

    # Lissage avec EMA
    tr_smooth = rma_tv(tr, period)
    plus_dm_smooth = rma_tv(plus_dm, period)
    minus_dm_smooth = rma_tv(minus_dm, period)

    # Calcul +DI et -DI
    plus_di = [math.nan] * n
    minus_di = [math.nan] * n
    for i in range(n):
        if not _is_bad(tr_smooth[i]) and tr_smooth[i] != 0:
            plus_di[i] = (plus_dm_smooth[i] / tr_smooth[i]) * 100.0
            minus_di[i] = (minus_dm_smooth[i] / tr_smooth[i]) * 100.0
        else:
            plus_di[i] = math.nan
            minus_di[i] = math.nan

    # Calcul ADX
    dx = [math.nan] * n
    for i in range(n):
        if not _is_bad(plus_di[i]) and not _is_bad(minus_di[i]):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum != 0:
                dx[i] = abs(plus_di[i] - minus_di[i]) / di_sum * 100.0
            else:
                dx[i] = 0.0
        else:
            dx[i] = math.nan

    adx = rma_tv(dx, adx_period)

    return adx, plus_di, minus_di
