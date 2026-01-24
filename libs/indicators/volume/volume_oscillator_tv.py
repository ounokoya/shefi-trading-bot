from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from libs.indicators.common.is_bad import _is_bad
from libs.indicators.moving_averages.ema_tv import ema_tv


def volume_oscillator_tv(
    volume: Sequence[float], fast: int, slow: int
) -> List[float]:
    """
    Calcul Volume Oscillator conforme TradingView (différence)
    
    Formule exacte TradingView:
    oscillator = EMA(volume, fast) - EMA(volume, slow)
    TradingView utilise EMA (Exponential Moving Average) pour Volume Oscillator par défaut
    
    Args:
        volume: Série des volumes (base asset comme LINK)
        fast: Période moyenne rapide (défaut TradingView: 10)
        slow: Période moyenne lente (défaut TradingView: 30)
    
    Returns:
        Liste des valeurs de l'oscillateur (différence)
    """
    n = len(volume)
    if fast <= 0 or slow <= 0 or fast >= slow or slow > n:
        return [math.nan] * n

    # Calcul des moyennes mobiles EMA (conforme TradingView pour volume)
    fast_ma = ema_tv(volume, fast)
    slow_ma = ema_tv(volume, slow)

    # Calcul oscillateur = fast - slow
    oscillator = [math.nan] * n
    for i in range(n):
        if not _is_bad(fast_ma[i]) and not _is_bad(slow_ma[i]):
            oscillator[i] = fast_ma[i] - slow_ma[i]
        else:
            oscillator[i] = math.nan

    return oscillator


def percentage_volume_oscillator_tv(
    volume: Sequence[float], fast: int, slow: int
) -> List[float]:
    """
    Calcul Percentage Volume Oscillator (PVO) conforme TradingView
    
    Formule exacte TradingView:
    PVO = ((EMA(volume, fast) - EMA(volume, slow)) / EMA(volume, slow)) * 100
    TradingView utilise EMA (Exponential Moving Average) pour Volume Oscillator par défaut
    
    Args:
        volume: Série des volumes (base asset comme LINK)
        fast: Période moyenne rapide (défaut TradingView: 10)
        slow: Période moyenne lente (défaut TradingView: 30)
    
    Returns:
        Liste des valeurs PVO en pourcentage
    """
    n = len(volume)
    if fast <= 0 or slow <= 0 or fast >= slow or slow > n:
        return [math.nan] * n

    # Calcul des moyennes mobiles EMA (conforme TradingView pour volume)
    fast_ma = ema_tv(volume, fast)
    slow_ma = ema_tv(volume, slow)

    # Calcul PVO = ((fast - slow) / slow) * 100
    pvo = [math.nan] * n
    for i in range(n):
        if not _is_bad(fast_ma[i]) and not _is_bad(slow_ma[i]) and slow_ma[i] != 0:
            pvo[i] = ((fast_ma[i] - slow_ma[i]) / slow_ma[i]) * 100.0
        else:
            pvo[i] = math.nan

    return pvo


def volume_oscillator_with_signal_tv(
    volume: Sequence[float], fast: int, slow: int, signal: int = 9
) -> Tuple[List[float], List[float], List[float]]:
    """
    Calcul Volume Oscillator complet avec ligne de signal (style MACD)
    
    Formules TradingView:
    - Oscillator = SMA(volume, fast) - SMA(volume, slow)
    - Signal = SMA(Oscillator, signal)
    - Histogram = Oscillator - Signal
    
    Args:
        volume: Série des volumes
        fast: Période moyenne rapide (défaut: 10)
        slow: Période moyenne lente (défaut: 30)
        signal: Période ligne de signal (défaut: 9)
    
    Returns:
        Tuple (oscillator, signal_line, histogram)
    """
    n = len(volume)
    if fast <= 0 or slow <= 0 or signal <= 0 or fast >= slow or slow > n or signal > n:
        return [math.nan] * n, [math.nan] * n, [math.nan] * n

    # Calcul oscillateur principal
    oscillator = volume_oscillator_tv(volume, fast, slow)
    
    # Calcul ligne de signal (EMA de l'oscillateur)
    signal_line = ema_tv(oscillator, signal)
    
    # Calcul histogram = oscillator - signal
    histogram = [math.nan] * n
    for i in range(n):
        if not _is_bad(oscillator[i]) and not _is_bad(signal_line[i]):
            histogram[i] = oscillator[i] - signal_line[i]
        else:
            histogram[i] = math.nan

    return oscillator, signal_line, histogram


def pvo_with_signal_tv(
    volume: Sequence[float], fast: int, slow: int, signal: int = 9
) -> Tuple[List[float], List[float], List[float]]:
    """
    Calcul Percentage Volume Oscillator complet avec ligne de signal
    
    Formules TradingView:
    - PVO = ((SMA(volume, fast) - SMA(volume, slow)) / SMA(volume, slow)) * 100
    - Signal = SMA(PVO, signal)
    - Histogram = PVO - Signal
    
    Args:
        volume: Série des volumes
        fast: Période moyenne rapide (défaut: 10)
        slow: Période moyenne lente (défaut: 30)
        signal: Période ligne de signal (défaut: 9)
    
    Returns:
        Tuple (pvo, signal_line, histogram)
    """
    n = len(volume)
    if fast <= 0 or slow <= 0 or signal <= 0 or fast >= slow or slow > n or signal > n:
        return [math.nan] * n, [math.nan] * n, [math.nan] * n

    # Calcul PVO principal
    pvo = percentage_volume_oscillator_tv(volume, fast, slow)
    
    # Calcul ligne de signal (EMA du PVO)
    signal_line = ema_tv(pvo, signal)
    
    # Calcul histogram = pvo - signal
    histogram = [math.nan] * n
    for i in range(n):
        if not _is_bad(pvo[i]) and not _is_bad(signal_line[i]):
            histogram[i] = pvo[i] - signal_line[i]
        else:
            histogram[i] = math.nan

    return pvo, signal_line, histogram


def volume_oscillator_signals_tv(
    oscillator: Sequence[float], signal_line: Sequence[float]
) -> List[str]:
    """
    Génération de signaux basés sur Volume Oscillator avec ligne de signal
    
    Args:
        oscillator: Valeurs de l'oscillateur
        signal_line: Valeurs de la ligne de signal
    
    Returns:
        Liste de signaux: 'BUY', 'SELL', 'HOLD', 'NEUTRAL'
    """
    if len(oscillator) != len(signal_line):
        raise ValueError("oscillator and signal_line must have same length")

    signals = []
    for i in range(len(oscillator)):
        if _is_bad(oscillator[i]) or _is_bad(signal_line[i]):
            signals.append('NEUTRAL')
        elif oscillator[i] > signal_line[i]:
            if i > 0 and not _is_bad(oscillator[i-1]) and not _is_bad(signal_line[i-1]):
                if oscillator[i-1] <= signal_line[i-1]:
                    signals.append('BUY')  # Croisement haussier
                else:
                    signals.append('HOLD')  # Maintien au-dessus
            else:
                signals.append('HOLD')
        elif oscillator[i] < signal_line[i]:
            if i > 0 and not _is_bad(oscillator[i-1]) and not _is_bad(signal_line[i-1]):
                if oscillator[i-1] >= signal_line[i-1]:
                    signals.append('SELL')  # Croisement baissier
                else:
                    signals.append('HOLD')  # Maintien en dessous
            else:
                signals.append('HOLD')
        else:
            signals.append('NEUTRAL')  # Égalité

    return signals
