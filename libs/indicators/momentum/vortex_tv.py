from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from libs.indicators.common.is_bad import _is_bad


def vortex_tv(
    high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int
) -> Tuple[List[float], List[float]]:
    """
    Calcul Vortex Indicator conforme TradingView (ta.vi)
    
    Formules exactes TradingView:
    - TR = max(high - low, abs(high - close[1]), abs(low - close[1]))
    - VM+ = abs(high - low[1])
    - VM- = abs(low - high[1])
    - VI+ = sum(VM+) / sum(TR) sur période
    - VI- = sum(VM-) / sum(TR) sur période
    
    Args:
        high: Prix hauts
        low: Prix bas
        close: Prix de clôture
        period: Période de calcul (défaut TradingView: 14)
    
    Returns:
        Tuple (vi_plus, vi_minus): Deux listes de valeurs VI+ et VI-
    """
    n = len(high)
    if n != len(low) or n != len(close):
        raise ValueError("high, low, close must have same length")
    if period <= 0 or period > n:
        return [math.nan] * n, [math.nan] * n

    vi_plus = [math.nan] * n
    vi_minus = [math.nan] * n

    # Calcul True Range et Vortex Movements
    tr = [math.nan] * n
    vm_plus = [math.nan] * n
    vm_minus = [math.nan] * n

    for i in range(n):
        if i == 0:
            # Première période : TR = high - low seulement
            tr[i] = float(high[i]) - float(low[i])
            vm_plus[i] = 0.0
            vm_minus[i] = 0.0
        else:
            h = float(high[i])
            l = float(low[i])
            pc = float(close[i - 1])  # previous close
            
            # True Range (formule TradingView exacte)
            r1 = h - l
            r2 = abs(h - pc)
            r3 = abs(l - pc)
            tr[i] = max(r1, r2, r3)
            
            # Vortex Movements (formule TradingView exacte)
            vm_plus[i] = abs(h - float(low[i - 1]))
            vm_minus[i] = abs(l - float(high[i - 1]))

    # Calcul des sommes glissantes et VI
    sum_tr = 0.0
    sum_vm_plus = 0.0
    sum_vm_minus = 0.0

    for i in range(n):
        if _is_bad(tr[i]) or _is_bad(vm_plus[i]) or _is_bad(vm_minus[i]):
            sum_tr = 0.0
            sum_vm_plus = 0.0
            sum_vm_minus = 0.0
            vi_plus[i] = math.nan
            vi_minus[i] = math.nan
            continue

        # Ajout valeurs courantes
        sum_tr += tr[i]
        sum_vm_plus += vm_plus[i]
        sum_vm_minus += vm_minus[i]

        # Maintien fenêtre glissante
        if i >= period:
            old_tr = tr[i - period]
            old_vm_plus = vm_plus[i - period]
            old_vm_minus = vm_minus[i - period]
            
            if not _is_bad(old_tr):
                sum_tr -= old_tr
            if not _is_bad(old_vm_plus):
                sum_vm_plus -= old_vm_plus
            if not _is_bad(old_vm_minus):
                sum_vm_minus -= old_vm_minus

        # Calcul VI seulement si fenêtre complète
        if i >= period - 1 and sum_tr > 0:
            vi_plus[i] = sum_vm_plus / sum_tr
            vi_minus[i] = sum_vm_minus / sum_tr
        else:
            vi_plus[i] = math.nan
            vi_minus[i] = math.nan

    return vi_plus, vi_minus


def vortex_signal(vi_plus: Sequence[float], vi_minus: Sequence[float]) -> List[str]:
    """
    Génération de signaux basés sur Vortex Indicator
    
    Args:
        vi_plus: Valeurs VI+
        vi_minus: Valeurs VI-
    
    Returns:
        Liste de signaux: 'BUY', 'SELL', 'HOLD', 'NEUTRAL'
    """
    if len(vi_plus) != len(vi_minus):
        raise ValueError("vi_plus and vi_minus must have same length")

    signals = []
    for i in range(len(vi_plus)):
        if _is_bad(vi_plus[i]) or _is_bad(vi_minus[i]):
            signals.append('NEUTRAL')
        elif vi_plus[i] > vi_minus[i]:
            if i > 0 and not _is_bad(vi_plus[i-1]) and not _is_bad(vi_minus[i-1]):
                if vi_plus[i-1] <= vi_minus[i-1]:
                    signals.append('BUY')  # Croisement haussier
                else:
                    signals.append('HOLD')  # Maintien tendance haussière
            else:
                signals.append('HOLD')
        elif vi_minus[i] > vi_plus[i]:
            if i > 0 and not _is_bad(vi_plus[i-1]) and not _is_bad(vi_minus[i-1]):
                if vi_minus[i-1] <= vi_plus[i-1]:
                    signals.append('SELL')  # Croisement baissier
                else:
                    signals.append('HOLD')  # Maintien tendance baissière
            else:
                signals.append('HOLD')
        else:
            signals.append('NEUTRAL')  # Égalité

    return signals
