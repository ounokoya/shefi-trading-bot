from __future__ import annotations


def compute_p_new(size: float, avg_price: float, q_notional: float, d_frac: float) -> float:
    if size <= 0:
        return 0.0
    numerator = size * avg_price - q_notional * d_frac
    denom = size * (1.0 + d_frac)
    if denom <= 0 or numerator <= 0:
        return 0.0
    return numerator / denom


def compute_p_new_short(size: float, avg_price: float, q_notional: float, d_frac: float) -> float:
    if size <= 0:
        return 0.0
    denom = size * (1.0 - d_frac)
    if denom <= 0:
        return 0.0
    numerator = size * avg_price + q_notional * d_frac
    if numerator <= 0:
        return 0.0
    return numerator / denom
