from __future__ import annotations


def cycle_ref_price(c) -> float:
    if getattr(c, "closed_qty", 0.0) > 0 and getattr(c, "avg_close", 0.0) > 0:
        return float(getattr(c, "avg_close"))
    return float(getattr(c, "avg_open"))


def cycle_recompute_next_tp_price(
    c,
    *,
    is_long: bool,
    tp_d_start_pct: float,
    tp_d_step_pct: float,
) -> None:
    if int(getattr(c, "tp_index", 0)) <= 0:
        ref_price = float(getattr(c, "avg_open"))
        tp_frac = float(tp_d_start_pct) / 100.0
    else:
        ref_price = float(getattr(c, "avg_close")) if (getattr(c, "closed_qty", 0.0) > 0 and getattr(c, "avg_close", 0.0) > 0) else float(getattr(c, "avg_open"))
        tp_frac = (float(getattr(c, "tp_index")) * float(tp_d_step_pct)) / 100.0

    if ref_price <= 0:
        setattr(c, "next_tp_price", 0.0)
        return

    if is_long:
        setattr(c, "next_tp_price", ref_price * (1.0 + tp_frac))
    else:
        setattr(c, "next_tp_price", ref_price * (1.0 - tp_frac))
