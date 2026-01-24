from __future__ import annotations

import math

import numpy as np
import pandas as pd


def quadratic_shape_features_series(
    s: pd.Series,
    *,
    window: int,
    prefix: str,
) -> pd.DataFrame:
    if window < 3:
        raise ValueError("window must be >= 3")

    x = pd.to_numeric(s, errors="coerce").astype(float).to_numpy()
    n = int(x.shape[0])

    s_end = np.full(n, np.nan, dtype=float)
    kappa = np.full(n, np.nan, dtype=float)
    kappa_abs = np.full(n, np.nan, dtype=float)
    t_star = np.full(n, np.nan, dtype=float)
    t_star_in_window = np.zeros(n, dtype=int)
    t_star_dist_to_end = np.full(n, np.nan, dtype=float)
    x_hat_t_star = np.full(n, np.nan, dtype=float)
    delta_vertex = np.full(n, np.nan, dtype=float)
    r2 = np.full(n, np.nan, dtype=float)
    r_end = np.full(n, np.nan, dtype=float)
    shape_state = np.zeros(n, dtype=int)

    t = np.arange(window, dtype=float)
    A = np.column_stack([t * t, t, np.ones(window, dtype=float)])
    pinv = np.linalg.pinv(A)

    for i in range(window - 1, n):
        w = x[i - window + 1 : i + 1]
        if np.isnan(w).any() or math.isnan(float(x[i])):
            continue

        coeff = pinv @ w
        a = float(coeff[0])
        b = float(coeff[1])
        c = float(coeff[2])

        y_hat = A @ coeff

        scale = float(np.std(w))
        if scale == 0.0:
            scale = float("nan")

        s_end_i = 2.0 * a * float(window - 1) + b
        kappa_i = 2.0 * a

        if not math.isnan(scale):
            s_end_i = s_end_i / scale
            kappa_i = kappa_i / scale

        s_end[i] = s_end_i
        kappa[i] = kappa_i
        kappa_abs[i] = abs(kappa_i) if not math.isnan(kappa_i) else float("nan")

        t_star_i = float("nan")
        in_window = 0
        x_hat_ts = float("nan")
        delta_v = float("nan")
        dist_end = float("nan")
        if a != 0.0:
            t_star_i = -b / (2.0 * a)
            if 0.0 <= t_star_i <= float(window - 1):
                in_window = 1
            dist_end = float(window - 1) - t_star_i
            x_hat_ts = a * (t_star_i**2) + b * t_star_i + c
            delta_v = float(w[-1]) - x_hat_ts
            if not math.isnan(scale):
                delta_v = delta_v / scale

        t_star[i] = t_star_i
        t_star_in_window[i] = int(in_window)
        t_star_dist_to_end[i] = dist_end
        x_hat_t_star[i] = x_hat_ts
        delta_vertex[i] = delta_v

        ss_res = float(np.sum((w - y_hat) ** 2))
        y_mean = float(np.mean(w))
        ss_tot = float(np.sum((w - y_mean) ** 2))
        if ss_tot == 0.0:
            r2[i] = 1.0
        else:
            r2[i] = 1.0 - (ss_res / ss_tot)

        r_end_i = float(w[-1]) - float(y_hat[-1])
        if not math.isnan(scale):
            r_end_i = r_end_i / scale
        r_end[i] = r_end_i

        if s_end_i > 0.0 and kappa_i > 0.0:
            shape_state[i] = 1
        elif s_end_i > 0.0 and kappa_i < 0.0:
            shape_state[i] = 2
        elif s_end_i < 0.0 and kappa_i < 0.0:
            shape_state[i] = 3
        elif s_end_i < 0.0 and kappa_i > 0.0:
            shape_state[i] = 4
        else:
            shape_state[i] = 0

    out = {
        f"{prefix}_w{window}_s_end": s_end,
        f"{prefix}_w{window}_kappa": kappa,
        f"{prefix}_w{window}_kappa_abs": kappa_abs,
        f"{prefix}_w{window}_t_star": t_star,
        f"{prefix}_w{window}_t_star_in_window": t_star_in_window,
        f"{prefix}_w{window}_t_star_dist_to_end": t_star_dist_to_end,
        f"{prefix}_w{window}_x_hat_t_star": x_hat_t_star,
        f"{prefix}_w{window}_delta_vertex": delta_vertex,
        f"{prefix}_w{window}_r2": r2,
        f"{prefix}_w{window}_r_end": r_end,
        f"{prefix}_w{window}_shape_state": shape_state,
    }

    return pd.DataFrame(out, index=s.index)
