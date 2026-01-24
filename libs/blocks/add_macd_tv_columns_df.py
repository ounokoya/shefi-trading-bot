from __future__ import annotations

import pandas as pd

from libs.indicators.momentum.macd_tv import macd_tv


def add_macd_tv_columns_df(
    df: pd.DataFrame,
    *,
    close_col: str = "close",
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    out_line_col: str = "macd_line",
    out_signal_col: str = "macd_signal",
    out_hist_col: str = "macd_hist",
) -> pd.DataFrame:
    out = df.copy()
    close = pd.to_numeric(out[close_col], errors="coerce").astype(float).tolist()
    macd_line, macd_signal, macd_hist = macd_tv(
        close,
        int(fast_period),
        int(slow_period),
        int(signal_period),
    )
    out[str(out_line_col)] = macd_line
    out[str(out_signal_col)] = macd_signal
    out[str(out_hist_col)] = macd_hist
    return out
