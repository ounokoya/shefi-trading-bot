from __future__ import annotations

import numpy as np
import pandas as pd


def extract_tranche_perfect_series_extremes_df(
    df: pd.DataFrame,
    *,
    series_col: str,
    ts_col: str = "ts",
    tranche_id_col: str = "tranche_id",
    tranche_sign_col: str = "tranche_sign",
    tranche_start_ts_col: str = "tranche_start_ts",
    tranche_end_ts_col: str = "tranche_end_ts",
    tranche_len_col: str = "tranche_len",
) -> pd.DataFrame:
    required = [
        ts_col,
        series_col,
        tranche_id_col,
        tranche_sign_col,
        tranche_start_ts_col,
        tranche_end_ts_col,
        tranche_len_col,
    ]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    tranche_id = pd.to_numeric(df[tranche_id_col], errors="coerce").astype("Int64")
    ts = pd.to_numeric(df[ts_col], errors="coerce").astype("Int64")
    series = pd.to_numeric(df[series_col], errors="coerce")

    work = df.copy()
    work[tranche_id_col] = tranche_id
    work[ts_col] = ts
    work[series_col] = series

    work = work.dropna(subset=[tranche_id_col, ts_col, series_col]).reset_index(drop=False)

    rows: list[dict[str, object]] = []

    for tid, g in work.groupby(tranche_id_col, sort=True):
        if g.empty:
            continue

        sign_vals = g[tranche_sign_col].dropna().astype(str).tolist()
        sign = sign_vals[0] if sign_vals else None
        if sign not in ("-", "+"):
            continue

        s = pd.to_numeric(g[series_col], errors="coerce")
        if s.isna().all():
            continue

        if sign == "-":
            idx_local = int(s.idxmin())
            extreme_kind = "LOW"
            open_side = "LONG"
        else:
            idx_local = int(s.idxmax())
            extreme_kind = "HIGH"
            open_side = "SHORT"

        row = {
            "tranche_id": int(tid),
            "tranche_sign": sign,
            "tranche_start_ts": int(pd.to_numeric(g[tranche_start_ts_col], errors="coerce").dropna().iloc[0]),
            "tranche_end_ts": int(pd.to_numeric(g[tranche_end_ts_col], errors="coerce").dropna().iloc[0]),
            "tranche_len": int(pd.to_numeric(g[tranche_len_col], errors="coerce").dropna().iloc[0]),
            "series_col": str(series_col),
            "extreme_kind": extreme_kind,
            "extreme_ts": int(pd.to_numeric(g.loc[idx_local, ts_col], errors="coerce")),
            "extreme_value": float(pd.to_numeric(g.loc[idx_local, series_col], errors="coerce")),
            "extreme_row_index": int(pd.to_numeric(g.loc[idx_local, "index"], errors="coerce")),
            "open_side": open_side,
        }

        if not (np.isfinite(float(row["extreme_value"])) and isinstance(row["extreme_ts"], int)):
            continue

        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["tranche_id"]).reset_index(drop=True)

    return out
