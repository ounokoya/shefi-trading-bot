from __future__ import annotations

import numpy as np
import pandas as pd


def extract_tranche_perfect_close_extremes_df(
    df: pd.DataFrame,
    *,
    ts_col: str = "ts",
    close_col: str = "close",
    tranche_id_col: str = "tranche_id",
    tranche_sign_col: str = "tranche_sign",
    tranche_start_ts_col: str = "tranche_start_ts",
    tranche_end_ts_col: str = "tranche_end_ts",
    tranche_len_col: str = "tranche_len",
) -> pd.DataFrame:
    required = [
        ts_col,
        close_col,
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
    close = pd.to_numeric(df[close_col], errors="coerce")

    work = df.copy()
    work[tranche_id_col] = tranche_id
    work[ts_col] = ts
    work[close_col] = close

    work = work.dropna(subset=[tranche_id_col, ts_col, close_col]).reset_index(drop=False)

    rows: list[dict[str, object]] = []

    for tid, g in work.groupby(tranche_id_col, sort=True):
        if g.empty:
            continue

        sign_vals = g[tranche_sign_col].dropna().astype(str).tolist()
        sign = sign_vals[0] if sign_vals else None
        if sign not in ("-", "+"):
            continue

        close_g = pd.to_numeric(g[close_col], errors="coerce")
        if close_g.isna().all():
            continue

        if sign == "-":
            idx_local = int(close_g.idxmin())
            extreme_kind = "LOW"
            open_side = "LONG"
        else:
            idx_local = int(close_g.idxmax())
            extreme_kind = "HIGH"
            open_side = "SHORT"

        row = {
            "tranche_id": int(tid),
            "tranche_sign": sign,
            "tranche_start_ts": int(pd.to_numeric(g[tranche_start_ts_col], errors="coerce").dropna().iloc[0]),
            "tranche_end_ts": int(pd.to_numeric(g[tranche_end_ts_col], errors="coerce").dropna().iloc[0]),
            "tranche_len": int(pd.to_numeric(g[tranche_len_col], errors="coerce").dropna().iloc[0]),
            "extreme_kind": extreme_kind,
            "extreme_ts": int(pd.to_numeric(g.loc[idx_local, ts_col], errors="coerce")),
            "extreme_close": float(pd.to_numeric(g.loc[idx_local, close_col], errors="coerce")),
            "extreme_row_index": int(pd.to_numeric(g.loc[idx_local, "index"], errors="coerce")),
            "open_side": open_side,
        }

        if not (np.isfinite(float(row["extreme_close"])) and isinstance(row["extreme_ts"], int)):
            continue

        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["tranche_id"]).reset_index(drop=True)

    return out
