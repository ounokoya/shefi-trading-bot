from __future__ import annotations

import numpy as np
import pandas as pd


def add_tranche_record_close_extreme_events_df(
    df: pd.DataFrame,
    *,
    ts_col: str = "ts",
    close_col: str = "close",
    tranche_id_col: str = "tranche_id",
) -> pd.DataFrame:
    out = df.copy()

    for c in (ts_col, close_col, tranche_id_col):
        if c not in out.columns:
            raise ValueError(f"Missing required column: {c}")

    ts = pd.to_numeric(out[ts_col], errors="coerce").astype("Int64").to_numpy()
    close = pd.to_numeric(out[close_col], errors="coerce").astype(float).to_numpy()
    tranche_id = pd.to_numeric(out[tranche_id_col], errors="coerce").astype("Int64").to_numpy()

    n = int(len(out))

    record_low_close = np.full(n, np.nan, dtype=float)
    record_low_ts = np.full(n, np.nan, dtype=float)
    record_low_is_candidate = np.zeros(n, dtype=int)
    record_low_candidate_ts = np.full(n, np.nan, dtype=float)
    record_low_candidate_close = np.full(n, np.nan, dtype=float)
    record_low_is_event = np.zeros(n, dtype=int)
    record_low_event_candidate_ts = np.full(n, np.nan, dtype=float)
    record_low_event_candidate_close = np.full(n, np.nan, dtype=float)

    record_high_close = np.full(n, np.nan, dtype=float)
    record_high_ts = np.full(n, np.nan, dtype=float)
    record_high_is_candidate = np.zeros(n, dtype=int)
    record_high_candidate_ts = np.full(n, np.nan, dtype=float)
    record_high_candidate_close = np.full(n, np.nan, dtype=float)
    record_high_is_event = np.zeros(n, dtype=int)
    record_high_event_candidate_ts = np.full(n, np.nan, dtype=float)
    record_high_event_candidate_close = np.full(n, np.nan, dtype=float)

    current_tranche_id: int | None = None

    low_best = float("inf")
    low_best_ts = float("nan")
    low_cand_close = float("nan")
    low_cand_ts = float("nan")
    low_cand_triggered = False

    high_best = float("-inf")
    high_best_ts = float("nan")
    high_cand_close = float("nan")
    high_cand_ts = float("nan")
    high_cand_triggered = False

    for i in range(n):
        tid = tranche_id[i]
        if pd.isna(tid) or pd.isna(ts[i]):
            current_tranche_id = None
            low_best = float("inf")
            low_best_ts = float("nan")
            low_cand_close = float("nan")
            low_cand_ts = float("nan")
            low_cand_triggered = False

            high_best = float("-inf")
            high_best_ts = float("nan")
            high_cand_close = float("nan")
            high_cand_ts = float("nan")
            high_cand_triggered = False
            continue

        tid_i = int(tid)
        if current_tranche_id is None or tid_i != current_tranche_id:
            current_tranche_id = tid_i

            low_best = float("inf")
            low_best_ts = float("nan")
            low_cand_close = float("nan")
            low_cand_ts = float("nan")
            low_cand_triggered = False

            high_best = float("-inf")
            high_best_ts = float("nan")
            high_cand_close = float("nan")
            high_cand_ts = float("nan")
            high_cand_triggered = False

        c = float(close[i])
        if not np.isfinite(c):
            continue

        # Record-low candidate (new close minimum since tranche start)
        if c < low_best:
            low_best = c
            low_best_ts = float(ts[i])
            low_cand_close = c
            low_cand_ts = float(ts[i])
            low_cand_triggered = False
            record_low_is_candidate[i] = 1

        # Confirm record-low candidate once (first close strictly above candidate close)
        if (
            (not low_cand_triggered)
            and np.isfinite(low_cand_close)
            and np.isfinite(low_cand_ts)
            and float(ts[i]) != float(low_cand_ts)
            and c > low_cand_close
        ):
            record_low_is_event[i] = 1
            record_low_event_candidate_ts[i] = float(low_cand_ts)
            record_low_event_candidate_close[i] = float(low_cand_close)
            low_cand_triggered = True

        # Record-high candidate (new close maximum since tranche start)
        if c > high_best:
            high_best = c
            high_best_ts = float(ts[i])
            high_cand_close = c
            high_cand_ts = float(ts[i])
            high_cand_triggered = False
            record_high_is_candidate[i] = 1

        # Confirm record-high candidate once (first close strictly below candidate close)
        if (
            (not high_cand_triggered)
            and np.isfinite(high_cand_close)
            and np.isfinite(high_cand_ts)
            and float(ts[i]) != float(high_cand_ts)
            and c < high_cand_close
        ):
            record_high_is_event[i] = 1
            record_high_event_candidate_ts[i] = float(high_cand_ts)
            record_high_event_candidate_close[i] = float(high_cand_close)
            high_cand_triggered = True

        record_low_close[i] = float(low_best) if np.isfinite(low_best) else float("nan")
        record_low_ts[i] = float(low_best_ts) if np.isfinite(low_best_ts) else float("nan")
        record_low_candidate_ts[i] = float(low_cand_ts) if np.isfinite(low_cand_ts) else float("nan")
        record_low_candidate_close[i] = float(low_cand_close) if np.isfinite(low_cand_close) else float("nan")

        record_high_close[i] = float(high_best) if np.isfinite(high_best) else float("nan")
        record_high_ts[i] = float(high_best_ts) if np.isfinite(high_best_ts) else float("nan")
        record_high_candidate_ts[i] = float(high_cand_ts) if np.isfinite(high_cand_ts) else float("nan")
        record_high_candidate_close[i] = float(high_cand_close) if np.isfinite(high_cand_close) else float("nan")

    out["tranche_record_low_close"] = record_low_close
    out["tranche_record_low_ts"] = pd.Series(record_low_ts).astype("Int64")
    out["tranche_record_low_is_candidate"] = record_low_is_candidate
    out["tranche_record_low_candidate_ts"] = pd.Series(record_low_candidate_ts).astype("Int64")
    out["tranche_record_low_candidate_close"] = record_low_candidate_close
    out["tranche_record_low_is_event"] = record_low_is_event
    out["tranche_record_low_event_candidate_ts"] = pd.Series(record_low_event_candidate_ts).astype("Int64")
    out["tranche_record_low_event_candidate_close"] = record_low_event_candidate_close

    out["tranche_record_high_close"] = record_high_close
    out["tranche_record_high_ts"] = pd.Series(record_high_ts).astype("Int64")
    out["tranche_record_high_is_candidate"] = record_high_is_candidate
    out["tranche_record_high_candidate_ts"] = pd.Series(record_high_candidate_ts).astype("Int64")
    out["tranche_record_high_candidate_close"] = record_high_candidate_close
    out["tranche_record_high_is_event"] = record_high_is_event
    out["tranche_record_high_event_candidate_ts"] = pd.Series(record_high_event_candidate_ts).astype("Int64")
    out["tranche_record_high_event_candidate_close"] = record_high_event_candidate_close

    return out
