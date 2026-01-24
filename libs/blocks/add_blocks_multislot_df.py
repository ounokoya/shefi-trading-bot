from __future__ import annotations

import math
import numpy as np
import pandas as pd


def add_blocks_multislot_df(
    df: pd.DataFrame,
    *,
    ts_col: str = "ts",
    high_col: str = "high",
    low_col: str = "low",
    tranche_id_col: str = "tranche_id",
    tranche_sign_col: str = "tranche_sign",
    tranche_pos_col: str = "tranche_pos",
    tranche_start_ts_col: str = "tranche_start_ts",
    tranche_end_ts_col: str = "tranche_end_ts",
    tranche_high_col: str = "tranche_high",
    tranche_low_col: str = "tranche_low",
    tranche_high_ts_col: str = "tranche_high_ts",
    tranche_low_ts_col: str = "tranche_low_ts",
    use_tranche_candidate_extremes: bool = False,
    candidate_extremes_scope: str = "tfav_only",
    tranche_cand_extreme_ts_col: str = "tranche_cand_extreme_ts",
    tranche_cand_extreme_price_col: str = "tranche_cand_extreme_price",
    fallback_to_perfect_extremes: bool = True,
    max_slots: int = 2,
) -> pd.DataFrame:
    if max_slots != 2:
        raise ValueError("Only max_slots=2 is supported")

    out = df.copy()

    ts_series = pd.to_numeric(out[ts_col], errors="coerce")
    if ts_series.isna().any():
        raise ValueError(f"{ts_col} contains NaN after numeric coercion")
    ts = ts_series.astype("int64").to_numpy()
    high = pd.to_numeric(out[high_col], errors="coerce").astype(float).to_numpy()
    low = pd.to_numeric(out[low_col], errors="coerce").astype(float).to_numpy()

    tranche_id = pd.to_numeric(out[tranche_id_col], errors="coerce").astype("Int64").to_numpy()
    tranche_pos = pd.to_numeric(out[tranche_pos_col], errors="coerce").astype("Int64").to_numpy()
    tranche_sign = out[tranche_sign_col].to_numpy(dtype=object)

    tranche_start_ts = pd.to_numeric(out[tranche_start_ts_col], errors="coerce").astype("Int64").to_numpy()
    tranche_end_ts = pd.to_numeric(out[tranche_end_ts_col], errors="coerce").astype("Int64").to_numpy()
    tranche_high = pd.to_numeric(out[tranche_high_col], errors="coerce").astype(float).to_numpy()
    tranche_low = pd.to_numeric(out[tranche_low_col], errors="coerce").astype(float).to_numpy()
    tranche_high_ts = pd.to_numeric(out[tranche_high_ts_col], errors="coerce").astype("Int64").to_numpy()
    tranche_low_ts = pd.to_numeric(out[tranche_low_ts_col], errors="coerce").astype("Int64").to_numpy()

    tranche_cand_extreme_ts = None
    tranche_cand_extreme_price = None
    if use_tranche_candidate_extremes:
        if candidate_extremes_scope not in ("all", "tfav_only"):
            raise ValueError("candidate_extremes_scope must be one of: all, tfav_only")
        if tranche_cand_extreme_ts_col not in out.columns or tranche_cand_extreme_price_col not in out.columns:
            raise ValueError(
                f"Missing candidate extreme columns: {tranche_cand_extreme_ts_col}, {tranche_cand_extreme_price_col}"
            )
        tranche_cand_extreme_ts = (
            pd.to_numeric(out[tranche_cand_extreme_ts_col], errors="coerce").astype("Int64").to_numpy()
        )
        tranche_cand_extreme_price = pd.to_numeric(out[tranche_cand_extreme_price_col], errors="coerce").astype(float).to_numpy()

    n = int(len(out))

    block_id_1 = np.full(n, np.nan, dtype=float)
    block_side_1 = np.full(n, None, dtype=object)
    block_role_1 = np.full(n, None, dtype=object)
    is_t0_1 = np.zeros(n, dtype=int)
    is_tfav_1 = np.zeros(n, dtype=int)
    is_t1_1 = np.zeros(n, dtype=int)
    fav_pct_1 = np.full(n, np.nan, dtype=float)
    adv_pre_pct_1 = np.full(n, np.nan, dtype=float)

    block_id_2 = np.full(n, np.nan, dtype=float)
    block_side_2 = np.full(n, None, dtype=object)
    block_role_2 = np.full(n, None, dtype=object)
    is_t0_2 = np.zeros(n, dtype=int)
    is_tfav_2 = np.zeros(n, dtype=int)
    is_t1_2 = np.zeros(n, dtype=int)
    fav_pct_2 = np.full(n, np.nan, dtype=float)
    adv_pre_pct_2 = np.full(n, np.nan, dtype=float)

    event_t0_block_id = np.full(n, np.nan, dtype=float)
    event_t0_side = np.full(n, None, dtype=object)
    event_t0_role = np.full(n, None, dtype=object)
    event_t0_tranche_id = np.full(n, np.nan, dtype=float)
    event_tfav_block_id = np.full(n, np.nan, dtype=float)
    event_tfav_side = np.full(n, None, dtype=object)
    event_tfav_role = np.full(n, None, dtype=object)
    event_tfav_tranche_id = np.full(n, np.nan, dtype=float)
    event_t1_block_id = np.full(n, np.nan, dtype=float)
    event_t1_side = np.full(n, None, dtype=object)
    event_t1_role = np.full(n, None, dtype=object)
    event_t1_tranche_id = np.full(n, np.nan, dtype=float)

    tranche_first_rows = np.where(pd.Series(tranche_pos).fillna(-1).to_numpy() == 0)[0]
    tranche_summaries: list[dict[str, object]] = []

    for idx in tranche_first_rows:
        if pd.isna(tranche_id[idx]):
            continue
        tid = int(tranche_id[idx])
        tranche_summaries.append(
            {
                "tranche_id": tid,
                "sign": tranche_sign[idx],
                "start_ts": int(tranche_start_ts[idx]) if not pd.isna(tranche_start_ts[idx]) else None,
                "end_ts": int(tranche_end_ts[idx]) if not pd.isna(tranche_end_ts[idx]) else None,
                "high": float(tranche_high[idx]),
                "low": float(tranche_low[idx]),
                "high_ts": int(tranche_high_ts[idx]) if not pd.isna(tranche_high_ts[idx]) else None,
                "low_ts": int(tranche_low_ts[idx]) if not pd.isna(tranche_low_ts[idx]) else None,
                "cand_extreme_ts": int(tranche_cand_extreme_ts[idx])
                if (use_tranche_candidate_extremes and tranche_cand_extreme_ts is not None and not pd.isna(tranche_cand_extreme_ts[idx]))
                else None,
                "cand_extreme_price": float(tranche_cand_extreme_price[idx])
                if (
                    use_tranche_candidate_extremes
                    and tranche_cand_extreme_price is not None
                    and not np.isnan(tranche_cand_extreme_price[idx])
                )
                else None,
            }
        )

    for k in range(0, len(tranche_summaries) - 2):
        a = tranche_summaries[k]
        b = tranche_summaries[k + 1]
        c = tranche_summaries[k + 2]

        a_sign = a["sign"]
        b_sign = b["sign"]
        c_sign = c["sign"]

        side: str | None = None
        if a_sign == "-" and b_sign == "+" and c_sign == "-":
            side = "LONG"
        elif a_sign == "+" and b_sign == "-" and c_sign == "+":
            side = "SHORT"
        else:
            continue

        b_id = int(b["tranche_id"])
        block_id = float(b_id)

        def _pick_candidate_or_perfect(
            tr: dict[str, object],
            *,
            allow_candidate: bool,
            perfect_ts_key: str,
            perfect_price_key: str,
        ) -> tuple[int, float] | None:
            if use_tranche_candidate_extremes and allow_candidate:
                ts_v = tr.get("cand_extreme_ts")
                p_v = tr.get("cand_extreme_price")
                if ts_v is not None and p_v is not None:
                    return int(ts_v), float(p_v)
                if not fallback_to_perfect_extremes:
                    return None
            ts_v = tr.get(perfect_ts_key)
            p_v = tr.get(perfect_price_key)
            if ts_v is None or p_v is None:
                return None
            return int(ts_v), float(p_v)

        if side == "LONG":
            t0 = _pick_candidate_or_perfect(
                a,
                allow_candidate=candidate_extremes_scope == "all",
                perfect_ts_key="low_ts",
                perfect_price_key="low",
            )
            tfav = _pick_candidate_or_perfect(
                b,
                allow_candidate=True,
                perfect_ts_key="high_ts",
                perfect_price_key="high",
            )
            t1 = _pick_candidate_or_perfect(
                c,
                allow_candidate=candidate_extremes_scope == "all",
                perfect_ts_key="low_ts",
                perfect_price_key="low",
            )
        else:
            t0 = _pick_candidate_or_perfect(
                a,
                allow_candidate=candidate_extremes_scope == "all",
                perfect_ts_key="high_ts",
                perfect_price_key="high",
            )
            tfav = _pick_candidate_or_perfect(
                b,
                allow_candidate=True,
                perfect_ts_key="low_ts",
                perfect_price_key="low",
            )
            t1 = _pick_candidate_or_perfect(
                c,
                allow_candidate=candidate_extremes_scope == "all",
                perfect_ts_key="high_ts",
                perfect_price_key="high",
            )

        if t0 is None or tfav is None or t1 is None:
            continue

        t0_ts, p0 = t0
        tfav_ts, pfav = tfav
        t1_ts, p1 = t1

        if not (t0_ts <= tfav_ts <= t1_ts):
            continue

        def _set_event(
            event_ts: int,
            *,
            role: str,
            arr_id: np.ndarray,
            arr_side: np.ndarray,
            arr_role: np.ndarray,
            arr_tranche_id: np.ndarray,
        ) -> None:
            idx = int(np.searchsorted(ts, int(event_ts), side="left"))
            if idx < 0 or idx >= n:
                return
            if int(ts[idx]) != int(event_ts):
                return
            if math.isnan(arr_id[idx]) or float(block_id) > float(arr_id[idx]):
                arr_id[idx] = float(block_id)
                arr_side[idx] = side
                arr_role[idx] = role
                if not pd.isna(tranche_id[idx]):
                    arr_tranche_id[idx] = float(tranche_id[idx])

        _set_event(
            t0_ts,
            role="avant",
            arr_id=event_t0_block_id,
            arr_side=event_t0_side,
            arr_role=event_t0_role,
            arr_tranche_id=event_t0_tranche_id,
        )
        _set_event(
            tfav_ts,
            role="milieu",
            arr_id=event_tfav_block_id,
            arr_side=event_tfav_side,
            arr_role=event_tfav_role,
            arr_tranche_id=event_tfav_tranche_id,
        )
        _set_event(
            t1_ts,
            role="apres",
            arr_id=event_t1_block_id,
            arr_side=event_t1_side,
            arr_role=event_t1_role,
            arr_tranche_id=event_t1_tranche_id,
        )

        if p0 == 0.0:
            continue

        fav_pct = (pfav - p0) / p0 if side == "LONG" else (p0 - pfav) / p0

        mask_pre = (ts >= t0_ts) & (ts <= tfav_ts)
        if side == "LONG":
            min_pre = float(np.nanmin(low[mask_pre]))
            adv_pre_pct = (p0 - min_pre) / p0
        else:
            max_pre = float(np.nanmax(high[mask_pre]))
            adv_pre_pct = (max_pre - p0) / p0

        a_id = int(a["tranche_id"])
        c_id = int(c["tranche_id"])

        start_idx = int(np.searchsorted(ts, t0_ts, side="left"))
        end_idx = int(np.searchsorted(ts, t1_ts, side="right")) - 1
        if start_idx < 0 or start_idx >= n or end_idx < 0 or end_idx >= n or start_idx > end_idx:
            continue

        for i in range(start_idx, end_idx + 1):
            if pd.isna(tranche_id[i]):
                continue

            role = None
            if int(tranche_id[i]) == a_id:
                role = "avant"
            elif int(tranche_id[i]) == b_id:
                role = "milieu"
            elif int(tranche_id[i]) == c_id:
                role = "apres"

            if role is None:
                continue

            def _fill_slot(slot: int) -> None:
                if slot == 1:
                    block_id_1[i] = block_id
                    block_side_1[i] = side
                    block_role_1[i] = role
                    fav_pct_1[i] = fav_pct
                    adv_pre_pct_1[i] = adv_pre_pct
                    is_t0_1[i] = 1 if int(ts[i]) == t0_ts else 0
                    is_tfav_1[i] = 1 if int(ts[i]) == tfav_ts else 0
                    is_t1_1[i] = 1 if int(ts[i]) == t1_ts else 0
                else:
                    block_id_2[i] = block_id
                    block_side_2[i] = side
                    block_role_2[i] = role
                    fav_pct_2[i] = fav_pct
                    adv_pre_pct_2[i] = adv_pre_pct
                    is_t0_2[i] = 1 if int(ts[i]) == t0_ts else 0
                    is_tfav_2[i] = 1 if int(ts[i]) == tfav_ts else 0
                    is_t1_2[i] = 1 if int(ts[i]) == t1_ts else 0

            def _event_score_existing(slot: int) -> int:
                if slot == 1:
                    if is_t1_1[i] == 1:
                        return 100
                    if is_t0_1[i] == 1:
                        return 90
                    if is_tfav_1[i] == 1:
                        return 50
                    return 0
                else:
                    if is_t1_2[i] == 1:
                        return 100
                    if is_t0_2[i] == 1:
                        return 90
                    if is_tfav_2[i] == 1:
                        return 50
                    return 0

            def _event_score_new() -> int:
                if int(ts[i]) == t1_ts:
                    return 100
                if int(ts[i]) == t0_ts:
                    return 90
                if int(ts[i]) == tfav_ts:
                    return 50
                return 0

            if math.isnan(block_id_1[i]):
                _fill_slot(1)
            elif math.isnan(block_id_2[i]):
                _fill_slot(2)
            else:
                # Si 3 blocs se superposent, on conserve d'abord ceux qui portent
                # un événement sur la bougie (t1 > t0 > tfav), puis départage par récence.
                s1 = _event_score_existing(1)
                s2 = _event_score_existing(2)
                sn = _event_score_new()

                k1 = (int(s1), float(block_id_1[i]))
                k2 = (int(s2), float(block_id_2[i]))
                kn = (int(sn), float(block_id))

                # slot "le moins prioritaire" = score plus faible, ou même score mais plus ancien
                if k1 <= k2:
                    worst_slot = 1
                    worst_key = k1
                else:
                    worst_slot = 2
                    worst_key = k2

                if kn > worst_key:
                    _fill_slot(worst_slot)

    # Ordre canonique: slot1 = bloc le plus récent.
    swap_idx = np.where((~np.isnan(block_id_1)) & (~np.isnan(block_id_2)) & (block_id_2 > block_id_1))[0]
    for i in swap_idx:
        block_id_1[i], block_id_2[i] = block_id_2[i], block_id_1[i]
        block_side_1[i], block_side_2[i] = block_side_2[i], block_side_1[i]
        block_role_1[i], block_role_2[i] = block_role_2[i], block_role_1[i]
        is_t0_1[i], is_t0_2[i] = is_t0_2[i], is_t0_1[i]
        is_tfav_1[i], is_tfav_2[i] = is_tfav_2[i], is_tfav_1[i]
        is_t1_1[i], is_t1_2[i] = is_t1_2[i], is_t1_1[i]
        fav_pct_1[i], fav_pct_2[i] = fav_pct_2[i], fav_pct_1[i]
        adv_pre_pct_1[i], adv_pre_pct_2[i] = adv_pre_pct_2[i], adv_pre_pct_1[i]

    out["block_id_1"] = pd.Series(block_id_1).astype("Int64")
    out["block_side_1"] = block_side_1
    out["block_role_1"] = block_role_1
    out["is_t0_1"] = is_t0_1
    out["is_tfav_1"] = is_tfav_1
    out["is_t1_1"] = is_t1_1
    out["fav_pct_1"] = fav_pct_1
    out["adv_pre_pct_1"] = adv_pre_pct_1

    out["block_id_2"] = pd.Series(block_id_2).astype("Int64")
    out["block_side_2"] = block_side_2
    out["block_role_2"] = block_role_2
    out["is_t0_2"] = is_t0_2
    out["is_tfav_2"] = is_tfav_2
    out["is_t1_2"] = is_t1_2
    out["fav_pct_2"] = fav_pct_2
    out["adv_pre_pct_2"] = adv_pre_pct_2

    out["event_t0_block_id"] = pd.Series(event_t0_block_id).astype("Int64")
    out["event_t0_side"] = event_t0_side
    out["event_t0_role"] = event_t0_role
    out["event_t0_tranche_id"] = pd.Series(event_t0_tranche_id).astype("Int64")
    out["event_tfav_block_id"] = pd.Series(event_tfav_block_id).astype("Int64")
    out["event_tfav_side"] = event_tfav_side
    out["event_tfav_role"] = event_tfav_role
    out["event_tfav_tranche_id"] = pd.Series(event_tfav_tranche_id).astype("Int64")
    out["event_t1_block_id"] = pd.Series(event_t1_block_id).astype("Int64")
    out["event_t1_side"] = event_t1_side
    out["event_t1_role"] = event_t1_role
    out["event_t1_tranche_id"] = pd.Series(event_t1_tranche_id).astype("Int64")

    return out
