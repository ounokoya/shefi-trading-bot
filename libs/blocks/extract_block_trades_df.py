from __future__ import annotations

import pandas as pd


def extract_block_trades_df(
    df: pd.DataFrame,
    *,
    ts_col: str = "ts",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_base = [ts_col, open_col, high_col, low_col, close_col]
    for c in required_base:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    if "macd_hist" not in df.columns:
        raise ValueError("Missing required column: macd_hist")
    if "tranche_id" not in df.columns:
        raise ValueError("Missing required column: tranche_id")

    has_event_cols = (
        "event_t0_block_id" in df.columns
        and "event_tfav_block_id" in df.columns
        and "event_t1_block_id" in df.columns
        and "event_t0_side" in df.columns
        and "event_tfav_side" in df.columns
        and "event_t1_side" in df.columns
    )

    base = df[[ts_col, open_col, high_col, low_col, close_col, "macd_hist", "tranche_id"]].copy()
    base[ts_col] = pd.to_numeric(base[ts_col], errors="coerce")
    if base[ts_col].isna().any():
        raise ValueError(f"{ts_col} contains NaN after numeric coercion")
    base[open_col] = pd.to_numeric(base[open_col], errors="coerce")
    base[high_col] = pd.to_numeric(base[high_col], errors="coerce")
    base[low_col] = pd.to_numeric(base[low_col], errors="coerce")
    base[close_col] = pd.to_numeric(base[close_col], errors="coerce")
    base["macd_hist"] = pd.to_numeric(base["macd_hist"], errors="coerce")
    base["tranche_id"] = pd.to_numeric(base["tranche_id"], errors="coerce").astype("Int64")

    ts_all = base[ts_col].astype("int64").to_numpy()
    open_all = base[open_col].astype(float).to_numpy()
    high_all = base[high_col].astype(float).to_numpy()
    low_all = base[low_col].astype(float).to_numpy()
    close_all = base[close_col].astype(float).to_numpy()
    hist_all = base["macd_hist"].astype(float).to_numpy()
    tranche_all = base["tranche_id"].to_numpy()

    ts_to_idx = {int(t): i for i, t in enumerate(ts_all.tolist())}

    def _slice_by_ts(start_ts: int, end_ts: int) -> slice:
        start_i = int(pd.Index(ts_all).searchsorted(int(start_ts), side="left"))
        end_i = int(pd.Index(ts_all).searchsorted(int(end_ts), side="right"))
        return slice(start_i, end_i)

    frames: list[pd.DataFrame] = []
    for slot in (1, 2):
        bid = f"block_id_{slot}"
        if bid not in df.columns:
            continue

        tmp = df[
            [
                ts_col,
                open_col,
                high_col,
                low_col,
                close_col,
                bid,
                f"block_side_{slot}",
                f"block_role_{slot}",
                f"is_t0_{slot}",
                f"is_tfav_{slot}",
                f"is_t1_{slot}",
                f"fav_pct_{slot}",
                f"adv_pre_pct_{slot}",
                "macd_hist",
                "tranche_id",
            ]
        ].copy()

        tmp.rename(
            columns={
                bid: "block_id",
                f"block_side_{slot}": "side",
                f"block_role_{slot}": "role",
                f"is_t0_{slot}": "is_t0",
                f"is_tfav_{slot}": "is_tfav",
                f"is_t1_{slot}": "is_t1",
                f"fav_pct_{slot}": "fav_pct",
                f"adv_pre_pct_{slot}": "adv_pre_pct",
            },
            inplace=True,
        )

        tmp["slot"] = slot
        frames.append(tmp)

    if not frames:
        raise ValueError("No block slots found (expected block_id_1 and/or block_id_2)")

    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df.dropna(subset=["block_id"]).copy()

    long_df["block_id"] = pd.to_numeric(long_df["block_id"], errors="coerce").astype("Int64")

    for flag in ("is_t0", "is_tfav", "is_t1"):
        long_df[flag] = pd.to_numeric(long_df[flag], errors="coerce").fillna(0).astype(int)

    trades_rows: list[dict[str, object]] = []
    issues_rows: list[dict[str, object]] = []

    for block_id, g in long_df.groupby("block_id", dropna=True):
        if pd.isna(block_id):
            continue

        g2 = g.sort_values(ts_col)
        sides = [s for s in g2["side"].dropna().unique().tolist() if s in ("LONG", "SHORT")]
        side = sides[0] if sides else None

        t0_ts = None
        tfav_ts = None
        t1_ts = None

        t0_count = 0
        tfav_count = 0
        t1_count = 0

        if has_event_cols:
            b = int(block_id)

            m0 = df["event_t0_block_id"] == b
            mf = df["event_tfav_block_id"] == b
            m1 = df["event_t1_block_id"] == b

            t0_count = int(m0.sum())
            tfav_count = int(mf.sum())
            t1_count = int(m1.sum())

            if t0_count >= 1:
                t0_ts = int(pd.to_numeric(df.loc[m0, ts_col], errors="coerce").iloc[0])
                if side is None:
                    s = df.loc[m0, "event_t0_side"].dropna().tolist()
                    side = str(s[0]) if s else None

            if tfav_count >= 1:
                tfav_ts = int(pd.to_numeric(df.loc[mf, ts_col], errors="coerce").iloc[0])
                if side is None:
                    s = df.loc[mf, "event_tfav_side"].dropna().tolist()
                    side = str(s[0]) if s else None

            if t1_count >= 1:
                t1_ts = int(pd.to_numeric(df.loc[m1, ts_col], errors="coerce").iloc[0])
                if side is None:
                    s = df.loc[m1, "event_t1_side"].dropna().tolist()
                    side = str(s[0]) if s else None
        else:
            t0_rows = g2[g2["is_t0"] == 1]
            tfav_rows = g2[g2["is_tfav"] == 1]
            t1_rows = g2[g2["is_t1"] == 1]

            t0_count = int(len(t0_rows))
            tfav_count = int(len(tfav_rows))
            t1_count = int(len(t1_rows))

            t0_ts = int(t0_rows[ts_col].iloc[0]) if t0_count >= 1 else None
            tfav_ts = int(tfav_rows[ts_col].iloc[0]) if tfav_count >= 1 else None
            t1_ts = int(t1_rows[ts_col].iloc[0]) if t1_count >= 1 else None

        # adv_pre_pct et fav_pct sont des ratios (0.05 = 5%).
        fav_pct = None
        adv_pre_pct = None

        issue = None
        if side is None:
            issue = "missing_side"
        elif len(sides) > 1:
            issue = "multiple_sides"
        elif t0_count != 1 or tfav_count != 1 or t1_count != 1:
            issue = "missing_or_multiple_flags"

        entry_px = None
        entry_open = None
        entry_close = None
        fav_px = None
        exit_px = None
        exit_open = None
        exit_close = None
        exit_pct = None

        dd_max_to_fav_pct = None
        dd_max_trade_pct = None
        dd_max_to_fav_pct_entry_open = None
        dd_max_to_fav_pct_entry_close = None
        dd_max_trade_pct_entry_open = None
        dd_max_trade_pct_entry_close = None
        n_hist_sign_changes_t0_to_tfav = None

        cross_ts = None
        cross_entry_open = None
        cross_entry_close = None
        cross_fav_pct = None
        cross_exit_pct_close = None

        cross_dd_max_to_fav_pct_entry_open = None
        cross_dd_max_to_fav_pct_entry_close = None
        cross_dd_max_trade_pct_entry_open = None
        cross_dd_max_trade_pct_entry_close = None

        if side in ("LONG", "SHORT") and t0_ts is not None and tfav_ts is not None and t1_ts is not None:
            i0 = ts_to_idx.get(int(t0_ts))
            if i0 is None:
                i0 = None
            if i0 is not None:
                entry_open = float(open_all[i0])
                entry_close = float(close_all[i0])

            if side == "LONG":
                # p0/pfav/p1 = extrêmes des tranches selon spec
                if i0 is not None:
                    entry_px = float(low_all[i0])
            else:
                if i0 is not None:
                    entry_px = float(high_all[i0])

            if entry_px is not None and entry_px != 0.0:
                # ADV_pre (spec) = adverse max sur [t0..tfav] rapporté à p0.
                sl_pre = _slice_by_ts(int(t0_ts), int(tfav_ts))
                if side == "LONG":
                    adv_pre_pct = float((entry_px - float(low_all[sl_pre].min())) / entry_px)
                else:
                    adv_pre_pct = float((float(high_all[sl_pre].max()) - entry_px) / entry_px)

            # fav_px (extrême à tfav)
            if tfav_ts is not None:
                ifv = ts_to_idx.get(int(tfav_ts))
                if ifv is not None:
                    fav_px = float(high_all[ifv]) if side == "LONG" else float(low_all[ifv])

            # exit (extrême à t1)
            if t1_ts is not None:
                i1 = ts_to_idx.get(int(t1_ts))
                if i1 is not None:
                    exit_open = float(open_all[i1])
                    exit_close = float(close_all[i1])
                    exit_px = float(low_all[i1]) if side == "LONG" else float(high_all[i1])

            if entry_px is not None and exit_px is not None and entry_px != 0.0:
                exit_pct = (exit_px - entry_px) / entry_px if side == "LONG" else (entry_px - exit_px) / entry_px

            if entry_px is not None and fav_px is not None and entry_px != 0.0:
                fav_pct = (fav_px - entry_px) / entry_px if side == "LONG" else (entry_px - fav_px) / entry_px

            if side == "LONG":
                pass
            else:
                pass

            sl_t0_tfav = _slice_by_ts(int(t0_ts), int(tfav_ts))
            sl_t0_t1 = _slice_by_ts(int(t0_ts), int(t1_ts))

            if sl_t0_tfav.stop > sl_t0_tfav.start and entry_px is not None and entry_px != 0.0:
                if side == "LONG":
                    dd_max_to_fav_pct = float((entry_px - float(low_all[sl_t0_tfav].min())) / entry_px)
                else:
                    dd_max_to_fav_pct = float((float(high_all[sl_t0_tfav].max()) - entry_px) / entry_px)

            if sl_t0_tfav.stop > sl_t0_tfav.start and entry_open is not None and entry_open != 0.0:
                if side == "LONG":
                    dd_max_to_fav_pct_entry_open = float((entry_open - float(low_all[sl_t0_tfav].min())) / entry_open)
                else:
                    dd_max_to_fav_pct_entry_open = float((float(high_all[sl_t0_tfav].max()) - entry_open) / entry_open)

            if sl_t0_tfav.stop > sl_t0_tfav.start and entry_close is not None and entry_close != 0.0:
                if side == "LONG":
                    dd_max_to_fav_pct_entry_close = float(
                        (entry_close - float(low_all[sl_t0_tfav].min())) / entry_close
                    )
                else:
                    dd_max_to_fav_pct_entry_close = float(
                        (float(high_all[sl_t0_tfav].max()) - entry_close) / entry_close
                    )

                hs = hist_all[sl_t0_tfav]
                eff_sign: list[int] = []
                prev = 0
                for v in hs:
                    if pd.isna(v) or v == 0.0:
                        s = prev
                    else:
                        s = 1 if v > 0.0 else -1
                    eff_sign.append(s)
                    if s != 0:
                        prev = s
                changes = 0
                prev_s = 0
                for s in eff_sign:
                    if s == 0:
                        continue
                    if prev_s == 0:
                        prev_s = s
                        continue
                    if s != prev_s:
                        changes += 1
                        prev_s = s
                n_hist_sign_changes_t0_to_tfav = int(changes)

            if sl_t0_t1.stop > sl_t0_t1.start and entry_px is not None and entry_px != 0.0:
                if side == "LONG":
                    dd_max_trade_pct = float((entry_px - float(low_all[sl_t0_t1].min())) / entry_px)
                else:
                    dd_max_trade_pct = float((float(high_all[sl_t0_t1].max()) - entry_px) / entry_px)

            if sl_t0_t1.stop > sl_t0_t1.start and entry_open is not None and entry_open != 0.0:
                if side == "LONG":
                    dd_max_trade_pct_entry_open = float((entry_open - float(low_all[sl_t0_t1].min())) / entry_open)
                else:
                    dd_max_trade_pct_entry_open = float((float(high_all[sl_t0_t1].max()) - entry_open) / entry_open)

            if sl_t0_t1.stop > sl_t0_t1.start and entry_close is not None and entry_close != 0.0:
                if side == "LONG":
                    dd_max_trade_pct_entry_close = float((entry_close - float(low_all[sl_t0_t1].min())) / entry_close)
                else:
                    dd_max_trade_pct_entry_close = float((float(high_all[sl_t0_t1].max()) - entry_close) / entry_close)

            # proxy "entrée au croisement" = début de tranche milieu (tranche_id == block_id)
            b = int(block_id)
            m_mid = base["tranche_id"] == b
            if int(m_mid.sum()) > 0:
                cross_ts = int(base.loc[m_mid, ts_col].min())
                ic = ts_to_idx.get(int(cross_ts))
                if ic is not None:
                    cross_entry_open = float(open_all[ic])
                    cross_entry_close = float(close_all[ic])

                if cross_entry_close != 0.0:
                    if side == "LONG":
                        cross_fav_pct = float((fav_px - cross_entry_close) / cross_entry_close)
                        cross_exit_pct_close = float((exit_close - cross_entry_close) / cross_entry_close)
                    else:
                        cross_fav_pct = float((cross_entry_close - fav_px) / cross_entry_close)
                        cross_exit_pct_close = float((cross_entry_close - exit_close) / cross_entry_close)

                sl_cross_tfav = _slice_by_ts(int(cross_ts), int(tfav_ts))
                sl_cross_t1 = _slice_by_ts(int(cross_ts), int(t1_ts))

                if sl_cross_tfav.stop > sl_cross_tfav.start and cross_entry_open is not None and cross_entry_open != 0.0:
                    if side == "LONG":
                        cross_dd_max_to_fav_pct_entry_open = float(
                            (cross_entry_open - float(low_all[sl_cross_tfav].min())) / cross_entry_open
                        )
                    else:
                        cross_dd_max_to_fav_pct_entry_open = float(
                            (float(high_all[sl_cross_tfav].max()) - cross_entry_open) / cross_entry_open
                        )

                if sl_cross_tfav.stop > sl_cross_tfav.start and cross_entry_close is not None and cross_entry_close != 0.0:
                    if side == "LONG":
                        cross_dd_max_to_fav_pct_entry_close = float(
                            (cross_entry_close - float(low_all[sl_cross_tfav].min())) / cross_entry_close
                        )
                    else:
                        cross_dd_max_to_fav_pct_entry_close = float(
                            (float(high_all[sl_cross_tfav].max()) - cross_entry_close) / cross_entry_close
                        )

                if sl_cross_t1.stop > sl_cross_t1.start and cross_entry_open is not None and cross_entry_open != 0.0:
                    if side == "LONG":
                        cross_dd_max_trade_pct_entry_open = float(
                            (cross_entry_open - float(low_all[sl_cross_t1].min())) / cross_entry_open
                        )
                    else:
                        cross_dd_max_trade_pct_entry_open = float(
                            (float(high_all[sl_cross_t1].max()) - cross_entry_open) / cross_entry_open
                        )

                if sl_cross_t1.stop > sl_cross_t1.start and cross_entry_close is not None and cross_entry_close != 0.0:
                    if side == "LONG":
                        cross_dd_max_trade_pct_entry_close = float(
                            (cross_entry_close - float(low_all[sl_cross_t1].min())) / cross_entry_close
                        )
                    else:
                        cross_dd_max_trade_pct_entry_close = float(
                            (float(high_all[sl_cross_t1].max()) - cross_entry_close) / cross_entry_close
                        )

        # rôle counts (approx) sur fenêtre [t0..t1]
        role_counts = "{}"
        n_rows_window = None
        if t0_ts is not None and t1_ts is not None:
            sl = _slice_by_ts(int(t0_ts), int(t1_ts))
            n_rows_window = int(sl.stop - sl.start)
            b = int(block_id)
            if sl.stop > sl.start:
                vals = tranche_all[sl]
                rc = {
                    "avant": int((vals == (b - 1)).sum()),
                    "milieu": int((vals == b).sum()),
                    "apres": int((vals == (b + 1)).sum()),
                }
                role_counts = str({k: v for k, v in rc.items() if v > 0})

        trades_rows.append(
            {
                "block_id": int(block_id),
                "side": side,
                "block_start_ts": int(t0_ts) if t0_ts is not None else int(g2[ts_col].min()),
                "block_end_ts": int(t1_ts) if t1_ts is not None else int(g2[ts_col].max()),
                "t0_ts": t0_ts,
                "tfav_ts": tfav_ts,
                "t1_ts": t1_ts,
                "entry_px": entry_px,
                "entry_open": entry_open,
                "entry_close": entry_close,
                "fav_px": fav_px,
                "exit_px": exit_px,
                "exit_open": exit_open,
                "exit_close": exit_close,
                "exit_pct": exit_pct,
                "dd_max_to_fav_pct": dd_max_to_fav_pct,
                "dd_max_trade_pct": dd_max_trade_pct,
                "dd_max_to_fav_pct_entry_open": dd_max_to_fav_pct_entry_open,
                "dd_max_to_fav_pct_entry_close": dd_max_to_fav_pct_entry_close,
                "dd_max_trade_pct_entry_open": dd_max_trade_pct_entry_open,
                "dd_max_trade_pct_entry_close": dd_max_trade_pct_entry_close,
                "n_hist_sign_changes_t0_to_tfav": n_hist_sign_changes_t0_to_tfav,
                "cross_ts": cross_ts,
                "cross_entry_open": cross_entry_open,
                "cross_entry_close": cross_entry_close,
                "cross_fav_pct": cross_fav_pct,
                "cross_exit_pct_close": cross_exit_pct_close,
                "cross_dd_max_to_fav_pct_entry_open": cross_dd_max_to_fav_pct_entry_open,
                "cross_dd_max_to_fav_pct_entry_close": cross_dd_max_to_fav_pct_entry_close,
                "cross_dd_max_trade_pct_entry_open": cross_dd_max_trade_pct_entry_open,
                "cross_dd_max_trade_pct_entry_close": cross_dd_max_trade_pct_entry_close,
                "fav_pct": fav_pct,
                "adv_pre_pct": adv_pre_pct,
                "n_rows_mapped": int(len(g2)),
                "role_counts": role_counts,
            }
        )

        if issue is not None:
            issues_rows.append(
                {
                    "block_id": int(block_id),
                    "issue": issue,
                    "side_values": str(g2["side"].dropna().unique().tolist()),
                    "t0_count": t0_count,
                    "tfav_count": tfav_count,
                    "t1_count": t1_count,
                }
            )

    trades_df = pd.DataFrame(trades_rows)
    if "block_id" in trades_df.columns:
        trades_df = trades_df.sort_values("block_id").reset_index(drop=True)

    issues_df = pd.DataFrame(issues_rows)
    if "block_id" in issues_df.columns:
        issues_df = issues_df.sort_values("block_id").reset_index(drop=True)

    return trades_df, issues_df
