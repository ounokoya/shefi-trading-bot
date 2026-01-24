from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.blocks.add_macd_tv_columns_df import add_macd_tv_columns_df  # noqa: E402
from libs.blocks.segment_macd_hist_tranches_df import segment_macd_hist_tranches_df  # noqa: E402
from libs.data_loader import get_crypto_data  # noqa: E402
from libs.indicators.asi import asi  # noqa: E402
from libs.indicators.momentum.cci_tv import cci_tv  # noqa: E402
from libs.indicators.momentum.dmi_tv import dmi_tv  # noqa: E402
from libs.indicators.volume.pvi_tv import pvi_tv  # noqa: E402
from libs.indicators.volume.pvt_tv import pvt_tv  # noqa: E402


@dataclass(frozen=True)
class TrancheEvent:
    flow_indicator: str
    symbol: str
    tranche_id: int
    tranche_sign: str
    tranche_start_i: int
    tranche_end_i: int
    event_i: int
    event_ts: int
    event_dt: str
    price_extreme_i_in_tranche: int
    price_extreme_close_in_tranche: float
    price_extreme_close: float
    pvt_extreme_in_tranche: float
    pvi_extreme_in_tranche: float | None
    asi_extreme_in_tranche: float | None
    dmi_category: str
    dmi_filter: str
    dmi_force_brute: str


def _dx_from_di(plus_di: float, minus_di: float) -> float | None:
    try:
        p = float(plus_di)
        m = float(minus_di)
    except Exception:
        return None
    if not math.isfinite(p) or not math.isfinite(m):
        return None
    s = p + m
    if s == 0.0:
        return 0.0
    return abs(p - m) / s * 100.0


def _dmi_category_and_filter_at_event(
    *,
    adx: pd.Series,
    plus_di: pd.Series,
    minus_di: pd.Series,
    event_i: int,
    adx_threshold: float,
) -> tuple[str, str]:
    a = adx.iloc[int(event_i)]
    if pd.isna(a) or (not math.isfinite(float(a))):
        category = "plat"
        a_val: float | None = None
    else:
        a_val = float(a)
        category = "plat" if float(a_val) < float(adx_threshold) else "tendenciel"

    dx = _dx_from_di(plus_di.iloc[int(event_i)], minus_di.iloc[int(event_i)])
    if dx is None or (not math.isfinite(float(dx))) or a_val is None or (not math.isfinite(float(a_val))):
        dmi_filter = "respiration"
    else:
        dmi_filter = "impulsion" if float(dx) > float(a_val) else "respiration"

    return str(category), str(dmi_filter)


def _dmi_force_brute_at_event(*, plus_di: pd.Series, minus_di: pd.Series, event_i: int) -> str:
    p = plus_di.iloc[int(event_i)]
    m = minus_di.iloc[int(event_i)]
    dx = _dx_from_di(p, m)
    if dx is None or pd.isna(p) or pd.isna(m):
        return "petite_force"

    try:
        p_f = float(p)
        m_f = float(m)
    except Exception:
        return "petite_force"
    if (not math.isfinite(p_f)) or (not math.isfinite(m_f)) or (not math.isfinite(float(dx))):
        return "petite_force"

    lo = min(p_f, m_f)
    hi = max(p_f, m_f)
    if float(dx) > float(hi):
        return "tres_fort"
    if float(dx) < float(lo):
        return "petite_force"
    return "moyen_fort"


def _cci_confluence_ok(
    *,
    tranche_sign: str,
    start_i: int,
    end_i: int,
    cci_cols: list[str],
    cci_level: float,
    df: pd.DataFrame,
) -> bool:
    if not cci_cols:
        return True

    w = df.iloc[int(start_i) : int(end_i) + 1]
    for col in cci_cols:
        s = pd.to_numeric(w[col], errors="coerce")
        if tranche_sign == "-":
            if not bool((s <= -float(cci_level)).any()):
                return False
        else:
            if not bool((s >= float(cci_level)).any()):
                return False
    return True


def _build_tranche_events(
    *,
    df: pd.DataFrame,
    symbol: str,
    flow_indicator: str,
    cci_cols: list[str],
    cci_level: float,
    require_dmi_category: str,
    require_dmi_filter: str,
    adx_threshold: float,
) -> list[TrancheEvent]:
    out: list[TrancheEvent] = []

    close_s = pd.to_numeric(df["close"], errors="coerce").astype(float)
    pvt_s = pd.to_numeric(df["pvt"], errors="coerce").astype(float)
    pvi_s = pd.to_numeric(df["pvi"], errors="coerce").astype(float) if "pvi" in df.columns else None
    asi_s = pd.to_numeric(df["asi"], errors="coerce").astype(float) if "asi" in df.columns else None

    adx_s = pd.to_numeric(df.get("adx"), errors="coerce").astype(float)
    plus_di_s = pd.to_numeric(df.get("plus_di"), errors="coerce").astype(float)
    minus_di_s = pd.to_numeric(df.get("minus_di"), errors="coerce").astype(float)

    require_cat = str(require_dmi_category).strip().lower()
    require_cat_enabled = bool(require_cat) and require_cat not in {"any", "*", "all"}

    require_filter = str(require_dmi_filter).strip().lower()
    require_filter_enabled = bool(require_filter) and require_filter not in {"any", "*", "all"}

    tranche_ids = pd.to_numeric(df["tranche_id"], errors="coerce").astype("Int64")
    if not len(tranche_ids.dropna()):
        return out

    for tid, w in df.groupby("tranche_id", sort=True):
        if pd.isna(tid):
            continue
        tid_i = int(tid)
        start_i = int(w.index.min())
        end_i = int(w.index.max())
        tranche_sign = str(w.iloc[0]["tranche_sign"])

        w_close = close_s.iloc[int(start_i) : int(end_i) + 1]
        if tranche_sign == "-":
            price_extreme_i_in_tranche = int(w_close.idxmin())
        else:
            price_extreme_i_in_tranche = int(w_close.idxmax())
        price_extreme_close_in_tranche = float(close_s.iloc[int(price_extreme_i_in_tranche)])

        w_pvt = pvt_s.iloc[int(start_i) : int(end_i) + 1]
        if tranche_sign == "-":
            pvt_extreme_i = int(w_pvt.idxmin())
            pvt_extreme = float(w_pvt.min())
        else:
            pvt_extreme_i = int(w_pvt.idxmax())
            pvt_extreme = float(w_pvt.max())

        asi_extreme: float | None = None
        asi_extreme_i: int | None = None
        if flow_indicator in {"asi", "asi_pvt"}:
            if asi_s is None:
                continue
            w_asi = asi_s.iloc[int(start_i) : int(end_i) + 1]
            if tranche_sign == "-":
                asi_extreme_i = int(w_asi.idxmin())
                asi_extreme = float(w_asi.min())
            else:
                asi_extreme_i = int(w_asi.idxmax())
                asi_extreme = float(w_asi.max())

        if flow_indicator == "asi":
            if asi_extreme_i is None:
                continue
            event_i = int(asi_extreme_i)
        elif flow_indicator == "asi_pvt":
            if asi_extreme_i is None:
                continue
            if int(asi_extreme_i) != int(pvt_extreme_i):
                continue
            event_i = int(pvt_extreme_i)
        else:
            event_i = int(price_extreme_i_in_tranche)

        price_extreme_close = float(close_s.iloc[int(event_i)])

        pvi_extreme: float | None = None
        if flow_indicator == "pvt_pvi":
            if pvi_s is None:
                continue
            w_pvi = pvi_s.iloc[int(start_i) : int(end_i) + 1]
            if tranche_sign == "-":
                pvi_extreme = float(w_pvi.min())
            else:
                pvi_extreme = float(w_pvi.max())

        if not _cci_confluence_ok(
            tranche_sign=tranche_sign,
            start_i=start_i,
            end_i=end_i,
            cci_cols=cci_cols,
            cci_level=float(cci_level),
            df=df,
        ):
            continue

        dmi_cat, dmi_filter = _dmi_category_and_filter_at_event(
            adx=adx_s,
            plus_di=plus_di_s,
            minus_di=minus_di_s,
            event_i=event_i,
            adx_threshold=float(adx_threshold),
        )

        dmi_force_brute = _dmi_force_brute_at_event(plus_di=plus_di_s, minus_di=minus_di_s, event_i=event_i)
        if require_cat_enabled and str(dmi_cat).lower() != require_cat:
            continue
        if require_filter_enabled and str(dmi_filter).lower() != require_filter:
            continue

        ts = int(df.iloc[int(event_i)]["ts"]) if "ts" in df.columns and not pd.isna(df.iloc[int(event_i)]["ts"]) else int(event_i)
        dt = str(df.iloc[int(event_i)]["dt"]) if "dt" in df.columns else ""

        out.append(
            TrancheEvent(
                flow_indicator=str(flow_indicator),
                symbol=str(symbol),
                tranche_id=int(tid_i),
                tranche_sign=str(tranche_sign),
                tranche_start_i=int(start_i),
                tranche_end_i=int(end_i),
                event_i=int(event_i),
                event_ts=int(ts),
                event_dt=str(dt),
                price_extreme_i_in_tranche=int(price_extreme_i_in_tranche),
                price_extreme_close_in_tranche=float(price_extreme_close_in_tranche),
                price_extreme_close=float(price_extreme_close),
                pvt_extreme_in_tranche=float(pvt_extreme),
                pvi_extreme_in_tranche=float(pvi_extreme) if pvi_extreme is not None else None,
                asi_extreme_in_tranche=float(asi_extreme) if asi_extreme is not None else None,
                dmi_category=str(dmi_cat),
                dmi_filter=str(dmi_filter),
                dmi_force_brute=str(dmi_force_brute),
            )
        )

    out.sort(key=lambda e: int(e.event_i))
    return out


def _accum_long_ok(a: TrancheEvent, b: TrancheEvent) -> bool:
    if not (float(b.price_extreme_close) > float(a.price_extreme_close)):
        return False
    if str(a.flow_indicator) == "asi":
        if a.asi_extreme_in_tranche is None or b.asi_extreme_in_tranche is None:
            return False
        if not (float(b.asi_extreme_in_tranche) > float(a.asi_extreme_in_tranche)):
            return False
        return True

    if str(a.flow_indicator) == "asi_pvt":
        if not (float(b.pvt_extreme_in_tranche) > float(a.pvt_extreme_in_tranche)):
            return False
        if a.asi_extreme_in_tranche is None or b.asi_extreme_in_tranche is None:
            return False
        if not (float(b.asi_extreme_in_tranche) > float(a.asi_extreme_in_tranche)):
            return False
        return True

    if not (float(b.pvt_extreme_in_tranche) > float(a.pvt_extreme_in_tranche)):
        return False
    if a.pvi_extreme_in_tranche is not None and b.pvi_extreme_in_tranche is not None:
        if not (float(b.pvi_extreme_in_tranche) > float(a.pvi_extreme_in_tranche)):
            return False
    return True


def _accum_short_ok(a: TrancheEvent, b: TrancheEvent) -> bool:
    if not (float(b.price_extreme_close) < float(a.price_extreme_close)):
        return False
    if str(a.flow_indicator) == "asi":
        if a.asi_extreme_in_tranche is None or b.asi_extreme_in_tranche is None:
            return False
        if not (float(b.asi_extreme_in_tranche) < float(a.asi_extreme_in_tranche)):
            return False
        return True

    if str(a.flow_indicator) == "asi_pvt":
        if not (float(b.pvt_extreme_in_tranche) < float(a.pvt_extreme_in_tranche)):
            return False
        if a.asi_extreme_in_tranche is None or b.asi_extreme_in_tranche is None:
            return False
        if not (float(b.asi_extreme_in_tranche) < float(a.asi_extreme_in_tranche)):
            return False
        return True

    if not (float(b.pvt_extreme_in_tranche) < float(a.pvt_extreme_in_tranche)):
        return False
    if a.pvi_extreme_in_tranche is not None and b.pvi_extreme_in_tranche is not None:
        if not (float(b.pvi_extreme_in_tranche) < float(a.pvi_extreme_in_tranche)):
            return False
    return True


def _build_signals(
    *,
    events: list[TrancheEvent],
    extreme_seq_len: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    events_short = [e for e in events if e.tranche_sign == "-"]
    events_long = [e for e in events if e.tranche_sign == "+"]

    n = int(extreme_seq_len)

    for k in range(n - 1, len(events_short)):
        seq = events_short[k - n + 1 : k + 1]
        ok = True
        for a, b in zip(seq[:-1], seq[1:]):
            if not _accum_long_ok(a, b):
                ok = False
                break
        if ok:
            last = seq[-1]
            rows.append(
                {
                    "symbol": last.symbol,
                    "side": "LONG",
                    "kind": "ACCUM_LONG",
                    "seq_len": int(n),
                    "signal_tranche_id": int(last.tranche_id),
                    "signal_i": int(last.event_i),
                    "signal_ts": int(last.event_ts),
                    "signal_dt": str(last.event_dt),
                    "signal_price": float(last.price_extreme_close),
                    "signal_pvt": float(last.pvt_extreme_in_tranche),
                    "signal_pvi": float(last.pvi_extreme_in_tranche) if last.pvi_extreme_in_tranche is not None else pd.NA,
                    "signal_asi": float(last.asi_extreme_in_tranche) if last.asi_extreme_in_tranche is not None else pd.NA,
                }
            )

    for k in range(n - 1, len(events_long)):
        seq = events_long[k - n + 1 : k + 1]
        ok = True
        for a, b in zip(seq[:-1], seq[1:]):
            if not _accum_short_ok(a, b):
                ok = False
                break
        if ok:
            last = seq[-1]
            rows.append(
                {
                    "symbol": last.symbol,
                    "side": "SHORT",
                    "kind": "ACCUM_SHORT",
                    "seq_len": int(n),
                    "signal_tranche_id": int(last.tranche_id),
                    "signal_i": int(last.event_i),
                    "signal_ts": int(last.event_ts),
                    "signal_dt": str(last.event_dt),
                    "signal_price": float(last.price_extreme_close),
                    "signal_pvt": float(last.pvt_extreme_in_tranche),
                    "signal_pvi": float(last.pvi_extreme_in_tranche) if last.pvi_extreme_in_tranche is not None else pd.NA,
                    "signal_asi": float(last.asi_extreme_in_tranche) if last.asi_extreme_in_tranche is not None else pd.NA,
                }
            )

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["symbol", "signal_i", "kind"]).reset_index(drop=True)
    return df


def _derive_structure_events(
    *,
    events: list[TrancheEvent],
    signals: pd.DataFrame,
    enable_rebottom: bool,
    max_bottom_breaks: int,
) -> pd.DataFrame:
    if not len(events) or signals is None or not len(signals):
        return pd.DataFrame()

    by_symbol: dict[str, list[TrancheEvent]] = {}
    for e in events:
        by_symbol.setdefault(str(e.symbol), []).append(e)

    sig_by_symbol: dict[str, pd.DataFrame] = {}
    for sym, g in signals.groupby("symbol"):
        sig_by_symbol[str(sym)] = g.sort_values(["signal_i", "kind"]).reset_index(drop=True)

    out_rows: list[dict[str, object]] = []

    for sym, evs in by_symbol.items():
        evs_sorted = sorted(evs, key=lambda x: int(x.event_i))
        sigs = sig_by_symbol.get(str(sym))
        if sigs is None or not len(sigs):
            continue

        flow_indicator = str(evs_sorted[0].flow_indicator) if evs_sorted else "pvt"

        bottom_ref_pvt: float | None = None
        bottom_ref_pvi: float | None = None
        bottom_ref_asi: float | None = None
        top_ref_pvt: float | None = None
        top_ref_pvi: float | None = None
        top_ref_asi: float | None = None

        bottom_flow_ref: float | None = None
        top_flow_ref: float | None = None
        mean_bottom_prices: list[float] = []
        mean_top_prices: list[float] = []

        breaks_used_long = 0
        breaks_used_short = 0

        sig_set = {(int(r["signal_i"]), str(r["kind"])) for _, r in sigs.iterrows()}

        for e in evs_sorted:
            if (int(e.event_i), "ACCUM_LONG") in sig_set:
                mean_bottom_prices.append(float(e.price_extreme_close))
                if bottom_flow_ref is None:
                    if flow_indicator == "asi":
                        if e.asi_extreme_in_tranche is None:
                            continue
                        bottom_ref_asi = float(e.asi_extreme_in_tranche)
                        bottom_flow_ref = float(e.asi_extreme_in_tranche)
                    elif flow_indicator == "pvt_pvi":
                        if e.pvi_extreme_in_tranche is None:
                            continue
                        bottom_ref_pvt = float(e.pvt_extreme_in_tranche)
                        bottom_ref_pvi = float(e.pvi_extreme_in_tranche)
                        bottom_flow_ref = float(e.pvt_extreme_in_tranche)
                    elif flow_indicator == "asi_pvt":
                        if e.asi_extreme_in_tranche is None:
                            continue
                        bottom_ref_pvt = float(e.pvt_extreme_in_tranche)
                        bottom_ref_asi = float(e.asi_extreme_in_tranche)
                        bottom_flow_ref = float(e.pvt_extreme_in_tranche)
                    else:
                        bottom_ref_pvt = float(e.pvt_extreme_in_tranche)
                        bottom_flow_ref = float(e.pvt_extreme_in_tranche)

            if (int(e.event_i), "ACCUM_SHORT") in sig_set:
                mean_top_prices.append(float(e.price_extreme_close))
                if top_flow_ref is None:
                    if flow_indicator == "asi":
                        if e.asi_extreme_in_tranche is None:
                            continue
                        top_ref_asi = float(e.asi_extreme_in_tranche)
                        top_flow_ref = float(e.asi_extreme_in_tranche)
                    elif flow_indicator == "pvt_pvi":
                        if e.pvi_extreme_in_tranche is None:
                            continue
                        top_ref_pvt = float(e.pvt_extreme_in_tranche)
                        top_ref_pvi = float(e.pvi_extreme_in_tranche)
                        top_flow_ref = float(e.pvt_extreme_in_tranche)
                    elif flow_indicator == "asi_pvt":
                        if e.asi_extreme_in_tranche is None:
                            continue
                        top_ref_pvt = float(e.pvt_extreme_in_tranche)
                        top_ref_asi = float(e.asi_extreme_in_tranche)
                        top_flow_ref = float(e.pvt_extreme_in_tranche)
                    else:
                        top_ref_pvt = float(e.pvt_extreme_in_tranche)
                        top_flow_ref = float(e.pvt_extreme_in_tranche)

            if bottom_flow_ref is not None and e.tranche_sign == "-":
                pvt_ext = float(e.pvt_extreme_in_tranche)
                pvi_ext = float(e.pvi_extreme_in_tranche) if e.pvi_extreme_in_tranche is not None else None
                asi_ext = float(e.asi_extreme_in_tranche) if e.asi_extreme_in_tranche is not None else None

                # cassure valide uniquement si un extrême est formé au-delà de la référence
                if flow_indicator == "asi":
                    if bottom_ref_asi is None or asi_ext is None:
                        continue
                    is_break = float(asi_ext) < float(bottom_ref_asi)
                    flow_extreme = float(asi_ext)
                    flow_ref = float(bottom_ref_asi)
                elif flow_indicator == "pvt_pvi":
                    if bottom_ref_pvt is None or bottom_ref_pvi is None or pvi_ext is None:
                        continue
                    is_break = (float(pvt_ext) < float(bottom_ref_pvt)) and (float(pvi_ext) < float(bottom_ref_pvi))
                    flow_extreme = float(pvt_ext)
                    flow_ref = float(bottom_ref_pvt)
                elif flow_indicator == "asi_pvt":
                    if bottom_ref_pvt is None or bottom_ref_asi is None or asi_ext is None:
                        continue
                    is_break = (float(pvt_ext) < float(bottom_ref_pvt)) and (float(asi_ext) < float(bottom_ref_asi))
                    flow_extreme = float(pvt_ext)
                    flow_ref = float(bottom_ref_pvt)
                else:
                    if bottom_ref_pvt is None:
                        continue
                    is_break = float(pvt_ext) < float(bottom_ref_pvt)
                    flow_extreme = float(pvt_ext)
                    flow_ref = float(bottom_ref_pvt)

                if bool(is_break):
                    out_rows.append(
                        {
                            "symbol": sym,
                            "side": "LONG",
                            "kind": "BOTTOM_BREAK",
                            "event_i": int(e.event_i),
                            "event_dt": str(e.event_dt),
                            "flow_extreme": float(flow_extreme),
                            "flow_ref": float(flow_ref),
                            "flow_extreme_pvt": float(pvt_ext),
                            "flow_extreme_pvi": float(pvi_ext) if pvi_ext is not None else pd.NA,
                            "flow_extreme_asi": float(asi_ext) if asi_ext is not None else pd.NA,
                            "flow_ref_pvt": float(bottom_ref_pvt) if bottom_ref_pvt is not None else pd.NA,
                            "flow_ref_pvi": float(bottom_ref_pvi) if bottom_ref_pvi is not None else pd.NA,
                            "flow_ref_asi": float(bottom_ref_asi) if bottom_ref_asi is not None else pd.NA,
                            "mean_price": float(sum(mean_bottom_prices) / len(mean_bottom_prices)) if mean_bottom_prices else pd.NA,
                        }
                    )
                    if bool(enable_rebottom) and int(breaks_used_long) < int(max_bottom_breaks):
                        breaks_used_long += 1
                        if flow_indicator == "asi":
                            bottom_ref_asi = float(asi_ext)
                            bottom_flow_ref = float(asi_ext)
                        elif flow_indicator == "pvt_pvi":
                            bottom_ref_pvt = float(pvt_ext)
                            bottom_ref_pvi = float(pvi_ext) if pvi_ext is not None else bottom_ref_pvi
                            bottom_flow_ref = float(pvt_ext)
                        elif flow_indicator == "asi_pvt":
                            bottom_ref_pvt = float(pvt_ext)
                            bottom_ref_asi = float(asi_ext) if asi_ext is not None else bottom_ref_asi
                            bottom_flow_ref = float(pvt_ext)
                        else:
                            bottom_ref_pvt = float(pvt_ext)
                            bottom_flow_ref = float(pvt_ext)
                        mean_bottom_prices = [float(e.price_extreme_close)]
                        out_rows.append(
                            {
                                "symbol": sym,
                                "side": "LONG",
                                "kind": "REBOTTOM",
                                "event_i": int(e.event_i),
                                "event_dt": str(e.event_dt),
                                "flow_extreme": float(flow_extreme),
                                "flow_ref": float(bottom_flow_ref) if bottom_flow_ref is not None else pd.NA,
                                "flow_extreme_pvt": float(pvt_ext),
                                "flow_extreme_pvi": float(pvi_ext) if pvi_ext is not None else pd.NA,
                                "flow_extreme_asi": float(asi_ext) if asi_ext is not None else pd.NA,
                                "flow_ref_pvt": float(bottom_ref_pvt) if bottom_ref_pvt is not None else pd.NA,
                                "flow_ref_pvi": float(bottom_ref_pvi) if bottom_ref_pvi is not None else pd.NA,
                                "flow_ref_asi": float(bottom_ref_asi) if bottom_ref_asi is not None else pd.NA,
                                "mean_price": float(sum(mean_bottom_prices) / len(mean_bottom_prices)) if mean_bottom_prices else pd.NA,
                            }
                        )

            if top_flow_ref is not None and e.tranche_sign == "+":
                pvt_ext = float(e.pvt_extreme_in_tranche)
                pvi_ext = float(e.pvi_extreme_in_tranche) if e.pvi_extreme_in_tranche is not None else None
                asi_ext = float(e.asi_extreme_in_tranche) if e.asi_extreme_in_tranche is not None else None

                # cassure valide uniquement si un extrême est formé au-delà de la référence
                if flow_indicator == "asi":
                    if top_ref_asi is None or asi_ext is None:
                        continue
                    is_break = float(asi_ext) > float(top_ref_asi)
                    flow_extreme = float(asi_ext)
                    flow_ref = float(top_ref_asi)
                elif flow_indicator == "pvt_pvi":
                    if top_ref_pvt is None or top_ref_pvi is None or pvi_ext is None:
                        continue
                    is_break = (float(pvt_ext) > float(top_ref_pvt)) and (float(pvi_ext) > float(top_ref_pvi))
                    flow_extreme = float(pvt_ext)
                    flow_ref = float(top_ref_pvt)
                elif flow_indicator == "asi_pvt":
                    if top_ref_pvt is None or top_ref_asi is None or asi_ext is None:
                        continue
                    is_break = (float(pvt_ext) > float(top_ref_pvt)) and (float(asi_ext) > float(top_ref_asi))
                    flow_extreme = float(pvt_ext)
                    flow_ref = float(top_ref_pvt)
                else:
                    if top_ref_pvt is None:
                        continue
                    is_break = float(pvt_ext) > float(top_ref_pvt)
                    flow_extreme = float(pvt_ext)
                    flow_ref = float(top_ref_pvt)

                if bool(is_break):
                    out_rows.append(
                        {
                            "symbol": sym,
                            "side": "SHORT",
                            "kind": "BOTTOM_BREAK",
                            "event_i": int(e.event_i),
                            "event_dt": str(e.event_dt),
                            "flow_extreme": float(flow_extreme),
                            "flow_ref": float(flow_ref),
                            "flow_extreme_pvt": float(pvt_ext),
                            "flow_extreme_pvi": float(pvi_ext) if pvi_ext is not None else pd.NA,
                            "flow_extreme_asi": float(asi_ext) if asi_ext is not None else pd.NA,
                            "flow_ref_pvt": float(top_ref_pvt) if top_ref_pvt is not None else pd.NA,
                            "flow_ref_pvi": float(top_ref_pvi) if top_ref_pvi is not None else pd.NA,
                            "flow_ref_asi": float(top_ref_asi) if top_ref_asi is not None else pd.NA,
                            "mean_price": float(sum(mean_top_prices) / len(mean_top_prices)) if mean_top_prices else pd.NA,
                        }
                    )
                    if bool(enable_rebottom) and int(breaks_used_short) < int(max_bottom_breaks):
                        breaks_used_short += 1
                        if flow_indicator == "asi":
                            top_ref_asi = float(asi_ext)
                            top_flow_ref = float(asi_ext)
                        elif flow_indicator == "pvt_pvi":
                            top_ref_pvt = float(pvt_ext)
                            top_ref_pvi = float(pvi_ext) if pvi_ext is not None else top_ref_pvi
                            top_flow_ref = float(pvt_ext)
                        elif flow_indicator == "asi_pvt":
                            top_ref_pvt = float(pvt_ext)
                            top_ref_asi = float(asi_ext) if asi_ext is not None else top_ref_asi
                            top_flow_ref = float(pvt_ext)
                        else:
                            top_ref_pvt = float(pvt_ext)
                            top_flow_ref = float(pvt_ext)
                        mean_top_prices = [float(e.price_extreme_close)]
                        out_rows.append(
                            {
                                "symbol": sym,
                                "side": "SHORT",
                                "kind": "REBOTTOM",
                                "event_i": int(e.event_i),
                                "event_dt": str(e.event_dt),
                                "flow_extreme": float(flow_extreme),
                                "flow_ref": float(top_flow_ref) if top_flow_ref is not None else pd.NA,
                                "flow_extreme_pvt": float(pvt_ext),
                                "flow_extreme_pvi": float(pvi_ext) if pvi_ext is not None else pd.NA,
                                "flow_extreme_asi": float(asi_ext) if asi_ext is not None else pd.NA,
                                "flow_ref_pvt": float(top_ref_pvt) if top_ref_pvt is not None else pd.NA,
                                "flow_ref_pvi": float(top_ref_pvi) if top_ref_pvi is not None else pd.NA,
                                "flow_ref_asi": float(top_ref_asi) if top_ref_asi is not None else pd.NA,
                                "mean_price": float(sum(mean_top_prices) / len(mean_top_prices)) if mean_top_prices else pd.NA,
                            }
                        )

        if mean_bottom_prices:
            out_rows.append(
                {
                    "symbol": sym,
                    "side": "LONG",
                    "kind": "MEAN_PRICE",
                    "event_i": int(evs_sorted[-1].event_i),
                    "event_dt": str(evs_sorted[-1].event_dt),
                    "flow_extreme": pd.NA,
                    "flow_ref": float(bottom_flow_ref) if bottom_flow_ref is not None else pd.NA,
                    "flow_extreme_pvt": pd.NA,
                    "flow_extreme_pvi": pd.NA,
                    "flow_extreme_asi": pd.NA,
                    "flow_ref_pvt": float(bottom_ref_pvt) if bottom_ref_pvt is not None else pd.NA,
                    "flow_ref_pvi": float(bottom_ref_pvi) if bottom_ref_pvi is not None else pd.NA,
                    "flow_ref_asi": float(bottom_ref_asi) if bottom_ref_asi is not None else pd.NA,
                    "mean_price": float(sum(mean_bottom_prices) / len(mean_bottom_prices)),
                }
            )
        if mean_top_prices:
            out_rows.append(
                {
                    "symbol": sym,
                    "side": "SHORT",
                    "kind": "MEAN_PRICE",
                    "event_i": int(evs_sorted[-1].event_i),
                    "event_dt": str(evs_sorted[-1].event_dt),
                    "flow_extreme": pd.NA,
                    "flow_ref": float(top_flow_ref) if top_flow_ref is not None else pd.NA,
                    "flow_extreme_pvt": pd.NA,
                    "flow_extreme_pvi": pd.NA,
                    "flow_extreme_asi": pd.NA,
                    "flow_ref_pvt": float(top_ref_pvt) if top_ref_pvt is not None else pd.NA,
                    "flow_ref_pvi": float(top_ref_pvi) if top_ref_pvi is not None else pd.NA,
                    "flow_ref_asi": float(top_ref_asi) if top_ref_asi is not None else pd.NA,
                    "mean_price": float(sum(mean_top_prices) / len(mean_top_prices)),
                }
            )

    df = pd.DataFrame(out_rows)
    if len(df):
        df = df.sort_values(["symbol", "event_i", "kind"]).reset_index(drop=True)
    return df


def _compute_indicators(
    df: pd.DataFrame,
    *,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    cci_periods: list[int],
    dmi_period: int,
    dmi_adx_smoothing: int,
    flow_indicator: str,
) -> pd.DataFrame:
    out = df.copy()

    out = add_macd_tv_columns_df(out, fast_period=int(macd_fast), slow_period=int(macd_slow), signal_period=int(macd_signal))
    out = segment_macd_hist_tranches_df(out, extremes_on="close")

    high_s = pd.to_numeric(out["high"], errors="coerce").astype(float)
    low_s = pd.to_numeric(out["low"], errors="coerce").astype(float)
    close_s = pd.to_numeric(out["close"], errors="coerce").astype(float)
    open_s = pd.to_numeric(out["open"], errors="coerce").astype(float)
    high = high_s.tolist()
    low = low_s.tolist()
    close = close_s.tolist()
    volume = pd.to_numeric(out["volume"], errors="coerce").astype(float).tolist()

    for p in cci_periods:
        out[f"cci_{int(p)}"] = cci_tv(high, low, close, int(p))

    adx, plus_di, minus_di = dmi_tv(high, low, close, int(dmi_period), adx_smoothing=int(dmi_adx_smoothing))
    out["adx"] = adx
    out["plus_di"] = plus_di
    out["minus_di"] = minus_di

    out["pvt"] = pvt_tv(close, volume)
    if flow_indicator == "pvt_pvi":
        out["pvi"] = pvi_tv(close, volume)

    if flow_indicator in {"asi", "asi_pvt"}:
        asi_df = asi(high=high_s, low=low_s, close=close_s, open_=open_s, limit_move_value="auto", limit_move_pct=0.10, offset=0)
        out["asi"] = asi_df["ASI"]
        out["si"] = asi_df["SI"]

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="ETHUSDT,SOLUSDT,ADAUSDT,AVAXUSDT,BTCUSDT,LINKUSDT")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--year-start", default="2020-01-01")
    ap.add_argument("--year-end", default="2025-12-31")

    ap.add_argument("--flow-indicator", choices=["pvt", "pvt_pvi", "asi", "asi_pvt"], default="pvt")

    ap.add_argument("--macd-fast", type=int, default=12)
    ap.add_argument("--macd-slow", type=int, default=26)
    ap.add_argument("--macd-signal", type=int, default=9)

    ap.add_argument("--cci-fast", type=int, default=14)
    ap.add_argument("--cci-medium", type=int, default=30)
    ap.add_argument("--cci-slow", type=int, default=0)
    ap.add_argument("--confluence-mode", choices=["0", "2", "3"], default="2")
    ap.add_argument("--cci-level", type=float, default=100.0)

    ap.add_argument("--dmi-period", type=int, default=14)
    ap.add_argument("--dmi-adx-smoothing", type=int, default=14)
    ap.add_argument("--dmi-adx-threshold", type=float, default=20.0)
    ap.add_argument("--require-dmi-category", default="any")
    ap.add_argument("--require-dmi-filter", default="any")

    ap.add_argument("--extreme-seq-len", type=int, default=2)
    ap.add_argument("--enable-rebottom", action="store_true")
    ap.add_argument("--max-bottom-breaks", type=int, default=0)

    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "processed" / "accum_spec_reports"))
    args = ap.parse_args()

    seq_len = int(args.extreme_seq_len)
    if seq_len not in {2, 3, 4}:
        raise SystemExit("extreme-seq-len must be 2, 3, or 4")

    max_bottom_breaks = int(args.max_bottom_breaks)
    if max_bottom_breaks < 0:
        raise SystemExit("max-bottom-breaks must be >= 0")

    confluence_mode = str(args.confluence_mode).strip()
    cci_periods: list[int] = []
    cci_cols: list[str] = []
    if confluence_mode == "2":
        cci_periods = [int(args.cci_fast), int(args.cci_medium)]
    elif confluence_mode == "3":
        if int(args.cci_slow) <= 0:
            raise SystemExit("confluence-mode=3 requires --cci-slow > 0")
        cci_periods = [int(args.cci_fast), int(args.cci_medium), int(args.cci_slow)]
    else:
        cci_periods = []

    cci_periods = [p for p in cci_periods if int(p) > 0]
    cci_cols = [f"cci_{int(p)}" for p in cci_periods]

    out_dir = Path(str(args.out_dir)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = [s.strip() for s in str(args.symbols).split(",") if s.strip()]

    all_events: list[dict[str, object]] = []
    all_signals: list[pd.DataFrame] = []
    all_structure: list[pd.DataFrame] = []

    for sym in symbols:
        df = get_crypto_data(
            symbol=str(sym),
            start_date=str(args.year_start),
            end_date=str(args.year_end),
            timeframe=str(args.interval),
            project_root=PROJECT_ROOT,
        )
        if df is None or not len(df):
            continue

        if "dt" not in df.columns and "ts" in df.columns:
            df["dt"] = pd.to_datetime(pd.to_numeric(df["ts"], errors="coerce").astype("Int64"), unit="ms", utc=True).dt.strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )

        df = _compute_indicators(
            df,
            macd_fast=int(args.macd_fast),
            macd_slow=int(args.macd_slow),
            macd_signal=int(args.macd_signal),
            cci_periods=cci_periods,
            dmi_period=int(args.dmi_period),
            dmi_adx_smoothing=int(args.dmi_adx_smoothing),
            flow_indicator=str(args.flow_indicator),
        )

        events = _build_tranche_events(
            df=df,
            symbol=str(sym),
            flow_indicator=str(args.flow_indicator),
            cci_cols=cci_cols,
            cci_level=float(args.cci_level),
            require_dmi_category=str(args.require_dmi_category),
            require_dmi_filter=str(args.require_dmi_filter),
            adx_threshold=float(args.dmi_adx_threshold),
        )

        ev_rows = [
            {
                "flow_indicator": str(e.flow_indicator),
                "symbol": e.symbol,
                "tranche_id": int(e.tranche_id),
                "tranche_sign": str(e.tranche_sign),
                "tranche_start_i": int(e.tranche_start_i),
                "tranche_end_i": int(e.tranche_end_i),
                "event_i": int(e.event_i),
                "event_ts": int(e.event_ts),
                "event_dt": str(e.event_dt),
                "price_extreme_i_in_tranche": int(e.price_extreme_i_in_tranche),
                "price_extreme_close_in_tranche": float(e.price_extreme_close_in_tranche),
                "price_extreme_close": float(e.price_extreme_close),
                "pvt_extreme_in_tranche": float(e.pvt_extreme_in_tranche),
                "pvi_extreme_in_tranche": float(e.pvi_extreme_in_tranche) if e.pvi_extreme_in_tranche is not None else pd.NA,
                "asi_extreme_in_tranche": float(e.asi_extreme_in_tranche) if e.asi_extreme_in_tranche is not None else pd.NA,
                "dmi_category": str(e.dmi_category),
                "dmi_filter": str(e.dmi_filter),
                "dmi_force_brute": str(e.dmi_force_brute),
            }
            for e in events
        ]
        all_events.extend(ev_rows)

        sig = _build_signals(events=events, extreme_seq_len=int(seq_len))
        all_signals.append(sig)

        structure = _derive_structure_events(
            events=events,
            signals=sig,
            enable_rebottom=bool(args.enable_rebottom),
            max_bottom_breaks=int(max_bottom_breaks),
        )
        all_structure.append(structure)

    df_events = pd.DataFrame(all_events)
    df_signals = pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame()
    df_structure = pd.concat(all_structure, ignore_index=True) if all_structure else pd.DataFrame()

    run_name = (
        f"accum_spec_{args.interval}_flow_{args.flow_indicator}_seq{int(seq_len)}"
        f"_cci{args.confluence_mode}_L{int(args.cci_level)}"
        f"_dmi_{args.require_dmi_category}_{args.require_dmi_filter}"
        f"_breaks{int(max_bottom_breaks)}"
    )
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    df_events.to_csv(run_dir / "events.csv", index=False)
    df_signals.to_csv(run_dir / "signals.csv", index=False)
    df_structure.to_csv(run_dir / "structure.csv", index=False)

    print(f"Wrote: {run_dir / 'events.csv'}")
    print(f"Wrote: {run_dir / 'signals.csv'}")
    print(f"Wrote: {run_dir / 'structure.csv'}")

    print("\n=== SUMMARY ===")
    print(f"events={len(df_events)} signals={len(df_signals)} structure={len(df_structure)}")
    if len(df_signals):
        print(df_signals["kind"].value_counts().to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
