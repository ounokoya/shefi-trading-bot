from __future__ import annotations

import math
from typing import Sequence

import pandas as pd

from libs.indicators.momentum.macd_tv import macd_tv
from libs.indicators.momentum.cci_tv import cci_tv
from libs.indicators.momentum.dmi_tv import dmi_tv
from libs.indicators.volume.nvi_tv import nvi_tv
from libs.indicators.volume.pvi_tv import pvi_tv
from libs.indicators.volume.pvt_tv import pvt_tv
from libs.indicators.volume.klinger_oscillator_tv import klinger_oscillator_tv
from libs.strategies.klinger_cci_extremes.config import KlingerCciExtremesConfig


def _safe_float(v: object) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return float(x)


def _dt(ts: int) -> str:
    if int(ts) <= 0:
        return ""
    return str(pd.to_datetime(int(ts), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S UTC"))


def _sign(x: float) -> int:
    if not math.isfinite(float(x)):
        return 0
    if float(x) > 0.0:
        return 1
    if float(x) < 0.0:
        return -1
    return 0


def _ensure_dt_col(df: pd.DataFrame, *, cfg: KlingerCciExtremesConfig) -> pd.DataFrame:
    out = df
    if str(cfg.dt_col) in out.columns:
        return out
    try:
        out[str(cfg.dt_col)] = pd.to_datetime(out[str(cfg.ts_col)].astype(int), unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        out[str(cfg.dt_col)] = ""
    return out


def _ensure_cci_col(df: pd.DataFrame, *, cfg: KlingerCciExtremesConfig, cci_col: str, cci_period: int) -> pd.DataFrame:
    if str(cci_col) in df.columns:
        return df

    high = pd.to_numeric(df[str(cfg.high_col)], errors="coerce").astype(float).tolist()
    low = pd.to_numeric(df[str(cfg.low_col)], errors="coerce").astype(float).tolist()
    close = pd.to_numeric(df[str(cfg.close_col)], errors="coerce").astype(float).tolist()
    df[str(cci_col)] = cci_tv(high, low, close, int(cci_period))
    return df


def _compute_dx_from_di(*, plus_di: pd.Series, minus_di: pd.Series) -> pd.Series:
    di_sum = (plus_di + minus_di).astype(float)
    num = (plus_di - minus_di).abs().astype(float)
    dx = 100.0 * (num / di_sum.replace(0.0, pd.NA))
    return pd.to_numeric(dx, errors="coerce").astype(float)


def _dmi_force_brute_from_values(*, dx: float | None, plus_di: float | None, minus_di: float | None) -> str:
    if dx is None or plus_di is None or minus_di is None:
        return "petite_force"
    if not (math.isfinite(float(dx)) and math.isfinite(float(plus_di)) and math.isfinite(float(minus_di))):
        return "petite_force"

    lo = min(float(plus_di), float(minus_di))
    hi = max(float(plus_di), float(minus_di))
    if float(dx) > float(hi):
        return "tres_fort"
    if float(dx) < float(lo):
        return "petite_force"
    return "moyen_fort"


def analyze_klinger_cci_tranches(
    df: pd.DataFrame,
    *,
    cfg: KlingerCciExtremesConfig | None = None,
) -> list[dict[str, object]]:
    cfg = cfg or KlingerCciExtremesConfig()

    for c in (cfg.ts_col, cfg.high_col, cfg.low_col, cfg.close_col, cfg.volume_col):
        if str(c) not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    work = df.copy()
    work = _ensure_dt_col(work, cfg=cfg)

    if bool(cfg.enable_cci_14):
        work = _ensure_cci_col(work, cfg=cfg, cci_col=str(cfg.cci_14_col), cci_period=int(cfg.cci_14_period))
    if bool(cfg.enable_cci_30):
        work = _ensure_cci_col(work, cfg=cfg, cci_col=str(cfg.cci_30_col), cci_period=int(cfg.cci_30_period))
    if bool(cfg.enable_cci_300):
        work = _ensure_cci_col(work, cfg=cfg, cci_col=str(cfg.cci_300_col), cci_period=int(cfg.cci_300_period))

    high_l = pd.to_numeric(work[str(cfg.high_col)], errors="coerce").astype(float).tolist()
    low_l = pd.to_numeric(work[str(cfg.low_col)], errors="coerce").astype(float).tolist()
    close_l = pd.to_numeric(work[str(cfg.close_col)], errors="coerce").astype(float).tolist()

    vol_l = pd.to_numeric(work[str(cfg.volume_col)], errors="coerce").astype(float).tolist()

    adx, di_plus, di_minus = dmi_tv(
        high_l,
        low_l,
        close_l,
        int(cfg.dmi_period),
        adx_smoothing=int(cfg.dmi_adx_smoothing),
    )
    work["adx"] = adx
    work["plus_di"] = di_plus
    work["minus_di"] = di_minus
    work["dx"] = _compute_dx_from_di(
        plus_di=pd.to_numeric(work["plus_di"], errors="coerce").astype(float),
        minus_di=pd.to_numeric(work["minus_di"], errors="coerce").astype(float),
    )

    flow_indicator = str(getattr(cfg, "flow_indicator", "klinger") or "klinger").strip().lower()
    flow_signal_period = int(getattr(cfg, "flow_signal_period", getattr(cfg, "kvo_signal", 13)) or 13)

    if flow_indicator in {"klinger", "kvo"}:
        kvo, ksig = klinger_oscillator_tv(
            high_l,
            low_l,
            close_l,
            vol_l,
            fast=int(cfg.kvo_fast),
            slow=int(cfg.kvo_slow),
            signal=int(cfg.kvo_signal),
            vf_use_abs_temp=bool(cfg.vf_use_abs_temp),
        )
        work["kvo"] = kvo
        work["kvo_signal"] = ksig
    else:
        if flow_indicator == "pvt":
            flow = pvt_tv(close_l, vol_l)
        elif flow_indicator == "nvi":
            flow = nvi_tv(close_l, vol_l, start=float(getattr(cfg, "nvi_start", 1000.0)))
        elif flow_indicator == "pvi":
            flow = pvi_tv(close_l, vol_l, start=float(getattr(cfg, "pvi_start", 1000.0)))
        elif flow_indicator in {"pvt_pvi", "pvi_pvt"}:
            pvt = pvt_tv(close_l, vol_l)
            pvi = pvi_tv(close_l, vol_l, start=float(getattr(cfg, "pvi_start", 1000.0)))
            flow = [a + b for a, b in zip(pvt, pvi)]
        else:
            raise ValueError(f"Unsupported flow_indicator: {flow_indicator}")

        work["kvo"] = flow
        work["kvo_signal"] = flow

    work["kvo_diff"] = work["kvo"].astype(float) - work["kvo_signal"].astype(float)

    tranche_source = str(getattr(cfg, "tranche_source", "kvo_diff") or "kvo_diff").strip().lower()

    tranche_col = "kvo_diff"
    if tranche_source in {"macd_hist", "macd", "hist"}:
        tranche_col = str(getattr(cfg, "macd_hist_col", "macd_hist"))
        if tranche_col not in work.columns:
            macd_line, macd_signal, macd_hist = macd_tv(
                close_l,
                int(getattr(cfg, "macd_fast", 12)),
                int(getattr(cfg, "macd_slow", 26)),
                int(getattr(cfg, "macd_signal", 9)),
            )
            work[str(getattr(cfg, "macd_line_col", "macd_line"))] = macd_line
            work[str(getattr(cfg, "macd_signal_col", "macd_signal"))] = macd_signal
            work[str(tranche_col)] = macd_hist

    src = pd.to_numeric(work[str(tranche_col)], errors="coerce").astype(float)
    src_prev = src.shift(1)
    up_mask = (src_prev <= 0) & (src > 0)
    dn_mask = (src_prev >= 0) & (src < 0)
    cross_pos = work.index[up_mask | dn_mask].to_list()

    out: list[dict[str, object]] = []
    if len(cross_pos) < 2:
        return out

    enabled_ccis: list[tuple[str, float]] = []
    if bool(cfg.enable_cci_14):
        enabled_ccis.append((str(cfg.cci_14_col), float(cfg.cci_extreme_level)))
    if bool(cfg.enable_cci_30):
        enabled_ccis.append((str(cfg.cci_30_col), float(cfg.cci_extreme_level)))
    if bool(cfg.enable_cci_300):
        enabled_ccis.append((str(cfg.cci_300_col), float(cfg.cci_extreme_level)))

    ref_col = str(cfg.cci_30_col) if int(cfg.reference_cci) == 30 else (str(cfg.cci_14_col) if int(cfg.reference_cci) == 14 else str(cfg.cci_300_col))

    close_s = pd.to_numeric(work[str(cfg.close_col)], errors="coerce").astype(float)

    for j in range(len(cross_pos) - 1):
        start_i = int(cross_pos[j])
        end_i = int(cross_pos[j + 1])
        seg = work.iloc[start_i : end_i + 1].copy()
        if not len(seg):
            continue

        side = ""
        v0 = _safe_float(seg[str(tranche_col)].iloc[0])
        v1 = _safe_float(seg[str(tranche_col)].iloc[1]) if len(seg) > 1 else None
        sd = _sign(float(v1 if v1 is not None else (v0 if v0 is not None else 0.0)))
        if sd > 0:
            side = "LONG"
        elif sd < 0:
            side = "SHORT"
        else:
            side = ""

        seg_dt0 = str(seg[str(cfg.dt_col)].iloc[0])
        seg_dt1 = str(seg[str(cfg.dt_col)].iloc[-1])

        kvo_seg = pd.to_numeric(seg["kvo"], errors="coerce").astype(float)
        kvo_seg_nn = kvo_seg.dropna()
        if not len(kvo_seg_nn):
            continue

        if side == "LONG":
            kvo_ext_i = int(kvo_seg_nn.idxmax())
        else:
            kvo_ext_i = int(kvo_seg_nn.idxmin())

        kvo_ext = _safe_float(work.loc[kvo_ext_i, "kvo"])
        ksig_at_ext = _safe_float(work.loc[kvo_ext_i, "kvo_signal"])
        close_at_ext = _safe_float(work.loc[kvo_ext_i, str(cfg.close_col)])
        dt_at_ext = str(work.loc[kvo_ext_i, str(cfg.dt_col)])

        def _confirm_extreme(pos: int, *, bars: int) -> bool:
            if int(bars) <= 0:
                return False
            if int(pos) >= int(end_i):
                return False
            end_chk = min(int(end_i), int(pos) + int(bars))
            post = work.iloc[int(pos) + 1 : int(end_chk) + 1]
            if not len(post):
                return False
            v0 = _safe_float(work.loc[int(pos), "kvo"])
            if v0 is None:
                return False
            pv = pd.to_numeric(post["kvo"], errors="coerce").astype(float)
            if side == "LONG":
                return bool((pv <= float(v0)).all())
            if side == "SHORT":
                return bool((pv >= float(v0)).all())
            return False

        confirm_weak = _confirm_extreme(kvo_ext_i, bars=int(cfg.confirm_bars_weak))
        confirm_strong = _confirm_extreme(kvo_ext_i, bars=int(cfg.confirm_bars_strong))

        # DMI (computed per-bar) and tranche classification
        adx_s = pd.to_numeric(seg["adx"], errors="coerce").astype(float)
        dx_s = pd.to_numeric(seg["dx"], errors="coerce").astype(float)
        pdi_s = pd.to_numeric(seg["plus_di"], errors="coerce").astype(float)
        mdi_s = pd.to_numeric(seg["minus_di"], errors="coerce").astype(float)

        # Values at the Klinger extreme (authoritative for category)
        adx_ext = _safe_float(work.loc[kvo_ext_i, "adx"])
        dx_ext = _safe_float(work.loc[kvo_ext_i, "dx"])
        pdi_ext = _safe_float(work.loc[kvo_ext_i, "plus_di"])
        mdi_ext = _safe_float(work.loc[kvo_ext_i, "minus_di"])

        dmi_side = ""
        if pdi_ext is not None and mdi_ext is not None:
            if float(pdi_ext) > float(mdi_ext):
                dmi_side = "LONG"
            elif float(mdi_ext) > float(pdi_ext):
                dmi_side = "SHORT"

        dmi_aligned = bool(side and dmi_side and str(side) == str(dmi_side))

        th = float(cfg.adx_force_threshold)
        dmi_force_confirmed = bool(adx_ext is not None and math.isfinite(float(adx_ext)) and float(adx_ext) >= th)

        dmi_category = "tendenciel" if bool(dmi_force_confirmed) else "plat"

        dmi_filter = "respiration"
        if dx_ext is not None and adx_ext is not None and math.isfinite(float(dx_ext)) and math.isfinite(float(adx_ext)):
            dmi_filter = "impulsion" if float(dx_ext) > float(adx_ext) else "respiration"

        dmi_force_brute = _dmi_force_brute_from_values(dx=dx_ext, plus_di=pdi_ext, minus_di=mdi_ext)

        cci_meta: dict[str, object] = {}
        confluence_ok = True
        confluence_sign = 0

        for cci_col, lvl in enabled_ccis:
            s = pd.to_numeric(seg[str(cci_col)], errors="coerce").astype(float)
            if not len(s.dropna()):
                cci_meta[f"{cci_col}_has_extreme"] = False
                confluence_ok = False
                continue

            if side == "LONG":
                mask = s >= float(abs(lvl))
                most_i = int(s.idxmax())
                most_v = float(s.loc[most_i])
                ok_sign = bool(most_v >= float(abs(lvl)))
                sign = 1 if ok_sign else 0
            else:
                mask = s <= -float(abs(lvl))
                most_i = int(s.idxmin())
                most_v = float(s.loc[most_i])
                ok_sign = bool(most_v <= -float(abs(lvl)))
                sign = -1 if ok_sign else 0

            has_ext = bool(mask.fillna(False).any())
            cci_meta[f"{cci_col}_has_extreme"] = bool(has_ext)
            cci_meta[f"{cci_col}_extreme_value"] = float(most_v) if math.isfinite(float(most_v)) else None
            cci_meta[f"{cci_col}_extreme_dt"] = str(work.loc[most_i, str(cfg.dt_col)])
            cci_meta[f"{cci_col}_extreme_close"] = _safe_float(work.loc[most_i, str(cfg.close_col)])

            if not bool(has_ext):
                confluence_ok = False
            if confluence_sign == 0:
                confluence_sign = int(sign)
            else:
                if int(sign) != int(confluence_sign):
                    confluence_ok = False

        # Episodes based on reference CCI 30
        episodes: list[dict[str, object]] = []
        if side in {"LONG", "SHORT"} and str(ref_col) in seg.columns:
            ref = pd.to_numeric(seg[str(ref_col)], errors="coerce").astype(float).to_numpy()
            seg_close = pd.to_numeric(seg[str(cfg.close_col)], errors="coerce").astype(float).to_numpy()
            seg_sig = pd.to_numeric(seg["kvo_signal"], errors="coerce").astype(float).to_numpy()
            lvl = float(abs(cfg.cci_extreme_level))
            in_ext = []
            for x in ref:
                if not math.isfinite(float(x)):
                    in_ext.append(False)
                elif side == "SHORT":
                    in_ext.append(bool(float(x) <= -float(lvl)))
                else:
                    in_ext.append(bool(float(x) >= float(lvl)))

            start = None
            for k in range(len(in_ext)):
                if start is None and bool(in_ext[k]):
                    start = int(k)
                elif start is not None and (not bool(in_ext[k])):
                    episodes.append({"start_off": int(start), "end_off": int(k - 1)})
                    start = None
            if start is not None:
                episodes.append({"start_off": int(start), "end_off": int(len(in_ext) - 1)})

            for ep in episodes:
                a = int(ep["start_off"])
                b = int(ep["end_off"])
                ep_slice = slice(a, b + 1)

                if side == "SHORT":
                    price_ext_val = float(pd.Series(seg_close[ep_slice]).min())
                    sig_ext_val = float(pd.Series(seg_sig[ep_slice]).min())
                else:
                    price_ext_val = float(pd.Series(seg_close[ep_slice]).max())
                    sig_ext_val = float(pd.Series(seg_sig[ep_slice]).max())

                # find pos for price extreme inside episode
                if side == "SHORT":
                    rel_i = int(pd.Series(seg_close[ep_slice]).idxmin())
                else:
                    rel_i = int(pd.Series(seg_close[ep_slice]).idxmax())

                abs_i = int(seg.index[int(rel_i)])
                ep["price_extreme_close"] = float(price_ext_val)
                ep["price_extreme_dt"] = str(work.loc[abs_i, str(cfg.dt_col)])
                ep["signal_extreme"] = float(sig_ext_val)

            deeper_progression = False
            deeper_pairs = 0
            for k in range(1, len(episodes)):
                p0 = _safe_float(episodes[k - 1].get("price_extreme_close"))
                p1 = _safe_float(episodes[k].get("price_extreme_close"))
                s0 = _safe_float(episodes[k - 1].get("signal_extreme"))
                s1 = _safe_float(episodes[k].get("signal_extreme"))
                if p0 is None or p1 is None or s0 is None or s1 is None:
                    continue
                if side == "SHORT":
                    ok = bool(float(p1) < float(p0) and float(s1) < float(s0))
                else:
                    ok = bool(float(p1) > float(p0) and float(s1) > float(s0))
                if ok:
                    deeper_progression = True
                    deeper_pairs += 1

            cci_meta["ref_cci_col"] = str(ref_col)
            cci_meta["ref_episodes"] = int(len(episodes))
            cci_meta["ref_deeper_progression"] = bool(deeper_progression)
            cci_meta["ref_deeper_pairs"] = int(deeper_pairs)

        out.append(
            {
                "start_i": int(start_i),
                "end_i": int(end_i),
                "start_dt": str(seg_dt0),
                "end_dt": str(seg_dt1),
                "side": str(side),
                "kvo_extreme_i": int(kvo_ext_i),
                "kvo_extreme_dt": str(dt_at_ext),
                "kvo_extreme": kvo_ext,
                "kvo_signal_at_extreme": ksig_at_ext,
                "close_at_extreme": close_at_ext,
                "kvo_confirm_weak": bool(confirm_weak),
                "kvo_confirm_strong": bool(confirm_strong),
                "dmi_side": str(dmi_side),
                "dmi_aligned": bool(dmi_aligned),
                "dmi_category": str(dmi_category),
                "dmi_filter": str(dmi_filter),
                "dmi_force_brute": str(dmi_force_brute),
                "dmi_force_confirmed": bool(dmi_force_confirmed),
                "dmi_adx_at_extreme": adx_ext,
                "dmi_dx_at_extreme": dx_ext,
                "dmi_plus_di_at_extreme": pdi_ext,
                "dmi_minus_di_at_extreme": mdi_ext,
                "cci_confluence_ok": bool(confluence_ok),
                "cci_meta": dict(cci_meta),
            }
        )

    return out
