#!/usr/bin/env python3
"""
Demo Zerem – Détection de structures multi-indicateurs
- Télécharge les N dernières bougies via Bybit
- Délimite les zones d’extrême via le CCI
- Détecte les structures haussières/baissières sur plusieurs séries cibles
- Produit un tableau de confluence des structures
"""

import argparse
import logging
import math
import itertools
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from libs.data_loader import get_crypto_data
from libs.indicators.asi import asi_by_market
from libs.indicators.volume.pvt_tv import pvt_tv
from libs.indicators.momentum.macd_tv import macd_tv
from libs.indicators.momentum.dmi_tv import dmi_tv
from libs.indicators.volume.klinger_oscillator_tv import klinger_oscillator_tv
from libs.indicators.volume.mfi_tv import mfi_tv
from libs.new_strategie.indicators import _compute_stoch
from libs.indicators.momentum.cci_tv import cci_tv

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def _indicator_params_for_tf(timeframe: str) -> Dict[str, int]:
    tf = str(timeframe or "").strip().lower()
    if tf == "5m":
        return {
            "cci_period": 96,
            "stoch_k_period": 96,
            "stoch_k_smooth": 2,
            "stoch_d_period": 3,
        }
    if tf == "15m":
        # 24h (1 jour) = 96 bougies de 15m
        return {
            "cci_period": 96,
            "stoch_k_period": 96,
            "stoch_k_smooth": 2,
            "stoch_d_period": 3,
        }
    if tf == "1h":
        return {
            "cci_period": 168,
            "stoch_k_period": 168,
            "stoch_k_smooth": 2,
            "stoch_d_period": 3,
        }
    if tf == "2h":
        # 1 semaine = 7*24h = 168h = 84 bougies de 2h
        return {
            "cci_period": 84,
            "stoch_k_period": 84,
            "stoch_k_smooth": 2,
            "stoch_d_period": 3,
        }
    if tf == "6h":
        # 1 mois (approx 30 jours) = 30*24h = 720h = 120 bougies de 6h
        return {
            "cci_period": 120,
            "stoch_k_period": 120,
            "stoch_k_smooth": 2,
            "stoch_d_period": 3,
        }
    if tf == "1d":
        # 3 mois (approx 90 jours) = 90 bougies daily
        return {
            "cci_period": 90,
            "stoch_k_period": 90,
            "stoch_k_smooth": 2,
            "stoch_d_period": 3,
        }
    return {
        "cci_period": 20,
        "stoch_k_period": 14,
        "stoch_k_smooth": 3,
        "stoch_d_period": 3,
    }


def _load_binance_data_vision_klines(project_root: Path, symbol: str, timeframe: str) -> pd.DataFrame:
    base = project_root / "data" / "raw" / "binance_data_vision" / "futures" / "um" / "daily" / "klines"
    tf = str(timeframe or "").strip().lower()
    root = base / symbol / tf
    if not root.exists():
        return pd.DataFrame()

    csv_paths = sorted(root.rglob("*.csv"))
    if not csv_paths:
        return pd.DataFrame()

    frames = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty or "open_time" not in df.columns:
            continue
        df = df.rename(columns={"open_time": "ts"})
        keep = [c for c in ["ts", "open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep]
        df["ts"] = df["ts"].astype(int)
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["ts"], keep="last").sort_values("ts").reset_index(drop=True)
    out["dt"] = pd.to_datetime(out["ts"], unit="ms")
    return out


def _merge_history(local_df: pd.DataFrame, recent_df: pd.DataFrame) -> pd.DataFrame:
    if local_df is None or local_df.empty:
        return recent_df
    if recent_df is None or recent_df.empty:
        return local_df

    keep_cols = [c for c in ["ts", "open", "high", "low", "close", "volume"] if c in recent_df.columns]
    local_use = local_df[[c for c in keep_cols if c in local_df.columns]].copy()
    recent_use = recent_df[keep_cols].copy()
    merged = pd.concat([local_use, recent_use], ignore_index=True)
    merged = merged.drop_duplicates(subset=["ts"], keep="last").sort_values("ts").reset_index(drop=True)
    merged["dt"] = pd.to_datetime(merged["ts"], unit="ms")
    return merged


def _parse_series_list(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    return [s.strip() for s in raw.split(",") if s.strip()]

def delimit_zones_by_cci(
    df: pd.DataFrame,
    cci_col: str = "cci",
    low_threshold: float = -100.0,
    high_threshold: float = 100.0
) -> List[Dict]:
    """
    Délimite les zones où CCI < low_threshold (creux) ou CCI > high_threshold (sommets).
    
    Args:
        df: DataFrame avec colonne CCI.
        cci_col: Nom de la colonne CCI.
        low_threshold: Seuil bas pour zones de creux (défaut -100).
        high_threshold: Seuil haut pour zones de sommets (défaut 100).
    
    Returns:
        Liste de zones avec clés: start_idx, end_idx, type ('creux' ou 'sommet').
    """
    zones = []
    in_zone = False
    zone_start = None
    zone_type = None
    
    cci_s = pd.to_numeric(df[cci_col], errors="coerce").astype(float)
    cci = cci_s.values
    
    for i, val in enumerate(cci):
        prev = cci[i - 1] if i > 0 else None
        if pd.isna(val):
            continue

        if not in_zone:
            # Entrée en zone ?
            if val < low_threshold and (prev is None or pd.isna(prev) or prev >= low_threshold):
                in_zone = True
                zone_start = i
                zone_type = "creux"
            elif val > high_threshold and (prev is None or pd.isna(prev) or prev <= high_threshold):
                in_zone = True
                zone_start = i
                zone_type = "sommet"
        else:
            # Sortie de zone ?
            if zone_type == "creux" and val >= low_threshold:
                zones.append({"start_idx": zone_start, "end_idx": i - 1, "type": "creux"})
                in_zone = False
                zone_start = None
                zone_type = None
            elif zone_type == "sommet" and val <= high_threshold:
                zones.append({"start_idx": zone_start, "end_idx": i - 1, "type": "sommet"})
                in_zone = False
                zone_start = None
                zone_type = None
    
    # Fermer zone ouverte à la fin
    if in_zone and zone_start is not None:
        zones.append({"start_idx": zone_start, "end_idx": len(df) - 1, "type": zone_type})
    
    return zones

def _find_structures_generic(
    df: pd.DataFrame,
    zones: List[Dict],
    value_col: str,
    series_name: str
) -> List[Dict]:
    """
    Générique : détecte structures haussières/baissières sur une série cible.
    Creux : valeur min dans la zone, structure haussière si creux_2 > creux_1.
    Sommets : valeur max dans la zone, structure baissière si sommet_2 < sommet_1.
    
    Args:
        df: DataFrame avec colonne cible.
        zones: Zones délimitées par CCI.
        value_col: Colonne de la série cible.
        series_name: Nom de la série pour logs.
    
    Returns:
        Liste d'extrêmes et structures détectées.
    """
    structures = []
    last_creux: Optional[Tuple[int, float]] = None
    last_sommet: Optional[Tuple[int, float]] = None
    
    values = pd.to_numeric(df[value_col], errors="coerce").fillna(0).values
    
    for zone in zones:
        start_idx = zone["start_idx"]
        end_idx = zone["end_idx"]
        zone_type = zone["type"]
        
        # Extraire extrême dans la zone
        if zone_type == "creux":
            idx_min = int(start_idx + values[start_idx:end_idx+1].argmin())
            val_min = float(values[idx_min])
            extremum = {
                "idx": idx_min,
                "value": val_min,
                "type": "creux",
                "zone": zone,
                "ts": df.iloc[idx_min].get("ts"),
                "dt": df.iloc[idx_min].get("dt"),
                "open": df.iloc[idx_min].get("open"),
            }
            structures.append(extremum)
            # Structure haussière ?
            if last_creux is not None:
                if val_min > last_creux[1]:
                    structures.append({
                        "idx": idx_min,
                        "type": "structure_haussiere",
                        "serie": series_name,
                        "creux_1": {"idx": last_creux[0], "value": last_creux[1]},
                        "creux_2": {"idx": idx_min, "value": val_min},
                        "ts": df.iloc[idx_min].get("ts"),
                        "dt": df.iloc[idx_min].get("dt"),
                        "open": df.iloc[idx_min].get("open"),
                    })
            last_creux = (idx_min, val_min)
        elif zone_type == "sommet":
            idx_max = int(start_idx + values[start_idx:end_idx+1].argmax())
            val_max = float(values[idx_max])
            extremum = {
                "idx": idx_max,
                "value": val_max,
                "type": "sommet",
                "zone": zone,
                "ts": df.iloc[idx_max].get("ts"),
                "dt": df.iloc[idx_max].get("dt"),
                "open": df.iloc[idx_max].get("open"),
            }
            structures.append(extremum)
            # Structure baissière ?
            if last_sommet is not None:
                if val_max < last_sommet[1]:
                    structures.append({
                        "idx": idx_max,
                        "type": "structure_baissiere",
                        "serie": series_name,
                        "sommet_1": {"idx": last_sommet[0], "value": last_sommet[1]},
                        "sommet_2": {"idx": idx_max, "value": val_max},
                        "ts": df.iloc[idx_max].get("ts"),
                        "dt": df.iloc[idx_max].get("dt"),
                        "open": df.iloc[idx_max].get("open"),
                    })
            last_sommet = (idx_max, val_max)
    
    return structures


def analyze_series_by_zones(
    df: pd.DataFrame,
    zones: List[Dict],
    value_col: str,
    series_name: str,
) -> List[Dict]:
    results = []
    last_creux: Optional[Dict] = None
    last_sommet: Optional[Dict] = None

    values = pd.to_numeric(df[value_col], errors="coerce").fillna(0).values

    for zone in zones:
        start_idx = zone["start_idx"]
        end_idx = zone["end_idx"]
        zone_type = zone["type"]
        zone_id = zone.get("zone_id")

        zone_start_dt = df.iloc[start_idx].get("dt") if start_idx < len(df) else None
        zone_end_dt = df.iloc[end_idx].get("dt") if end_idx < len(df) else None

        if zone_type == "creux":
            idx_ext = int(start_idx + values[start_idx:end_idx + 1].argmin())
            val_ext = float(values[idx_ext])
            cur_ext = {
                "idx": idx_ext,
                "value": val_ext,
                "type": "creux",
                "serie": series_name,
                "ts": df.iloc[idx_ext].get("ts"),
                "dt": df.iloc[idx_ext].get("dt"),
                "open": df.iloc[idx_ext].get("open"),
            }
            structure_type = None
            if last_creux is not None and val_ext > float(last_creux["value"]):
                structure_type = "structure_haussiere"
            results.append({
                "zone_id": zone_id,
                "zone": zone,
                "zone_start_dt": zone_start_dt,
                "zone_end_dt": zone_end_dt,
                "serie": series_name,
                "extremum": cur_ext,
                "prev_extremum": last_creux,
                "structure": structure_type,
            })
            last_creux = cur_ext

        elif zone_type == "sommet":
            idx_ext = int(start_idx + values[start_idx:end_idx + 1].argmax())
            val_ext = float(values[idx_ext])
            cur_ext = {
                "idx": idx_ext,
                "value": val_ext,
                "type": "sommet",
                "serie": series_name,
                "ts": df.iloc[idx_ext].get("ts"),
                "dt": df.iloc[idx_ext].get("dt"),
                "open": df.iloc[idx_ext].get("open"),
            }
            structure_type = None
            if last_sommet is not None and val_ext < float(last_sommet["value"]):
                structure_type = "structure_baissiere"
            results.append({
                "zone_id": zone_id,
                "zone": zone,
                "zone_start_dt": zone_start_dt,
                "zone_end_dt": zone_end_dt,
                "serie": series_name,
                "extremum": cur_ext,
                "prev_extremum": last_sommet,
                "structure": structure_type,
            })
            last_sommet = cur_ext

    return results


def _hist_cross_up(prev_h: float, h: float) -> bool:
    if not math.isfinite(float(h)) or not math.isfinite(float(prev_h)):
        return False
    return float(prev_h) <= 0.0 and float(h) > 0.0


def _hist_cross_down(prev_h: float, h: float) -> bool:
    if not math.isfinite(float(h)) or not math.isfinite(float(prev_h)):
        return False
    return float(prev_h) >= 0.0 and float(h) < 0.0


def _safe_float(v: object) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def simulate_trades_from_stream(
    df: pd.DataFrame,
    *,
    series_to_col: Dict[str, str],
    mode: str,
    signal_from: str,
    selected_series: List[str],
    cci_col: str,
    cci_low: float,
    cci_high: float,
    macd_hist_col: str,
    trade_direction: str,
    min_confluence: int,
    use_fixed_stop: bool,
    stop_buffer_pct: float,
    stop_ref_series: str,
    start_i: int,
) -> List[Dict]:
    cci = pd.to_numeric(df[cci_col], errors="coerce").astype(float).to_numpy()
    macd_hist = pd.to_numeric(df[macd_hist_col], errors="coerce").astype(float).to_numpy()

    base_series = [signal_from] if str(mode) == "single" else list(selected_series)
    base_series = [s for s in base_series if s in series_to_col]
    if not base_series:
        return []

    stop_ref = str(stop_ref_series or "").strip()
    tracking_series = list(base_series)
    if bool(use_fixed_stop) and stop_ref and stop_ref in series_to_col and stop_ref not in tracking_series:
        tracking_series.append(stop_ref)

    last_confirmed: Dict[str, Dict[str, Optional[Dict]]] = {}
    for s in tracking_series:
        last_confirmed[s] = {"creux": None, "sommet": None}

    in_zone = False
    zone_type: Optional[str] = None
    zone_id = -1

    potential: Dict[str, Optional[Dict]] = {s: None for s in tracking_series}

    pending_bull: Optional[Dict] = None
    pending_bear: Optional[Dict] = None

    pos = 0
    entry: Optional[Dict] = None
    trades: List[Dict] = []

    def _close_position(i: int, reason: str, exit_price_override: Optional[float] = None) -> None:
        nonlocal pos, entry, trades
        if pos == 0 or entry is None:
            return
        if exit_price_override is not None:
            exit_price = _safe_float(exit_price_override)
        else:
            exit_price = _safe_float(df.iloc[i].get("close"))
        if exit_price is None or float(exit_price) == 0.0:
            return

        entry_price = float(entry["entry_price"])
        side = int(entry["side"])
        if side == 1:
            pct = 100.0 * ((float(exit_price) - entry_price) / entry_price)
        else:
            pct = 100.0 * ((entry_price - float(exit_price)) / entry_price)

        trade = {
            **entry,
            "exit_i": int(i),
            "exit_ts": df.iloc[i].get("ts"),
            "exit_dt": df.iloc[i].get("dt"),
            "exit_price": float(exit_price),
            "pct": float(pct),
            "exit_reason": str(reason),
        }
        trades.append(trade)
        pos = 0
        entry = None

    def _compute_fixed_stop(side: int, signal: Dict) -> Optional[float]:
        if not bool(use_fixed_stop):
            return None
        per_series = signal.get("per_series") or {}
        price_meta = per_series.get("price")
        if not isinstance(price_meta, dict):
            return None
        prev_conf = price_meta.get("prev_confirmed")
        if not isinstance(prev_conf, dict):
            return None
        ref = _safe_float(prev_conf.get("value"))
        if ref is None or float(ref) == 0.0:
            return None
        buf = float(stop_buffer_pct or 0.0)
        if int(side) == 1:
            return float(ref) * (1.0 - buf)
        return float(ref) * (1.0 + buf)

    def _open_position(i: int, side: int, signal: Dict) -> None:
        nonlocal pos, entry
        entry_price = _safe_float(df.iloc[i].get("close"))
        if entry_price is None or float(entry_price) == 0.0:
            return
        stop_price = _compute_fixed_stop(int(side), signal)
        pos = int(side)
        entry = {
            "side": int(side),
            "entry_i": int(i),
            "entry_ts": df.iloc[i].get("ts"),
            "entry_dt": df.iloc[i].get("dt"),
            "entry_price": float(entry_price),
            "stop_price": stop_price,
            "signal": signal,
        }

    def _apply_signal(i: int, direction: str, signal: Dict) -> None:
        nonlocal pos, pending_bull, pending_bear
        td = str(trade_direction or "both").strip().lower()

        if str(direction) == "bull":
            if td == "short":
                return
            if pos == -1:
                _close_position(i, "flip_to_long")
            if pos == 0:
                _open_position(i, 1, signal)
            pending_bull = None
            return

        if str(direction) == "bear":
            if td == "long":
                if pos == 1:
                    _close_position(i, "close_long")
                pending_bear = None
                return
            if pos == 1:
                _close_position(i, "flip_to_short")
            if pos == 0:
                _open_position(i, -1, signal)
            pending_bear = None
            return

    n = int(len(df))
    for i in range(n):
        if int(i) >= int(start_i) and pos != 0 and entry is not None:
            stop_price = _safe_float(entry.get("stop_price"))
            if stop_price is not None:
                if int(pos) == 1:
                    lo = _safe_float(df.iloc[i].get("low"))
                    if lo is not None and float(lo) <= float(stop_price):
                        _close_position(i, "stop", exit_price_override=float(stop_price))
                elif int(pos) == -1:
                    hi = _safe_float(df.iloc[i].get("high"))
                    if hi is not None and float(hi) >= float(stop_price):
                        _close_position(i, "stop", exit_price_override=float(stop_price))

        v = cci[i] if i < len(cci) else math.nan
        prev = cci[i - 1] if i > 0 and i - 1 < len(cci) else None

        if not in_zone:
            if math.isfinite(float(v)) and float(v) < float(cci_low) and (prev is None or (math.isfinite(float(prev)) and float(prev) >= float(cci_low)) or (prev is not None and not math.isfinite(float(prev)))):
                in_zone = True
                zone_type = "creux"
                zone_id += 1
                for s in tracking_series:
                    potential[s] = None
            elif math.isfinite(float(v)) and float(v) > float(cci_high) and (prev is None or (math.isfinite(float(prev)) and float(prev) <= float(cci_high)) or (prev is not None and not math.isfinite(float(prev)))):
                in_zone = True
                zone_type = "sommet"
                zone_id += 1
                for s in tracking_series:
                    potential[s] = None
        else:
            if zone_type == "creux" and math.isfinite(float(v)) and float(v) >= float(cci_low):
                for s in tracking_series:
                    if potential[s] is not None:
                        last_confirmed[s]["creux"] = {**potential[s], "status": "confirmed", "confirmed_i": int(i)}
                    potential[s] = None
                in_zone = False
                zone_type = None
            elif zone_type == "sommet" and math.isfinite(float(v)) and float(v) <= float(cci_high):
                for s in tracking_series:
                    if potential[s] is not None:
                        last_confirmed[s]["sommet"] = {**potential[s], "status": "confirmed", "confirmed_i": int(i)}
                    potential[s] = None
                in_zone = False
                zone_type = None

        if in_zone and zone_type in {"creux", "sommet"}:
            struct_ok: Dict[str, bool] = {}
            per_series: Dict[str, Dict] = {}
            for s in tracking_series:
                col = series_to_col.get(s)
                if col is None:
                    continue
                vv = _safe_float(df.iloc[i].get(col))
                if vv is None:
                    continue

                cur = potential.get(s)
                if cur is None:
                    potential[s] = {
                        "idx": int(i),
                        "value": float(vv),
                        "type": str(zone_type),
                        "serie": str(s),
                        "ts": df.iloc[i].get("ts"),
                        "dt": df.iloc[i].get("dt"),
                        "status": "potential",
                        "zone_id": int(zone_id),
                    }
                else:
                    if zone_type == "creux" and float(vv) < float(cur["value"]):
                        potential[s] = {**cur, "idx": int(i), "value": float(vv), "ts": df.iloc[i].get("ts"), "dt": df.iloc[i].get("dt")}
                    elif zone_type == "sommet" and float(vv) > float(cur["value"]):
                        potential[s] = {**cur, "idx": int(i), "value": float(vv), "ts": df.iloc[i].get("ts"), "dt": df.iloc[i].get("dt")}

                pot = potential.get(s)
                if pot is None:
                    continue

                prev_conf = last_confirmed[s][zone_type]
                ok = False
                if prev_conf is not None:
                    if zone_type == "creux" and float(pot["value"]) > float(prev_conf["value"]):
                        ok = True
                    elif zone_type == "sommet" and float(pot["value"]) < float(prev_conf["value"]):
                        ok = True

                if s in base_series:
                    struct_ok[s] = bool(ok)
                per_series[s] = {"potential": pot, "prev_confirmed": prev_conf}

            n_ok = sum(1 for v0 in struct_ok.values() if bool(v0))
            if int(n_ok) >= int(max(1, min_confluence)):
                signal = {
                    "zone_id": int(zone_id),
                    "zone_type": str(zone_type),
                    "created_i": int(i),
                    "created_dt": df.iloc[i].get("dt"),
                    "mode": str(mode),
                    "series_ok": [s for s, ok in struct_ok.items() if bool(ok)],
                    "per_series": per_series,
                }
                if zone_type == "creux":
                    pending_bull = signal
                elif zone_type == "sommet":
                    pending_bear = signal
            else:
                if zone_type == "creux":
                    pending_bull = None
                elif zone_type == "sommet":
                    pending_bear = None

        if i <= 0 or i >= len(macd_hist):
            continue

        if int(i) < int(start_i):
            continue

        if _hist_cross_up(float(macd_hist[i - 1]), float(macd_hist[i])) and pending_bull is not None:
            if int(pending_bull.get("created_i") or 0) < int(i):
                _apply_signal(i, "bull", pending_bull)
        elif _hist_cross_down(float(macd_hist[i - 1]), float(macd_hist[i])) and pending_bear is not None:
            if int(pending_bear.get("created_i") or 0) < int(i):
                _apply_signal(i, "bear", pending_bear)

    if int(len(df)) > 0 and pos != 0:
        _close_position(int(len(df) - 1), "eod")

    return trades

def find_structures_price(df: pd.DataFrame, zones: List[Dict], price_col: str = "close") -> List[Dict]:
    """Détecte structures sur les prix (close par défaut)."""
    return _find_structures_generic(df, zones, price_col, "price")

def find_structures_asi(df: pd.DataFrame, zones: List[Dict], asi_col: str = "asi") -> List[Dict]:
    """Détecte structures sur l'ASI (Accumulative Swing Index)."""
    return _find_structures_generic(df, zones, asi_col, "asi")

def find_structures_pvt(df: pd.DataFrame, zones: List[Dict], pvt_col: str = "pvt") -> List[Dict]:
    """Détecte structures sur le PVT (Price-Volume Trend)."""
    return _find_structures_generic(df, zones, pvt_col, "pvt")

def find_structures_macd_line(df: pd.DataFrame, zones: List[Dict], macd_line_col: str = "macd_line") -> List[Dict]:
    """Détecte structures sur la ligne MACD."""
    return _find_structures_generic(df, zones, macd_line_col, "macd_line")

def find_structures_dmi_dx(df: pd.DataFrame, zones: List[Dict], dx_col: str = "dx") -> List[Dict]:
    """Détecte structures sur le DX (Directional Movement Index)."""
    return _find_structures_generic(df, zones, dx_col, "dmi_dx")

def find_structures_klinger_kvo(df: pd.DataFrame, zones: List[Dict], kvo_col: str = "kvo") -> List[Dict]:
    """Détecte structures sur le Klinger Volume Oscillator (KVO)."""
    return _find_structures_generic(df, zones, kvo_col, "klinger_kvo")

def find_structures_klinger_signal(df: pd.DataFrame, zones: List[Dict], ks_col: str = "klinger_signal") -> List[Dict]:
    """Détecte structures sur la ligne signal du Klinger."""
    return _find_structures_generic(df, zones, ks_col, "klinger_signal")

def find_structures_mfi(df: pd.DataFrame, zones: List[Dict], mfi_col: str = "mfi") -> List[Dict]:
    """Détecte structures sur le MFI (Money Flow Index)."""
    return _find_structures_generic(df, zones, mfi_col, "mfi")

def find_structures_stoch_k(df: pd.DataFrame, zones: List[Dict], stoch_k_col: str = "stoch_k") -> List[Dict]:
    """Détecte structures sur le %K de Stochastique."""
    return _find_structures_generic(df, zones, stoch_k_col, "stoch_k")

def find_structures_stoch_d(df: pd.DataFrame, zones: List[Dict], stoch_d_col: str = "stoch_d") -> List[Dict]:
    """Détecte structures sur le %D de Stochastique."""
    return _find_structures_generic(df, zones, stoch_d_col, "stoch_d")

def compute_confluence(structures_by_series: Dict[str, List[Dict]], df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les structures par timestamp et compte les confluences.
    
    Args:
        structures_by_series: dict série -> liste de structures (incluant extrêmes).
        df: DataFrame original pour récupérer ts et dt.
    
    Returns:
        DataFrame avec colonnes: idx, ts, dt, open, n_haussier, n_baissier, series_haussier, series_baissier.
    """
    # Indexer par timestamp (idx)
    by_idx: Dict[int, Dict] = {}
    
    for serie, structs in structures_by_series.items():
        for s in structs:
            idx = s["idx"]
            if idx not in by_idx:
                by_idx[idx] = {"ts": None, "dt": None, "open": None, "n_haussier": 0, "n_baissier": 0, "series_haussier": [], "series_baissier": []}
            if s["type"] == "structure_haussiere":
                by_idx[idx]["n_haussier"] += 1
                by_idx[idx]["series_haussier"].append(serie)
            elif s["type"] == "structure_baissiere":
                by_idx[idx]["n_baissier"] += 1
                by_idx[idx]["series_baissier"].append(serie)
    
    # Construire DataFrame avec vrais timestamps et open
    rows = []
    for idx, info in sorted(by_idx.items()):
        if idx < len(df):
            rows.append({
                "idx": idx,
                "ts": df.iloc[idx].get("ts"),
                "dt": df.iloc[idx].get("dt"),
                "open": df.iloc[idx].get("open"),
                "n_haussier": info["n_haussier"],
                "n_baissier": info["n_baissier"],
                "series_haussier": info["series_haussier"],
                "series_baissier": info["series_baissier"]
            })
    return pd.DataFrame(rows)


def _tf_to_timedelta(timeframe: str, n: int) -> pd.Timedelta:
    """Convertit un timeframe texte en timedelta approximative pour n bougies."""
    unit = timeframe[-1]
    try:
        value = int(timeframe[:-1])
    except ValueError:
        value = 1
    minutes = value
    if unit == "h":
        minutes = value * 60
    elif unit == "d":
        minutes = value * 60 * 24
    elif unit == "w":
        minutes = value * 60 * 24 * 7
    return pd.Timedelta(minutes=minutes * n)


def _max_drawdown(equity_curve: List[float]) -> float:
    peak = 0.0
    max_dd = 0.0
    for v in equity_curve:
        peak = max(float(peak), float(v))
        dd = float(peak) - float(v)
        max_dd = max(float(max_dd), float(dd))
    return float(max_dd)


def _trade_mae_pct(df: pd.DataFrame, trade: Dict) -> Optional[float]:
    entry_i = trade.get("entry_i")
    exit_i = trade.get("exit_i")
    side = trade.get("side")
    entry_price = _safe_float(trade.get("entry_price"))
    if entry_price is None or float(entry_price) == 0.0:
        return None
    if entry_i is None or exit_i is None or side is None:
        return None
    try:
        a = int(entry_i)
        b = int(exit_i)
    except Exception:
        return None
    if a < 0 or b < 0 or a >= len(df) or b >= len(df):
        return None
    if b < a:
        a, b = b, a
    sl = df.iloc[a : b + 1]
    if sl.empty:
        return None
    if int(side) == 1:
        lo = pd.to_numeric(sl.get("low"), errors="coerce").astype(float)
        m = float(lo.min()) if len(lo) else math.nan
        if not math.isfinite(m):
            return None
        return 100.0 * ((float(m) - float(entry_price)) / float(entry_price))
    if int(side) == -1:
        hi = pd.to_numeric(sl.get("high"), errors="coerce").astype(float)
        m = float(hi.max()) if len(hi) else math.nan
        if not math.isfinite(m):
            return None
        return -100.0 * ((float(m) - float(entry_price)) / float(entry_price))
    return None


def _grid_search(
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    timeframes: List[str],
    series_universe: List[str],
    max_combo_size: int,
    cci_low: float,
    cci_high: float,
    trade_direction: str,
    use_fixed_stop: bool,
    stop_buffer_pct: float,
    top_n: int,
    out_csv: Optional[str],
    ensure_year_cache: bool,
) -> pd.DataFrame:
    rows: List[Dict] = []
    start_dt = pd.Timestamp(start_date, tz="UTC")
    end_dt = pd.Timestamp(end_date, tz="UTC")
    if end_dt < start_dt:
        raise SystemExit("end-date must be >= start-date")

    for tf in timeframes:
        tf = str(tf).strip()
        if not tf:
            continue

        if bool(ensure_year_cache):
            _ = get_crypto_data(
                symbol,
                start_dt.strftime("%Y-%m-%d"),
                end_dt.strftime("%Y-%m-%d"),
                tf,
                project_root,
            )
        tf_params = _indicator_params_for_tf(tf)
        warmup = max(200, int(tf_params["cci_period"]) * 3)
        start_fetch = (start_dt - _tf_to_timedelta(tf, warmup + 10)).strftime("%Y-%m-%d")
        end_fetch = end_dt.strftime("%Y-%m-%d")
        df_recent = get_crypto_data(symbol, start_fetch, end_fetch, tf, project_root)
        df_local = pd.DataFrame()
        if str(symbol).upper() == "LINKUSDT":
            df_local = _load_binance_data_vision_klines(project_root, "LINKUSDT", tf)
        df = _merge_history(df_local, df_recent)
        if df.empty:
            continue
        if "dt" not in df.columns:
            df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df["dt"] = pd.to_datetime(df["dt"], utc=True, errors="coerce")
        df = df[(df["dt"] >= (start_dt - _tf_to_timedelta(tf, warmup + 10))) & (df["dt"] <= end_dt)].reset_index(drop=True)
        if df.empty:
            continue

        high = df["high"].astype(float).tolist()
        low = df["low"].astype(float).tolist()
        close = df["close"].astype(float).tolist()
        volume = df["volume"].astype(float).tolist()

        df["cci"] = cci_tv(high, low, close, period=int(tf_params["cci_period"]))
        df["asi"] = asi_by_market(df, market="crypto")["ASI"]
        df["pvt"] = pvt_tv(close, volume)
        macd_line, macd_signal, macd_hist = macd_tv(close, fast_period=12, slow_period=26, signal_period=9)
        df["macd_line"] = macd_line
        df["macd_hist"] = macd_hist
        adx, di_plus, di_minus = dmi_tv(high, low, close, period=14, adx_smoothing=14)
        df["dx"] = [abs(p - m) for p, m in zip(di_plus, di_minus)]
        kvo, ks = klinger_oscillator_tv(high, low, close, volume, fast=34, slow=55, signal=13)
        df["kvo"] = kvo
        df["klinger_signal"] = ks
        df["mfi"] = mfi_tv(high, low, close, volume, period=14)
        stoch_k, stoch_d = _compute_stoch(
            df,
            high_col="high",
            low_col="low",
            close_col="close",
            k_period=int(tf_params["stoch_k_period"]),
            k_smooth_period=int(tf_params["stoch_k_smooth"]),
            d_period=int(tf_params["stoch_d_period"]),
        )
        df["stoch_k"] = stoch_k
        df["stoch_d"] = stoch_d

        series_to_col = {
            "price": "close",
            "asi": "asi",
            "pvt": "pvt",
            "macd_line": "macd_line",
            "dmi_dx": "dx",
            "klinger_kvo": "kvo",
            "klinger_signal": "klinger_signal",
            "mfi": "mfi",
            "stoch_k": "stoch_k",
            "stoch_d": "stoch_d",
        }

        start_i = int(df.index[df["dt"] >= start_dt].min()) if (df["dt"] >= start_dt).any() else 0

        max_k = int(max_combo_size)
        if max_k <= 0:
            max_k = int(len(series_universe))
        max_k = min(int(max_k), int(len(series_universe)))

        for k in range(1, max_k + 1):
            for combo in itertools.combinations(series_universe, k):
                combo_list = list(combo)
                trades = simulate_trades_from_stream(
                    df,
                    series_to_col=series_to_col,
                    mode="confluence",
                    signal_from="price",
                    selected_series=combo_list,
                    cci_col="cci",
                    cci_low=float(cci_low),
                    cci_high=float(cci_high),
                    macd_hist_col="macd_hist",
                    trade_direction=str(trade_direction),
                    min_confluence=int(len(combo_list)),
                    use_fixed_stop=bool(use_fixed_stop),
                    stop_buffer_pct=float(stop_buffer_pct),
                    stop_ref_series="price",
                    start_i=int(start_i),
                )

                pcts = [float(t.get("pct") or 0.0) for t in trades]
                equity = float(sum(pcts))
                curve = []
                cur = 0.0
                for p in pcts:
                    cur += float(p)
                    curve.append(float(cur))
                max_dd = _max_drawdown(curve)

                if float(max_dd) <= 0.0:
                    if float(equity) > 0.0:
                        eq_dd_ratio = float("inf")
                    elif float(equity) < 0.0:
                        eq_dd_ratio = float("-inf")
                    else:
                        eq_dd_ratio = 0.0
                else:
                    eq_dd_ratio = float(equity) / float(max_dd)

                maes = []
                for t in trades:
                    mae = _trade_mae_pct(df, t)
                    if mae is not None and math.isfinite(float(mae)):
                        maes.append(float(mae))
                worst_mae = float(min(maes)) if maes else 0.0

                wins = sum(1 for p in pcts if float(p) > 0.0)
                n_trades = int(len(pcts))
                winrate = (100.0 * float(wins) / float(n_trades)) if n_trades > 0 else 0.0

                rows.append(
                    {
                        "symbol": str(symbol),
                        "timeframe": str(tf),
                        "combo": ",".join(combo_list),
                        "k": int(len(combo_list)),
                        "n_trades": int(n_trades),
                        "equity": float(equity),
                        "max_dd": float(max_dd),
                        "eq_dd_ratio": float(eq_dd_ratio),
                        "worst_mae": float(worst_mae),
                        "winrate": float(winrate),
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["timeframe", "eq_dd_ratio", "equity"], ascending=[True, False, False]).reset_index(drop=True)
    if out_csv:
        Path(out_csv).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
    if int(top_n) > 0:
        def _print_block(title: str, df_block: pd.DataFrame, *, show_tf: bool) -> None:
            print(f"\n{title}")
            for _, r in df_block.iterrows():
                tf_part = f"tf={r['timeframe']} " if bool(show_tf) else ""
                print(
                    f"{tf_part}combo={r['combo']} min_conf={int(r['k'])} n={int(r['n_trades'])} "
                    f"equity={float(r['equity']):+.3f}% dd={float(r['max_dd']):.3f}% ratio={float(r['eq_dd_ratio']):.3f} "
                    f"worst_mae={float(r['worst_mae']):+.3f}% winrate={float(r['winrate']):.1f}%"
                )

        print("\n=== Grid search results (TOP by ratio) ===")

        for tf in sorted(out["timeframe"].astype(str).unique().tolist()):
            df_tf = out[out["timeframe"].astype(str) == str(tf)].copy()
            df_tf = df_tf.sort_values(["eq_dd_ratio", "equity"], ascending=[False, False])
            print(f"\n--- timeframe={tf} (tested={len(df_tf)}) ---")

            _print_block(
                f"[TOP {int(top_n)}] ALL combos",
                df_tf.head(int(top_n)),
                show_tf=False,
            )

            df_single = df_tf[df_tf["k"].astype(int) == 1]
            _print_block(
                f"[TOP {int(top_n)}] SINGLE (k=1)",
                df_single.head(int(top_n)),
                show_tf=False,
            )

            df_conf = df_tf[df_tf["k"].astype(int) >= 2]
            _print_block(
                f"[TOP {int(top_n)}] CONFLUENCE (k>=2)",
                df_conf.head(int(top_n)),
                show_tf=False,
            )

        df_global = out.sort_values(["eq_dd_ratio", "equity"], ascending=[False, False])
        print(f"\n=== Global TOP (all timeframes) tested={len(df_global)} ===")
        _print_block(
            f"[TOP {int(top_n)}] ALL combos",
            df_global.head(int(top_n)),
            show_tf=True,
        )
        df_global_single = df_global[df_global["k"].astype(int) == 1]
        _print_block(
            f"[TOP {int(top_n)}] SINGLE (k=1)",
            df_global_single.head(int(top_n)),
            show_tf=True,
        )
        df_global_conf = df_global[df_global["k"].astype(int) >= 2]
        _print_block(
            f"[TOP {int(top_n)}] CONFLUENCE (k>=2)",
            df_global_conf.head(int(top_n)),
            show_tf=True,
        )
    return out

def main():
    parser = argparse.ArgumentParser(description="Demo Zerem – Détection de structures multi-indicateurs")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbole Bybit")
    parser.add_argument("--timeframe", default="5m", help="Timeframe")
    parser.add_argument("--limit", type=int, default=1000, help="Nombre de bougies à télécharger")
    parser.add_argument("--start-date", help="Date de début (YYYY-MM-DD), sinon utilise limit")
    parser.add_argument("--end-date", help="Date de fin (YYYY-MM-DD), sinon utilise limit")
    parser.add_argument("--cci-low", type=float, default=-100.0, help="Seuil bas CCI pour creux")
    parser.add_argument("--cci-high", type=float, default=100.0, help="Seuil haut CCI pour sommets")
    parser.add_argument("--mode", choices=["single", "confluence"], default="confluence", help="Mode de sortie")
    parser.add_argument("--signal-from", default="price", help="Série à utiliser en mode single")
    parser.add_argument("--series", help="Liste de séries (séparées par virgule) à inclure en mode confluence. Si absent : toutes.")
    parser.add_argument("--simulate-trades", action="store_true", help="Simule un trading bougie-par-bougie via structure + timing MACD")
    parser.add_argument("--trade-direction", choices=["long", "short", "both"], default="both", help="Direction trading")
    parser.add_argument("--trade-min-confluence", type=int, default=1, help="Seuil min de confluence pour déclencher un signal trade")
    parser.add_argument("--use-fixed-stop", action="store_true", help="Active un stop fixe basé sur l’extrême price confirmé précédent")
    parser.add_argument("--stop-buffer-pct", type=float, default=0.0, help="Buffer stop en pourcentage (ex: 0.001 = 0.1%)")
    parser.add_argument("--grid-search", action="store_true", help="Teste toutes les combinaisons de séries et timeframes et classe par equity")
    parser.add_argument("--grid-timeframes", default="5m,15m,1h,2h,6h,1d", help="Liste timeframes séparées par virgule")
    parser.add_argument("--grid-year", type=int, help="Raccourci: backtest sur toute l’année YYYY (ex: 2025)")
    parser.add_argument("--grid-ensure-year-cache", action="store_true", help="Construit le cache CSV {symbol}_{tf}_{YYYY-01-01}_{YYYY-12-31}.csv si absent")
    parser.add_argument("--grid-series", help="Sous-ensemble de séries pour le grid-search (sinon toutes)")
    parser.add_argument("--grid-max-combo-size", type=int, default=0, help="Taille max des combinaisons (0 = toutes)")
    parser.add_argument("--grid-top", type=int, default=30, help="Nombre de résultats à afficher")
    parser.add_argument("--grid-out-csv", help="Chemin CSV de sortie")
    args = parser.parse_args()

    if bool(args.grid_search):
        if args.grid_year:
            y = int(args.grid_year)
            args.start_date = f"{y:04d}-01-01"
            args.end_date = f"{y:04d}-12-31"
        if not (args.start_date and args.end_date):
            raise SystemExit("grid-search requires --start-date and --end-date (or --grid-year)")

        tmp_tf = [x.strip() for x in str(args.grid_timeframes or "").split(",") if x.strip()]
        if not tmp_tf:
            raise SystemExit("grid-timeframes is empty")

        series_to_col = {
            "price": "close",
            "asi": "asi",
            "pvt": "pvt",
            "macd_line": "macd_line",
            "dmi_dx": "dx",
            "klinger_kvo": "kvo",
            "klinger_signal": "klinger_signal",
            "mfi": "mfi",
            "stoch_k": "stoch_k",
            "stoch_d": "stoch_d",
        }
        all_series = list(series_to_col.keys())
        grid_series = _parse_series_list(args.grid_series) or all_series
        unknown = [s for s in grid_series if s not in all_series]
        if unknown:
            raise SystemExit(f"Unknown series in --grid-series: {unknown}")

        _grid_search(
            symbol=str(args.symbol),
            start_date=str(args.start_date),
            end_date=str(args.end_date),
            timeframes=tmp_tf,
            series_universe=list(grid_series),
            max_combo_size=int(args.grid_max_combo_size),
            cci_low=float(args.cci_low),
            cci_high=float(args.cci_high),
            trade_direction="both",
            use_fixed_stop=True,
            stop_buffer_pct=0.01,
            top_n=int(args.grid_top),
            out_csv=str(args.grid_out_csv) if args.grid_out_csv else None,
            ensure_year_cache=bool(args.grid_ensure_year_cache),
        )
        return

    tf_params = _indicator_params_for_tf(args.timeframe)
    warmup = max(200, int(tf_params["cci_period"]) * 3)
    logger.info(
        f"Indicator params: cci_period={tf_params['cci_period']} stoch=({tf_params['stoch_k_period']},{tf_params['stoch_k_smooth']},{tf_params['stoch_d_period']})"
    )

    # Charger les données récentes (Bybit) + historique local pour LINK si dispo
    if args.start_date and args.end_date:
        df_recent = get_crypto_data(args.symbol, args.start_date, args.end_date, args.timeframe, project_root)
    else:
        end_dt = pd.Timestamp.now(tz="UTC")
        start_dt = end_dt - _tf_to_timedelta(args.timeframe, args.limit + warmup + 10)
        df_recent = get_crypto_data(args.symbol, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), args.timeframe, project_root)

    df_local = pd.DataFrame()
    if str(args.symbol).upper() == "LINKUSDT":
        df_local = _load_binance_data_vision_klines(project_root, "LINKUSDT", args.timeframe)

    df = _merge_history(df_local, df_recent)
    if len(df) > args.limit + warmup:
        df = df.tail(args.limit + warmup).reset_index(drop=True)

    if df.empty:
        logger.error("Aucune donnée chargée")
        return

    logger.info(f"Loaded {len(df)} rows for {args.symbol} {args.timeframe}")

    # Calculer les indicateurs nécessaires
    high = df["high"].astype(float).tolist()
    low = df["low"].astype(float).tolist()
    close = df["close"].astype(float).tolist()
    volume = df["volume"].astype(float).tolist()

    df["cci"] = cci_tv(high, low, close, period=int(tf_params["cci_period"]))
    df["asi"] = asi_by_market(df, market="crypto")["ASI"]
    df["pvt"] = pvt_tv(close, volume)
    macd_line, macd_signal, macd_hist = macd_tv(close, fast_period=12, slow_period=26, signal_period=9)
    df["macd_line"] = macd_line
    df["macd_hist"] = macd_hist
    adx, di_plus, di_minus = dmi_tv(high, low, close, period=14, adx_smoothing=14)
    df["dx"] = [abs(p - m) for p, m in zip(di_plus, di_minus)]
    kvo, ks = klinger_oscillator_tv(high, low, close, volume, fast=34, slow=55, signal=13)
    df["kvo"] = kvo
    df["klinger_signal"] = ks
    df["mfi"] = mfi_tv(high, low, close, volume, period=14)
    stoch_k, stoch_d = _compute_stoch(
        df,
        high_col="high",
        low_col="low",
        close_col="close",
        k_period=int(tf_params["stoch_k_period"]),
        k_smooth_period=int(tf_params["stoch_k_smooth"]),
        d_period=int(tf_params["stoch_d_period"]),
    )
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_d

    # Délimiter zones CCI
    zones = delimit_zones_by_cci(df, cci_col="cci", low_threshold=args.cci_low, high_threshold=args.cci_high)
    for zone_id, z in enumerate(zones):
        z["zone_id"] = zone_id
    logger.info(f"Zones CCI détectées : {len(zones)}")

    series_to_col = {
        "price": "close",
        "asi": "asi",
        "pvt": "pvt",
        "macd_line": "macd_line",
        "dmi_dx": "dx",
        "klinger_kvo": "kvo",
        "klinger_signal": "klinger_signal",
        "mfi": "mfi",
        "stoch_k": "stoch_k",
        "stoch_d": "stoch_d",
    }

    all_series = list(series_to_col.keys())
    by_series_zone_results: Dict[str, List[Dict]] = {}
    for serie in all_series:
        by_series_zone_results[serie] = analyze_series_by_zones(df, zones, series_to_col[serie], serie)

    # Log rapide : nombre de structures déclenchées par série
    for serie, zone_results in by_series_zone_results.items():
        n_struct = sum(1 for r in zone_results if r.get("structure"))
        logger.info(f"{serie}: {n_struct} structures détectées")

    if args.mode == "single":
        serie = args.signal_from
        if serie not in by_series_zone_results:
            raise SystemExit(f"Unknown series for --signal-from: {serie}")
        print(f"\n=== Mode single (serie={serie}) ===")
        for r in by_series_zone_results[serie]:
            structure = r.get("structure")
            if not structure:
                continue
            zone = r["zone"]
            zone_type = zone["type"]
            zstart = r.get("zone_start_dt")
            zend = r.get("zone_end_dt")
            cur = r["extremum"]
            prev = r.get("prev_extremum")
            prev_part = ""
            if prev is not None:
                prev_part = f" prev_{prev['type']}={prev['value']:.4f} ({prev.get('dt')})"
            print(
                f"zone_{zone_type} [{zstart} -> {zend}]\t"
                f"{cur['type']}={cur['value']:.4f} ({cur.get('dt')})\t"
                f"{structure}{prev_part}"
            )

    elif args.mode == "confluence":
        selected_series = _parse_series_list(args.series) or all_series
        unknown = [s for s in selected_series if s not in by_series_zone_results]
        if unknown:
            raise SystemExit(f"Unknown series in --series: {unknown}")

        print(f"\n=== Mode confluence (series={selected_series}) ===")

        # Indexer par zone_id
        results_by_zone: Dict[int, List[Dict]] = {}
        for serie in selected_series:
            for r in by_series_zone_results[serie]:
                zid = r.get("zone_id")
                if zid is None:
                    continue
                results_by_zone.setdefault(int(zid), []).append(r)

        for zid in sorted(results_by_zone.keys()):
            zone_results = results_by_zone[zid]
            if not zone_results:
                continue
            zone = zone_results[0]["zone"]
            zone_type = zone["type"]
            zstart = zone_results[0].get("zone_start_dt")
            zend = zone_results[0].get("zone_end_dt")

            if zone_type == "creux":
                confluent = [r for r in zone_results if r.get("structure") == "structure_haussiere"]
                if not confluent:
                    continue
                confluent_series = [r["serie"] for r in confluent]
                print(f"\nZone creux [{zstart} -> {zend}]")
                print(f"Séries en confluence haussière : {', '.join(confluent_series)}")
                for r in confluent:
                    cur = r["extremum"]
                    prev = r.get("prev_extremum")
                    prev_txt = ""
                    if prev is not None:
                        prev_txt = f" (prev_creux={prev['value']:.4f})"
                    print(f"  {r['serie']} : creux={cur['value']:.4f}  structure_haussiere{prev_txt}")

            elif zone_type == "sommet":
                confluent = [r for r in zone_results if r.get("structure") == "structure_baissiere"]
                if not confluent:
                    continue
                confluent_series = [r["serie"] for r in confluent]
                print(f"\nZone sommet [{zstart} -> {zend}]")
                print(f"Séries en confluence baissière : {', '.join(confluent_series)}")
                for r in confluent:
                    cur = r["extremum"]
                    prev = r.get("prev_extremum")
                    prev_txt = ""
                    if prev is not None:
                        prev_txt = f" (prev_sommet={prev['value']:.4f})"
                    print(f"  {r['serie']} : sommet={cur['value']:.4f}  structure_baissiere{prev_txt}")

    if bool(args.simulate_trades):
        selected_series = _parse_series_list(args.series) or all_series
        start_i = max(0, int(len(df) - int(args.limit)))
        trades = simulate_trades_from_stream(
            df,
            series_to_col=series_to_col,
            mode=str(args.mode),
            signal_from=str(args.signal_from),
            selected_series=list(selected_series),
            cci_col="cci",
            cci_low=float(args.cci_low),
            cci_high=float(args.cci_high),
            macd_hist_col="macd_hist",
            trade_direction=str(args.trade_direction),
            min_confluence=int(args.trade_min_confluence),
            use_fixed_stop=bool(args.use_fixed_stop),
            stop_buffer_pct=float(args.stop_buffer_pct),
            stop_ref_series="price",
            start_i=int(start_i),
        )

        print(
            f"\n=== Trades (direction={args.trade_direction}, min_confluence={args.trade_min_confluence}, start_i={start_i}) ==="
        )
        if not trades:
            print("No trades")
        else:
            pcts = [float(t.get("pct") or 0.0) for t in trades]
            wins = [p for p in pcts if float(p) > 0.0]
            n_trades = int(len(trades))
            winrate = (100.0 * float(len(wins)) / float(n_trades)) if n_trades > 0 else 0.0
            avg_pct = (float(sum(pcts)) / float(n_trades)) if n_trades > 0 else 0.0
            sum_pct = float(sum(pcts))
            print(f"Summary: n={n_trades} winrate={winrate:.1f}% avg_pct={avg_pct:+.3f}% sum_pct={sum_pct:+.3f}%")
            for t in trades:
                side = "LONG" if int(t.get("side") or 0) == 1 else "SHORT"
                pct = float(t.get("pct") or 0.0)
                ep = float(t.get("entry_price") or 0.0)
                xp = float(t.get("exit_price") or 0.0)
                edt = t.get("entry_dt")
                xdt = t.get("exit_dt")
                sp = t.get("stop_price")
                zr = t.get("signal") or {}
                zid = zr.get("zone_id")
                zok = zr.get("series_ok") or []
                zok_str = ",".join([str(x) for x in list(zok)])
                sp_txt = ""
                if sp is not None:
                    spv = _safe_float(sp)
                    if spv is not None:
                        sp_txt = f" stop={float(spv):.6f}"
                print(f"{side} {edt} @ {ep:.6f} -> {xdt} @ {xp:.6f} {pct:+.3f}% zone={zid} series_ok={zok_str}{sp_txt}")

if __name__ == "__main__":
    main()
