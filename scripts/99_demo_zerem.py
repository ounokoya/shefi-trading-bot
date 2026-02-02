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
import re
import sys
import time
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
from libs.zerem.metrics import score_trades
from libs.zerem.trades import simulate_trades_from_stream as simulate_trades_from_stream_lib

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
    extreme_confirm_bars: int,
    entry_require_hist_abs_growth: bool,
    entry_cci_tf_cols: List[str],
    exit_b_mode: str,
) -> List[Dict]:
    cci = pd.to_numeric(df[cci_col], errors="coerce").astype(float).to_numpy()
    macd_hist = pd.to_numeric(df[macd_hist_col], errors="coerce").astype(float).to_numpy()
    if "macd_line" in df.columns:
        macd_line = pd.to_numeric(df["macd_line"], errors="coerce").astype(float).to_numpy()
    else:
        macd_line = pd.to_numeric(pd.Series([math.nan] * int(len(df))), errors="coerce").astype(float).to_numpy()
    if "macd_signal" in df.columns:
        macd_signal = pd.to_numeric(df["macd_signal"], errors="coerce").astype(float).to_numpy()
    else:
        macd_signal = pd.to_numeric(pd.Series([math.nan] * int(len(df))), errors="coerce").astype(float).to_numpy()
    if "stoch_k" in df.columns:
        stoch_k = pd.to_numeric(df["stoch_k"], errors="coerce").astype(float).to_numpy()
    else:
        stoch_k = pd.to_numeric(pd.Series([math.nan] * int(len(df))), errors="coerce").astype(float).to_numpy()
    if "stoch_d" in df.columns:
        stoch_d = pd.to_numeric(df["stoch_d"], errors="coerce").astype(float).to_numpy()
    else:
        stoch_d = pd.to_numeric(pd.Series([math.nan] * int(len(df))), errors="coerce").astype(float).to_numpy()
    if "kvo" in df.columns:
        kvo = pd.to_numeric(df["kvo"], errors="coerce").astype(float).to_numpy()
    else:
        kvo = pd.to_numeric(pd.Series([math.nan] * int(len(df))), errors="coerce").astype(float).to_numpy()
    if "klinger_signal" in df.columns:
        klinger_signal = pd.to_numeric(df["klinger_signal"], errors="coerce").astype(float).to_numpy()
    else:
        klinger_signal = pd.to_numeric(pd.Series([math.nan] * int(len(df))), errors="coerce").astype(float).to_numpy()

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
    zone_start_i: Optional[int] = None

    potential: Dict[str, Optional[Dict]] = {s: None for s in tracking_series}
    potential_ready: Dict[str, Optional[Dict]] = {s: None for s in tracking_series}

    pending_bull: Optional[Dict] = None
    pending_bear: Optional[Dict] = None

    last_bull_hist_cross_i: Optional[int] = None
    last_bear_hist_cross_i: Optional[int] = None

    cci_tf_map: Dict[str, object] = {}
    for c in list(entry_cci_tf_cols or []):
        if c in df.columns:
            cci_tf_map[str(c)] = pd.to_numeric(df[str(c)], errors="coerce").astype(float).to_numpy()

    def _cci_tf_confluence_ok(i0: int, zt: str) -> bool:
        if not cci_tf_map:
            return True
        if int(i0) < 0:
            return False
        for c, arr in cci_tf_map.items():
            try:
                x = float(arr[int(i0)])
            except Exception:
                return False
            if not math.isfinite(float(x)):
                return False
            if str(zt) == "creux":
                if float(x) >= float(cci_low):
                    return False
            elif str(zt) == "sommet":
                if float(x) <= float(cci_high):
                    return False
        return True

    pos = 0
    entry: Optional[Dict] = None
    trades: List[Dict] = []
    exit_arm_i: Optional[int] = None

    def _close_position(i: int, reason: str, exit_price_override: Optional[float] = None) -> None:
        nonlocal pos, entry, trades, exit_arm_i
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
        exit_arm_i = None

    def _cross_up(prev_a: float, a: float, prev_b: float, b: float) -> bool:
        if not (math.isfinite(float(prev_a)) and math.isfinite(float(a)) and math.isfinite(float(prev_b)) and math.isfinite(float(b))):
            return False
        return float(prev_a) <= float(prev_b) and float(a) > float(b)

    def _cross_down(prev_a: float, a: float, prev_b: float, b: float) -> bool:
        if not (math.isfinite(float(prev_a)) and math.isfinite(float(a)) and math.isfinite(float(prev_b)) and math.isfinite(float(b))):
            return False
        return float(prev_a) >= float(prev_b) and float(a) < float(b)

    def _hist_abs_growth(prev_h: float, h: float) -> bool:
        if not (math.isfinite(float(prev_h)) and math.isfinite(float(h))):
            return False
        return abs(float(h)) > abs(float(prev_h))

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
        nonlocal pos, entry, exit_arm_i
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
        exit_arm_i = None

    def _apply_signal(i: int, direction: str, signal: Dict) -> None:
        nonlocal pos, pending_bull, pending_bear
        td = str(trade_direction or "both").strip().lower()

        if str(direction) == "bull":
            if td == "short":
                return
            if pos == 0:
                _open_position(i, 1, signal)
            pending_bull = None
            return

        if str(direction) == "bear":
            if td == "long":
                pending_bear = None
                return
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

        if int(i) >= int(start_i) and pos != 0 and entry is not None and exit_arm_i is not None and int(i) > 0 and int(i) > int(exit_arm_i):
            em = str(exit_b_mode or "macd").strip().lower()
            if em == "none":
                pass
            if em == "macd":
                if int(pos) == 1 and _cross_down(float(macd_line[i - 1]), float(macd_line[i]), float(macd_signal[i - 1]), float(macd_signal[i])):
                    _close_position(i, "exit_macd")
                elif int(pos) == -1 and _cross_up(float(macd_line[i - 1]), float(macd_line[i]), float(macd_signal[i - 1]), float(macd_signal[i])):
                    _close_position(i, "exit_macd")
            elif em == "stoch":
                if int(pos) == 1 and _cross_down(float(stoch_k[i - 1]), float(stoch_k[i]), float(stoch_d[i - 1]), float(stoch_d[i])):
                    _close_position(i, "exit_stoch")
                elif int(pos) == -1 and _cross_up(float(stoch_k[i - 1]), float(stoch_k[i]), float(stoch_d[i - 1]), float(stoch_d[i])):
                    _close_position(i, "exit_stoch")
            elif em == "klinger":
                if int(pos) == 1 and _cross_down(float(kvo[i - 1]), float(kvo[i]), float(klinger_signal[i - 1]), float(klinger_signal[i])):
                    _close_position(i, "exit_klinger")
                elif int(pos) == -1 and _cross_up(float(kvo[i - 1]), float(kvo[i]), float(klinger_signal[i - 1]), float(klinger_signal[i])):
                    _close_position(i, "exit_klinger")

        v = cci[i] if i < len(cci) else math.nan
        prev = cci[i - 1] if i > 0 and i - 1 < len(cci) else None

        if not in_zone:
            if math.isfinite(float(v)) and float(v) < float(cci_low) and (prev is None or (math.isfinite(float(prev)) and float(prev) >= float(cci_low)) or (prev is not None and not math.isfinite(float(prev)))):
                in_zone = True
                zone_type = "creux"
                zone_id += 1
                zone_start_i = int(i)
                for s in tracking_series:
                    potential[s] = None
            elif math.isfinite(float(v)) and float(v) > float(cci_high) and (prev is None or (math.isfinite(float(prev)) and float(prev) <= float(cci_high)) or (prev is not None and not math.isfinite(float(prev)))):
                in_zone = True
                zone_type = "sommet"
                zone_id += 1
                zone_start_i = int(i)
                for s in tracking_series:
                    potential[s] = None
        else:
            if zone_type == "creux" and math.isfinite(float(v)) and float(v) >= float(cci_low):
                struct_ok: Dict[str, bool] = {}
                per_series: Dict[str, Dict] = {}
                for s in tracking_series:
                    prev_conf = last_confirmed[s]["creux"]
                    cur_pot = potential.get(s)
                    confirmed = None
                    if cur_pot is not None:
                        confirmed = {**cur_pot, "status": "confirmed", "confirmed_i": int(i)}
                    ok = False
                    if prev_conf is not None and confirmed is not None:
                        if float(confirmed["value"]) > float(prev_conf["value"]):
                            ok = True
                    if s in base_series:
                        struct_ok[s] = bool(ok)
                    per_series[s] = {"potential": confirmed, "prev_confirmed": prev_conf}

                n_ok = sum(1 for v0 in struct_ok.values() if bool(v0))
                if int(n_ok) >= int(max(1, min_confluence)) and _cci_tf_confluence_ok(max(0, int(i) - 1), "creux"):
                    pending_bull = {
                        "zone_id": int(zone_id),
                        "zone_type": "creux",
                        "created_i": int(i),
                        "created_dt": df.iloc[i].get("dt"),
                        "extreme_start_i": int(zone_start_i) if zone_start_i is not None else int(i),
                        "mode": str(mode),
                        "series_ok": [s for s, ok in struct_ok.items() if bool(ok)],
                        "per_series": per_series,
                    }
                for s in tracking_series:
                    if potential[s] is not None:
                        last_confirmed[s]["creux"] = {**potential[s], "status": "confirmed", "confirmed_i": int(i)}
                    potential[s] = None
                    potential_ready[s] = None
                in_zone = False
                zone_type = None
                zone_start_i = None
            elif zone_type == "sommet" and math.isfinite(float(v)) and float(v) <= float(cci_high):
                struct_ok: Dict[str, bool] = {}
                per_series: Dict[str, Dict] = {}
                for s in tracking_series:
                    prev_conf = last_confirmed[s]["sommet"]
                    cur_pot = potential.get(s)
                    confirmed = None
                    if cur_pot is not None:
                        confirmed = {**cur_pot, "status": "confirmed", "confirmed_i": int(i)}
                    ok = False
                    if prev_conf is not None and confirmed is not None:
                        if float(confirmed["value"]) < float(prev_conf["value"]):
                            ok = True
                    if s in base_series:
                        struct_ok[s] = bool(ok)
                    per_series[s] = {"potential": confirmed, "prev_confirmed": prev_conf}

                n_ok = sum(1 for v0 in struct_ok.values() if bool(v0))
                if int(n_ok) >= int(max(1, min_confluence)) and _cci_tf_confluence_ok(max(0, int(i) - 1), "sommet"):
                    pending_bear = {
                        "zone_id": int(zone_id),
                        "zone_type": "sommet",
                        "created_i": int(i),
                        "created_dt": df.iloc[i].get("dt"),
                        "extreme_start_i": int(zone_start_i) if zone_start_i is not None else int(i),
                        "mode": str(mode),
                        "series_ok": [s for s, ok in struct_ok.items() if bool(ok)],
                        "per_series": per_series,
                    }
                for s in tracking_series:
                    if potential[s] is not None:
                        last_confirmed[s]["sommet"] = {**potential[s], "status": "confirmed", "confirmed_i": int(i)}
                    potential[s] = None
                    potential_ready[s] = None
                in_zone = False
                zone_type = None
                zone_start_i = None

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
                prev_ready = potential_ready.get(s)
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
                    potential_ready[s] = None
                    if int(extreme_confirm_bars) <= 0 and entry is not None and s == stop_ref and int(i) >= int(start_i):
                        if int(pos) == 1 and str(zone_type) == "sommet" and int(entry.get("entry_i") or 0) <= int(i):
                            exit_arm_i = int(i)
                        elif int(pos) == -1 and str(zone_type) == "creux" and int(entry.get("entry_i") or 0) <= int(i):
                            exit_arm_i = int(i)
                else:
                    if zone_type == "creux" and float(vv) < float(cur["value"]):
                        potential[s] = {**cur, "idx": int(i), "value": float(vv), "ts": df.iloc[i].get("ts"), "dt": df.iloc[i].get("dt")}
                        potential_ready[s] = None
                        if int(extreme_confirm_bars) <= 0 and entry is not None and s == stop_ref and int(i) >= int(start_i):
                            if int(pos) == -1 and int(entry.get("entry_i") or 0) <= int(i):
                                exit_arm_i = int(i)
                    elif zone_type == "sommet" and float(vv) > float(cur["value"]):
                        potential[s] = {**cur, "idx": int(i), "value": float(vv), "ts": df.iloc[i].get("ts"), "dt": df.iloc[i].get("dt")}
                        potential_ready[s] = None
                        if int(extreme_confirm_bars) <= 0 and entry is not None and s == stop_ref and int(i) >= int(start_i):
                            if int(pos) == 1 and int(entry.get("entry_i") or 0) <= int(i):
                                exit_arm_i = int(i)

                pot_cur = potential.get(s)
                if pot_cur is None:
                    continue

                pot: Optional[Dict] = None
                if int(extreme_confirm_bars) <= 0:
                    pot = pot_cur
                else:
                    pot_i = int(pot_cur.get("idx") or 0)
                    if int(i) >= int(pot_i) + int(extreme_confirm_bars):
                        pot = pot_cur
                        if prev_ready is None:
                            potential_ready[s] = pot_cur
                            if entry is not None and s == stop_ref and int(i) >= int(start_i):
                                if int(pos) == 1 and str(zone_type) == "sommet" and int(entry.get("entry_i") or 0) <= int(pot_i):
                                    exit_arm_i = int(i)
                                elif int(pos) == -1 and str(zone_type) == "creux" and int(entry.get("entry_i") or 0) <= int(pot_i):
                                    exit_arm_i = int(i)
                    else:
                        potential_ready[s] = None

                prev_conf = last_confirmed[s][zone_type]
                ok = False
                if prev_conf is not None and pot is not None:
                    if zone_type == "creux" and float(pot["value"]) > float(prev_conf["value"]):
                        ok = True
                    elif zone_type == "sommet" and float(pot["value"]) < float(prev_conf["value"]):
                        ok = True

                if s in base_series:
                    struct_ok[s] = bool(ok)
                per_series[s] = {"potential": pot, "prev_confirmed": prev_conf}

            n_ok = sum(1 for v0 in struct_ok.values() if bool(v0))
            if int(n_ok) >= int(max(1, min_confluence)) and _cci_tf_confluence_ok(int(i), str(zone_type)):
                signal = {
                    "zone_id": int(zone_id),
                    "zone_type": str(zone_type),
                    "created_i": int(i),
                    "created_dt": df.iloc[i].get("dt"),
                    "extreme_start_i": int(zone_start_i) if zone_start_i is not None else int(i),
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

        ok_abs = True
        if bool(entry_require_hist_abs_growth):
            ok_abs = _hist_abs_growth(float(macd_hist[i - 1]), float(macd_hist[i]))

        if _hist_cross_up(float(macd_hist[i - 1]), float(macd_hist[i])) and bool(ok_abs):
            last_bull_hist_cross_i = int(i)
        elif _hist_cross_down(float(macd_hist[i - 1]), float(macd_hist[i])) and bool(ok_abs):
            last_bear_hist_cross_i = int(i)

        if int(i) < int(start_i):
            continue

        if pending_bull is not None and float(macd_hist[i]) > 0.0:
            es = int(pending_bull.get("extreme_start_i") or pending_bull.get("created_i") or 0)
            if last_bull_hist_cross_i is not None and int(last_bull_hist_cross_i) >= int(es):
                _apply_signal(i, "bull", pending_bull)
        elif pending_bear is not None and float(macd_hist[i]) < 0.0:
            es = int(pending_bear.get("extreme_start_i") or pending_bear.get("created_i") or 0)
            if last_bear_hist_cross_i is not None and int(last_bear_hist_cross_i) >= int(es):
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


def _tf_to_minutes(timeframe: str) -> int:
    unit = str(timeframe).strip()[-1:]
    try:
        value = int(str(timeframe).strip()[:-1])
    except ValueError:
        value = 1
    minutes = int(value)
    if unit == "h":
        minutes = int(value) * 60
    elif unit == "d":
        minutes = int(value) * 60 * 24
    elif unit == "w":
        minutes = int(value) * 60 * 24 * 7
    return int(minutes)


def _tf_to_pandas_rule(timeframe: str) -> str:
    tf = str(timeframe).strip()
    unit = tf[-1:]
    try:
        value = int(tf[:-1])
    except ValueError:
        value = 1
    if unit == "m":
        return f"{int(value)}min"
    if unit == "h":
        return f"{int(value)}h"
    if unit == "d":
        return f"{int(value)}D"
    if unit == "w":
        return f"{int(value)}W"
    return f"{int(value)}min"


def _count_missing_klines_from_ts(ts: pd.Series, *, interval_ms: int) -> int:
    if interval_ms <= 0:
        return 0
    if ts is None:
        return 0
    s = pd.to_numeric(ts, errors="coerce").dropna().astype(int)
    if s.empty:
        return 0
    s = s.drop_duplicates().sort_values()
    diffs = s.diff().dropna().astype(int)
    missing = 0
    for d in diffs.tolist():
        if int(d) <= int(interval_ms):
            continue
        missing += max(0, (int(d) // int(interval_ms)) - 1)
    return int(missing)


def _cache_range_is_complete(
    df: pd.DataFrame,
    *,
    timeframe: str,
    want_start: pd.Timestamp,
    want_end: pd.Timestamp,
    max_missing: int,
) -> bool:
    if df is None or df.empty or "ts" not in df.columns:
        return False
    minutes = _tf_to_minutes(str(timeframe))
    if int(minutes) <= 0:
        return True

    interval_ms = int(minutes) * 60 * 1000

    ws = pd.Timestamp(want_start)
    we = pd.Timestamp(want_end)
    if ws.tz is None:
        ws = ws.tz_localize("UTC")
    else:
        ws = ws.tz_convert("UTC")
    if we.tz is None:
        we = we.tz_localize("UTC")
    else:
        we = we.tz_convert("UTC")
    start_ms = int(ws.timestamp() * 1000)
    end_ms = int(we.timestamp() * 1000)

    ts_all = pd.to_numeric(df["ts"], errors="coerce").dropna().astype(int)
    if ts_all.empty:
        return False
    ts_all = ts_all.drop_duplicates().sort_values()
    ts = ts_all[(ts_all >= int(start_ms)) & (ts_all <= int(end_ms))]
    if ts.empty:
        return False

    expected = int(((int(end_ms) - int(start_ms)) // int(interval_ms)) + 1) if int(end_ms) >= int(start_ms) else 0
    missing_by_count = max(0, int(expected) - int(len(ts))) if int(expected) > 0 else 0
    missing_by_gaps = _count_missing_klines_from_ts(ts, interval_ms=int(interval_ms))
    missing_total = max(int(missing_by_count), int(missing_by_gaps))
    return int(missing_total) <= int(max_missing)


def _ensure_cci_tf_column(df: pd.DataFrame, *, base_tf: str, target_tf: str) -> str:
    col = f"cci_tf_{str(target_tf)}"
    if col in df.columns:
        return col
    if "dt" not in df.columns:
        raise ValueError("Missing dt column")

    if _tf_to_minutes(str(target_tf)) < _tf_to_minutes(str(base_tf)):
        raise ValueError(f"target_tf must be >= base_tf: base_tf={base_tf} target_tf={target_tf}")

    rule = _tf_to_pandas_rule(str(target_tf))
    tmp = df[["dt", "open", "high", "low", "close", "volume"]].copy()
    tmp["dt"] = pd.to_datetime(tmp["dt"], utc=True, errors="coerce")
    tmp = tmp.dropna(subset=["dt"]).sort_values("dt")
    tmp = tmp.set_index("dt")
    res = tmp.resample(rule, label="right", closed="right").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    res = res.dropna(subset=["high", "low", "close"]).reset_index()
    if res.empty:
        df[col] = math.nan
        return col

    tf_params = _indicator_params_for_tf(str(target_tf))
    high = res["high"].astype(float).tolist()
    low = res["low"].astype(float).tolist()
    close = res["close"].astype(float).tolist()
    res[col] = cci_tv(high, low, close, period=int(tf_params["cci_period"]))
    res = res[["dt", col]].sort_values("dt")

    base_dt = df[["dt"]].copy()
    base_dt["dt"] = pd.to_datetime(base_dt["dt"], utc=True, errors="coerce")
    base_dt = base_dt.sort_values("dt")
    aligned = pd.merge_asof(base_dt, res, on="dt", direction="backward")
    df[col] = aligned[col].ffill().to_numpy()
    return col


def _parse_pct_list(raw: Optional[str]) -> List[float]:
    if raw is None:
        return []
    s = str(raw).strip()
    if not s:
        return []
    out: List[float] = []
    for part in s.split(","):
        p = str(part).strip()
        if not p:
            continue
        p = p.replace("%", "").strip()
        v = float(p)
        if v >= 1.0:
            v = v / 100.0
        if v < 0.0:
            raise SystemExit(f"Invalid pct value in list: {part}")
        out.append(float(v))
    return out


def _load_cached_klines_covering_range(
    *,
    project_root: Path,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    cache_validate: bool = True,
    cache_max_missing: int = 5,
) -> Optional[pd.DataFrame]:
    cache_dir = project_root / "data" / "raw" / "klines_cache"
    if not cache_dir.exists():
        return None
    pat = re.compile(
        rf"^{re.escape(str(symbol))}_{re.escape(str(timeframe))}_(\d{{4}}-\d{{2}}-\d{{2}})_(\d{{4}}-\d{{2}}-\d{{2}})\.csv$"
    )
    want_start = pd.Timestamp(start_date, tz="UTC")
    want_end = pd.Timestamp(end_date, tz="UTC")

    best_path: Optional[Path] = None
    best_span_days: Optional[int] = None
    for p in cache_dir.glob(f"{symbol}_{timeframe}_*.csv"):
        m = pat.match(p.name)
        if not m:
            continue
        s0, e0 = m.group(1), m.group(2)
        try:
            have_start = pd.Timestamp(s0, tz="UTC")
            have_end = pd.Timestamp(e0, tz="UTC")
        except Exception:
            continue
        if have_start <= want_start and have_end >= want_end:
            span_days = int((have_end - have_start).days)
            if best_span_days is None or span_days < best_span_days:
                best_span_days = span_days
                best_path = p

    if best_path is None:
        return None

    logging.info(f"Loading data from covering cache: {best_path}")
    df = pd.read_csv(best_path)
    if "open_time" not in df.columns and "ts" in df.columns:
        df["open_time"] = df["ts"]
    if "ts" in df.columns:
        df["ts"] = df["ts"].astype(int)
        df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    if "open_time" in df.columns:
        df["open_time"] = df["open_time"].astype(int)
    if not df.empty and "ts" in df.columns:
        df = df.sort_values("ts").reset_index(drop=True)

    if bool(cache_validate):
        ok = _cache_range_is_complete(
            df,
            timeframe=str(timeframe),
            want_start=want_start,
            want_end=want_end,
            max_missing=int(cache_max_missing),
        )
        if not bool(ok):
            logging.warning(
                f"[cache] incomplete file: deleting {best_path} (tf={timeframe} range={start_date}..{end_date} max_missing={cache_max_missing})"
            )
            try:
                best_path.unlink(missing_ok=True)
            except Exception as e:
                logging.warning(f"[cache] failed to delete {best_path}: {e}")
            return None
    return df


def _load_cached_klines_union_range(
    *,
    project_root: Path,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    cache_validate: bool = True,
    cache_max_missing: int = 5,
) -> Optional[pd.DataFrame]:
    cache_dir = project_root / "data" / "raw" / "klines_cache"
    if not cache_dir.exists():
        return None

    pat = re.compile(
        rf"^{re.escape(str(symbol))}_{re.escape(str(timeframe))}_(\d{{4}}-\d{{2}}-\d{{2}})_(\d{{4}}-\d{{2}}-\d{{2}})\.csv$"
    )
    want_start = pd.Timestamp(start_date, tz="UTC")
    want_end = pd.Timestamp(end_date, tz="UTC")

    candidates: List[Tuple[Path, pd.Timestamp, pd.Timestamp]] = []
    for p in cache_dir.glob(f"{symbol}_{timeframe}_*.csv"):
        m = pat.match(p.name)
        if not m:
            continue
        s0, e0 = m.group(1), m.group(2)
        try:
            have_start = pd.Timestamp(s0, tz="UTC")
            have_end = pd.Timestamp(e0, tz="UTC")
        except Exception:
            continue
        if have_end < want_start or have_start > want_end:
            continue
        candidates.append((p, have_start, have_end))

    if not candidates:
        return None

    # Prefer a minimal set of chunks that covers [want_start, want_end].
    # 1) Remove intervals fully contained in another interval.
    candidates.sort(key=lambda x: (x[1], -x[2].value))
    filtered: List[Tuple[Path, pd.Timestamp, pd.Timestamp]] = []
    for p, s, e in candidates:
        dominated = False
        for _, s2, e2 in filtered:
            if s2 <= s and e2 >= e:
                dominated = True
                break
        if not dominated:
            filtered.append((p, s, e))

    # 2) Greedy interval cover.
    tol = _tf_to_timedelta(timeframe, 1)
    current = want_start
    chosen: List[Tuple[Path, pd.Timestamp, pd.Timestamp]] = []
    while current <= want_end:
        best: Optional[Tuple[Path, pd.Timestamp, pd.Timestamp]] = None
        best_end: Optional[pd.Timestamp] = None
        for p, s, e in filtered:
            if s <= (current + tol) and e >= current:
                if best_end is None or e > best_end:
                    best = (p, s, e)
                    best_end = e
        if best is None or best_end is None:
            # Gap
            return None
        chosen.append(best)
        current = best_end + tol
        if best_end >= want_end:
            break

    # Load selected chunks only
    dfs = []
    chosen_paths = []
    for p, _, _ in chosen:
        logging.info(f"Loading data from cache chunk: {p}")
        df = pd.read_csv(p)
        if "open_time" not in df.columns and "ts" in df.columns:
            df["open_time"] = df["ts"]
        if "ts" in df.columns:
            df["ts"] = df["ts"].astype(int)
            df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        if "open_time" in df.columns:
            df["open_time"] = df["open_time"].astype(int)
        dfs.append(df)
        chosen_paths.append(p)

    out = pd.concat(dfs, ignore_index=True)
    if out.empty or "ts" not in out.columns:
        return None
    out = out.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    out["dt"] = pd.to_datetime(out["ts"], unit="ms", utc=True)
    if out["dt"].min() > want_start or out["dt"].max() < want_end:
        return None

    if bool(cache_validate):
        ok = _cache_range_is_complete(
            out,
            timeframe=str(timeframe),
            want_start=want_start,
            want_end=want_end,
            max_missing=int(cache_max_missing),
        )
        if not bool(ok):
            logging.warning(
                f"[cache] incomplete union range: deleting {len(chosen_paths)} file(s) (tf={timeframe} range={start_date}..{end_date} max_missing={cache_max_missing})"
            )
            for p in chosen_paths:
                try:
                    p.unlink(missing_ok=True)
                except Exception as e:
                    logging.warning(f"[cache] failed to delete {p}: {e}")
            return None
    return out


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
    stop_buffers_pct: List[float],
    top_n: int,
    out_csv: Optional[str],
    ensure_year_cache: bool,
    offline: bool,
    allow_warmup_before_start: bool,
    extreme_confirm_bars: int,
    entry_require_hist_abs_growth: bool,
    entry_cci_tf_confluence: bool,
    entry_cci_tf_max_combo_size: int,
    exit_b_mode: str,
    cache_validate: bool,
    cache_max_missing: int,
    progress_every: int = 200,
) -> pd.DataFrame:
    rows: List[Dict] = []
    start_dt = pd.Timestamp(start_date, tz="UTC")
    end_dt = pd.Timestamp(end_date, tz="UTC")
    if end_dt < start_dt:
        raise SystemExit("end-date must be >= start-date")

    def _n_combos(n: int, k: int) -> int:
        try:
            return int(math.comb(int(n), int(k)))
        except Exception:
            return 0

    for tf in timeframes:
        tf = str(tf).strip()
        if not tf:
            continue

        tf_t0 = time.perf_counter()
        logging.info(f"[grid-search] timeframe={tf} loading data...")

        tf_params = _indicator_params_for_tf(tf)
        warmup = max(200, int(tf_params["cci_period"]) * 3)
        if bool(allow_warmup_before_start):
            start_fetch = (start_dt - _tf_to_timedelta(tf, warmup + 10)).strftime("%Y-%m-%d")
        else:
            start_fetch = start_dt.strftime("%Y-%m-%d")
        end_fetch = end_dt.strftime("%Y-%m-%d")

        if bool(ensure_year_cache) and not bool(offline):
            _ = get_crypto_data(
                symbol,
                start_dt.strftime("%Y-%m-%d"),
                end_dt.strftime("%Y-%m-%d"),
                tf,
                project_root,
                cache_max_missing=int(cache_max_missing),
                cache_validate=bool(cache_validate),
            )

        df_recent = _load_cached_klines_covering_range(
            project_root=project_root,
            symbol=symbol,
            timeframe=tf,
            start_date=start_fetch,
            end_date=end_fetch,
            cache_validate=bool(cache_validate),
            cache_max_missing=int(cache_max_missing),
        )
        if df_recent is None:
            df_recent = _load_cached_klines_union_range(
                project_root=project_root,
                symbol=symbol,
                timeframe=tf,
                start_date=start_fetch,
                end_date=end_fetch,
                cache_validate=bool(cache_validate),
                cache_max_missing=int(cache_max_missing),
            )
        if df_recent is None:
            # Fallback: reuse year cache if present, and only fetch the missing warmup part
            df_main = _load_cached_klines_covering_range(
                project_root=project_root,
                symbol=symbol,
                timeframe=tf,
                start_date=start_dt.strftime("%Y-%m-%d"),
                end_date=end_fetch,
                cache_validate=bool(cache_validate),
                cache_max_missing=int(cache_max_missing),
            )
            if df_main is None:
                df_main = _load_cached_klines_union_range(
                    project_root=project_root,
                    symbol=symbol,
                    timeframe=tf,
                    start_date=start_dt.strftime("%Y-%m-%d"),
                    end_date=end_fetch,
                    cache_validate=bool(cache_validate),
                    cache_max_missing=int(cache_max_missing),
                )
            if df_main is None:
                if bool(offline):
                    logging.warning(f"[grid-search] offline mode: missing cached data for tf={tf} range={start_dt.strftime('%Y-%m-%d')}..{end_fetch}. Skipping timeframe.")
                    continue
                df_main = get_crypto_data(
                    symbol,
                    start_dt.strftime("%Y-%m-%d"),
                    end_fetch,
                    tf,
                    project_root,
                    cache_max_missing=int(cache_max_missing),
                    cache_validate=bool(cache_validate),
                )
            df_warm = pd.DataFrame()
            if pd.Timestamp(start_fetch, tz="UTC") < start_dt:
                df_warm = _load_cached_klines_covering_range(
                    project_root=project_root,
                    symbol=symbol,
                    timeframe=tf,
                    start_date=start_fetch,
                    end_date=start_dt.strftime("%Y-%m-%d"),
                    cache_validate=bool(cache_validate),
                    cache_max_missing=int(cache_max_missing),
                )
                if df_warm is None:
                    df_warm = _load_cached_klines_union_range(
                        project_root=project_root,
                        symbol=symbol,
                        timeframe=tf,
                        start_date=start_fetch,
                        end_date=start_dt.strftime("%Y-%m-%d"),
                        cache_validate=bool(cache_validate),
                        cache_max_missing=int(cache_max_missing),
                    )
                if df_warm is None:
                    if bool(offline):
                        logging.warning(f"[grid-search] offline mode: missing warmup cached data for tf={tf} range={start_fetch}..{start_dt.strftime('%Y-%m-%d')}. Proceeding without warmup.")
                        df_warm = pd.DataFrame()
                    else:
                        df_warm = get_crypto_data(
                            symbol,
                            start_fetch,
                            start_dt.strftime("%Y-%m-%d"),
                            tf,
                            project_root,
                            cache_max_missing=int(cache_max_missing),
                            cache_validate=bool(cache_validate),
                        )
            df_recent = _merge_history(df_warm, df_main)

        df_local = pd.DataFrame()
        if str(symbol).upper() == "LINKUSDT":
            df_local = _load_binance_data_vision_klines(project_root, "LINKUSDT", tf)
        df = _merge_history(df_local, df_recent)
        if df.empty:
            continue
        if "dt" not in df.columns:
            df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df["dt"] = pd.to_datetime(df["dt"], utc=True, errors="coerce")
        if bool(allow_warmup_before_start):
            df = df[(df["dt"] >= (start_dt - _tf_to_timedelta(tf, warmup + 10))) & (df["dt"] <= end_dt)].reset_index(drop=True)
        else:
            df = df[(df["dt"] >= start_dt) & (df["dt"] <= end_dt)].reset_index(drop=True)
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
        df["macd_signal"] = macd_signal
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
            "macd_signal": "macd_signal",
            "macd_hist": "macd_hist",
            "dmi_dx": "dx",
            "klinger_kvo": "kvo",
            "klinger_signal": "klinger_signal",
            "mfi": "mfi",
            "stoch_k": "stoch_k",
            "stoch_d": "stoch_d",
        }

        start_i = 0

        max_k = int(max_combo_size)
        if max_k <= 0:
            max_k = int(len(series_universe))
        max_k = min(int(max_k), int(len(series_universe)))

        sb_list = list(stop_buffers_pct or [])
        if not sb_list:
            sb_list = [0.01]

        base_minutes = _tf_to_minutes(str(tf))
        tf_sup = [t for t in list(timeframes) if _tf_to_minutes(str(t)) >= int(base_minutes) and str(t) != str(tf)]
        max_total = int(entry_cci_tf_max_combo_size or 1)
        if max_total < 1:
            max_total = 1
        max_add = max(0, int(max_total) - 1)
        max_add = min(int(max_add), int(len(tf_sup)))

        tf_combos: List[Tuple[str, ...]] = [tuple()]
        if bool(entry_cci_tf_confluence) and int(max_add) > 0:
            tf_combos = []
            for k_tf in range(0, int(max_add) + 1):
                for combo_tf in itertools.combinations(tf_sup, k_tf):
                    tf_combos.append(tuple(combo_tf))

        n_series = int(len(series_universe))
        n_series_combos = 0
        for k in range(1, int(max_k) + 1):
            n_series_combos += _n_combos(n_series, int(k))
        total_tests_tf = int(len(sb_list)) * int(n_series_combos) * int(len(tf_combos))
        logging.info(
            f"[grid-search] timeframe={tf} tests={total_tests_tf} (series_combos={n_series_combos} stop_buf={len(sb_list)} tf_combos={len(tf_combos)})"
        )

        test_i = 0
        last_log_t = time.perf_counter()

        for stop_buf in sb_list:
            for k in range(1, max_k + 1):
                for combo in itertools.combinations(series_universe, k):
                    combo_list = list(combo)

                    for combo_tf in tf_combos:
                        test_i += 1
                        if int(progress_every) > 0 and (test_i == 1 or (test_i % int(progress_every) == 0)):
                            now = time.perf_counter()
                            elapsed = max(1e-9, now - tf_t0)
                            rate = float(test_i) / float(elapsed)
                            eta_s = (float(total_tests_tf - test_i) / rate) if rate > 0 else math.inf
                            logging.info(
                                f"[grid-search] timeframe={tf} progress {test_i}/{total_tests_tf} ({100.0*test_i/max(1,total_tests_tf):.1f}%) rate={rate:.2f} tests/s eta={eta_s/60.0:.1f} min"
                            )
                            last_log_t = now

                        entry_cci_tf_cols = []
                        if bool(entry_cci_tf_confluence) and combo_tf:
                            for t2 in list(combo_tf):
                                entry_cci_tf_cols.append(
                                    _ensure_cci_tf_column(df, base_tf=str(tf), target_tf=str(t2))
                                )

                        trades = simulate_trades_from_stream_lib(
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
                            stop_buffer_pct=float(stop_buf),
                            stop_ref_series="price",
                            start_i=int(start_i),
                            extreme_confirm_bars=int(extreme_confirm_bars),
                            entry_require_hist_abs_growth=bool(entry_require_hist_abs_growth),
                            entry_cci_tf_cols=list(entry_cci_tf_cols),
                            exit_b_mode=str(exit_b_mode),
                        )

                        sc = score_trades(df, trades)
                        equity = float(sc.equity)
                        max_dd = float(sc.max_dd)
                        eq_dd_ratio = float(sc.eq_dd_ratio)
                        eq_mae_ratio = float(sc.eq_mae_ratio)
                        ratio_final = float(sc.ratio_final)
                        worst_mae = float(sc.worst_mae)
                        n_trades = int(sc.n_trades)
                        winrate = float(sc.winrate)

                        rows.append(
                            {
                                "symbol": str(symbol),
                                "timeframe": str(tf),
                                "cci_tf_combo": ",".join([str(tf)] + [str(x) for x in list(combo_tf)]) if bool(entry_cci_tf_confluence) else "",
                                "stop_buffer_pct": float(stop_buf),
                                "combo": ",".join(combo_list),
                                "k": int(len(combo_list)),
                                "n_trades": int(n_trades),
                                "equity": float(equity),
                                "max_dd": float(max_dd),
                                "eq_dd_ratio": float(eq_dd_ratio),
                                "eq_mae_ratio": float(eq_mae_ratio),
                                "ratio_final": float(ratio_final),
                                "worst_mae": float(worst_mae),
                                "winrate": float(winrate),
                            }
                        )

        logging.info(f"[grid-search] timeframe={tf} done in {(time.perf_counter()-tf_t0)/60.0:.1f} min")

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["timeframe", "ratio_final", "equity"], ascending=[True, False, False]).reset_index(drop=True)
    if out_csv:
        Path(out_csv).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
    if int(top_n) > 0:
        def _print_block(title: str, df_block: pd.DataFrame, *, show_tf: bool) -> None:
            print(f"\n{title}")
            for _, r in df_block.iterrows():
                tf_part = f"tf={r['timeframe']} " if bool(show_tf) else ""
                stop_part = ""
                if "stop_buffer_pct" in r:
                    try:
                        stop_part = f"stop={100.0*float(r['stop_buffer_pct']):.1f}% "
                    except Exception:
                        stop_part = ""
                print(
                    f"{tf_part}{stop_part}combo={r['combo']} min_conf={int(r['k'])} n={int(r['n_trades'])} "
                    f"equity={float(r['equity']):+.3f}% dd={float(r['max_dd']):.3f}% ratio={float(r['ratio_final']):.3f} "
                    f"worst_mae={float(r['worst_mae']):+.3f}% winrate={float(r['winrate']):.1f}%"
                )

        print("\n=== Grid search results (TOP by ratio) ===")

        for tf in sorted(out["timeframe"].astype(str).unique().tolist()):
            df_tf = out[out["timeframe"].astype(str) == str(tf)].copy()
            df_tf = df_tf.sort_values(["ratio_final", "equity"], ascending=[False, False])
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

        df_global = out.sort_values(["ratio_final", "equity"], ascending=[False, False])
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
    parser.add_argument("--stop-buffer-pct", type=float, default=0.0, help="Buffer stop (fraction) (ex: 0.01 = 1%%)")
    parser.add_argument("--extreme-confirm-bars", type=int, default=0, help="Confirme un extrême seulement après N bougies clôturées après l’extrême (0 = désactivé)")
    parser.add_argument("--entry-no-hist-abs-growth", action="store_true", help="Désactive la contrainte de croissance de |macd_hist| sur l’entrée")
    parser.add_argument("--entry-cci-tf-confluence", action="store_true", help="Filtre optionnel: exige que d’autres TF (>= TF de simulation) soient aussi en zone extrême CCI lors de la création du signal d’entrée")
    parser.add_argument("--entry-cci-tf-max-combo-size", type=int, default=1, help="Taille max des combinaisons TF pour le filtre CCI (inclut le TF de simulation). 1 = désactivé")
    parser.add_argument("--entry-cci-tf-confluence-tfs", default="", help="(hors grid) Liste de TF supplémentaires séparés par virgule (>= TF courant) à exiger en zone CCI extrême")
    parser.add_argument("--exit-b-mode", choices=["macd", "stoch", "klinger", "none"], default="macd", help="Mode de clôture B: cross contre-mouvement après extrême CCI")
    parser.add_argument("--grid-search", action="store_true", help="Teste toutes les combinaisons de séries et timeframes et classe par ratio_final")
    parser.add_argument("--grid-timeframes", default="5m,15m,1h,2h,6h,1d", help="Liste timeframes séparées par virgule")
    parser.add_argument("--grid-year", type=int, help="Raccourci: backtest sur toute l’année YYYY (ex: 2025)")
    parser.add_argument("--grid-ensure-year-cache", action="store_true", help="Construit le cache CSV {symbol}_{tf}_{YYYY-01-01}_{YYYY-12-31}.csv si absent")
    parser.add_argument("--grid-offline", action="store_true", help="N’utilise que les CSV en cache (aucun appel Bybit). Si données manquantes: skip TF.")
    parser.add_argument("--grid-allow-warmup-before-start", action="store_true", help="Autorise le warmup avant start-date (sinon: période strictement start..end)")
    parser.add_argument(
        "--grid-stop-buffers",
        default="1",
        help="Liste buffers stop en %% (ex: 2,5,10). Interprété comme %% si >1 sinon fraction (0.02)",
    )
    parser.add_argument("--grid-series", help="Sous-ensemble de séries pour le grid-search (sinon toutes)")
    parser.add_argument("--grid-max-combo-size", type=int, default=0, help="Taille max des combinaisons (0 = toutes)")
    parser.add_argument("--grid-top", type=int, default=30, help="Nombre de résultats à afficher")
    parser.add_argument("--grid-out-csv", help="Chemin CSV de sortie")
    parser.add_argument("--grid-progress-every", type=int, default=200, help="(grid-search) Log progression toutes les N simulations")
    parser.add_argument("--cache-max-missing", type=int, default=5, help="Tolérance: nb max de bougies manquantes dans un CSV cache avant suppression + refetch")
    parser.add_argument("--cache-no-validate", action="store_true", help="Désactive la validation complétude des CSV cache")
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
            "macd_signal": "macd_signal",
            "macd_hist": "macd_hist",
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
            stop_buffers_pct=_parse_pct_list(args.grid_stop_buffers),
            top_n=int(args.grid_top),
            out_csv=str(args.grid_out_csv) if args.grid_out_csv else None,
            ensure_year_cache=bool(args.grid_ensure_year_cache),
            offline=bool(args.grid_offline),
            allow_warmup_before_start=bool(args.grid_allow_warmup_before_start),
            extreme_confirm_bars=int(args.extreme_confirm_bars),
            entry_require_hist_abs_growth=not bool(args.entry_no_hist_abs_growth),
            entry_cci_tf_confluence=bool(args.entry_cci_tf_confluence),
            entry_cci_tf_max_combo_size=int(args.entry_cci_tf_max_combo_size),
            exit_b_mode=str(args.exit_b_mode),
            cache_validate=not bool(args.cache_no_validate),
            cache_max_missing=int(args.cache_max_missing),
            progress_every=int(args.grid_progress_every),
        )
        return

    tf_params = _indicator_params_for_tf(args.timeframe)
    warmup = max(200, int(tf_params["cci_period"]) * 3)
    logger.info(
        f"Indicator params: cci_period={tf_params['cci_period']} stoch=({tf_params['stoch_k_period']},{tf_params['stoch_k_smooth']},{tf_params['stoch_d_period']})"
    )

    # Charger les données récentes (Bybit) + historique local pour LINK si dispo
    if args.start_date and args.end_date:
        df_recent = get_crypto_data(
            args.symbol,
            args.start_date,
            args.end_date,
            args.timeframe,
            project_root,
            cache_max_missing=int(args.cache_max_missing),
            cache_validate=not bool(args.cache_no_validate),
        )
    else:
        end_dt = pd.Timestamp.now(tz="UTC")
        start_dt = end_dt - _tf_to_timedelta(args.timeframe, args.limit + warmup + 10)
        df_recent = get_crypto_data(
            args.symbol,
            start_dt.strftime("%Y-%m-%d"),
            end_dt.strftime("%Y-%m-%d"),
            args.timeframe,
            project_root,
            cache_max_missing=int(args.cache_max_missing),
            cache_validate=not bool(args.cache_no_validate),
        )

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
    df["macd_signal"] = macd_signal
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
        "macd_signal": "macd_signal",
        "macd_hist": "macd_hist",
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

        entry_cci_tf_cols: List[str] = []
        if bool(args.entry_cci_tf_confluence):
            raw = str(args.entry_cci_tf_confluence_tfs or "").strip()
            extra_tfs = [x.strip() for x in raw.split(",") if x.strip()]
            if extra_tfs:
                for t2 in extra_tfs:
                    entry_cci_tf_cols.append(_ensure_cci_tf_column(df, base_tf=str(args.timeframe), target_tf=str(t2)))

        trades = simulate_trades_from_stream_lib(
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
            extreme_confirm_bars=int(args.extreme_confirm_bars),
            entry_require_hist_abs_growth=not bool(args.entry_no_hist_abs_growth),
            entry_cci_tf_cols=list(entry_cci_tf_cols),
            exit_b_mode=str(args.exit_b_mode),
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
