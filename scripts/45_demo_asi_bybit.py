#!/usr/bin/env python3
"""
Demo ASI (Accumulative Swing Index) - Bybit Data

Ce script démontre l'utilisation de l'indicateur ASI avec des données réelles
de Bybit et compare les résultats avec la précision TradingView.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any

from libs.indicators.asi import asi, asi_signals, asi_by_market, MARKET_CONFIGS


def _interval_to_bybit(interval: str) -> str:
    s = str(interval).strip()
    if not s:
        raise ValueError("interval cannot be empty")

    if s.isdigit():
        return s

    s_lower = s.lower()
    if s_lower in {"d", "1d"}:
        return "D"
    if s_lower in {"w", "1w"}:
        return "W"
    if s == "M" or s_lower in {"1mo", "mo", "1month"}:
        return "M"

    import re

    m = re.fullmatch(r"(\d+)([mhd])", s_lower)
    if not m:
        raise ValueError(f"unsupported interval format: {interval}")

    n = int(m.group(1))
    unit = m.group(2)
    if unit == "m":
        minutes = n
    elif unit == "h":
        minutes = n * 60
    elif unit == "d":
        if n != 1:
            raise ValueError(f"Bybit only supports daily interval as 1d/D, got: {interval}")
        return "D"
    else:
        raise ValueError(f"unsupported interval unit: {unit}")

    allowed_minutes = {1, 3, 5, 15, 30, 60, 120, 240, 360, 720}
    if minutes not in allowed_minutes:
        allowed_str = ", ".join(str(x) for x in sorted(allowed_minutes))
        raise ValueError(f"unsupported minute interval for Bybit: {minutes} (from {interval}). Allowed: {allowed_str}")
    return str(minutes)


def _interval_to_timedelta(interval: str) -> pd.Timedelta:
    s = str(interval).strip().lower()
    if not s:
        raise ValueError("interval cannot be empty")

    if s in {"d", "1d"}:
        return pd.Timedelta(days=1)
    if s in {"w", "1w"}:
        return pd.Timedelta(days=7)

    if s.isdigit():
        return pd.Timedelta(minutes=int(s))

    import re

    m = re.fullmatch(r"(\d+)([mhd])", s)
    if not m:
        raise ValueError(f"unsupported interval format: {interval}")

    n = int(m.group(1))
    unit = m.group(2)
    if unit == "m":
        return pd.Timedelta(minutes=n)
    if unit == "h":
        return pd.Timedelta(hours=n)
    if unit == "d":
        return pd.Timedelta(days=n)
    raise ValueError(f"unsupported interval unit: {unit}")


def _fetch_bybit_klines_last_n(
    *,
    symbol: str,
    interval: str,
    limit: int,
    category: str,
    base_url: str,
    timeout_s: float,
) -> pd.DataFrame:
    import requests

    url = f"{base_url.rstrip('/')}/v5/market/kline"

    page_limit = int(limit)
    if page_limit <= 0:
        page_limit = 200
    if page_limit > 1000:
        page_limit = 1000

    params: dict[str, Any] = {
        "category": str(category),
        "symbol": str(symbol),
        "interval": str(interval),
        "limit": str(int(page_limit)),
    }

    r = requests.get(url, params=params, timeout=float(timeout_s))
    r.raise_for_status()
    payload = r.json()
    if str(payload.get("retCode")) != "0":
        raise RuntimeError(f"Bybit retCode={payload.get('retCode')} retMsg={payload.get('retMsg')}")

    result = payload.get("result") or {}
    rows = result.get("list") or []
    out_rows: list[dict[str, object]] = []

    for row in rows:
        if not isinstance(row, list) or len(row) < 6:
            continue
        try:
            ts = int(row[0])
            open_price = float(row[1])
            high_price = float(row[2])
            low_price = float(row[3])
            close_price = float(row[4])
            volume = float(row[5])
            turnover = float(row[6]) if len(row) > 6 else float("nan")
        except Exception:
            continue

        out_rows.append({
            "ts": ts,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "turnover": turnover,
        })

    df = pd.DataFrame(out_rows)
    if len(df):
        df = df.sort_values("ts").reset_index(drop=True)
        df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    return df


def demo_asi_bybit(symbol="BTCUSDT", interval="15m", limit=500, category="linear"):
    """
    Démonstration ASI avec données Bybit
    """
    print(f"🔥 Demo ASI - {symbol} {interval} (category: {category})")
    print(f"📊 Période: {limit} bougies")
    print("=" * 60)
    
    # Récupération des données
    print("📡 Récupération données Bybit...")
    df = _fetch_bybit_klines_last_n(
        symbol=symbol,
        interval=_interval_to_bybit(interval),
        limit=limit,
        category=category,
        base_url="https://api.bybit.com",
        timeout_s=30.0
    )
    
    if df is None or df.empty:
        print("❌ Erreur récupération données")
        return
    
    print(f"✅ Données récupérées: {len(df)} bougies")
    print(f"📅 Période: {df['dt'].iloc[0]} -> {df['dt'].iloc[-1]}")
    
    # Conversion en DataFrame avec index timestamp pour compatibilité
    df_indexed = df.copy()
    df_indexed['timestamp'] = pd.to_datetime(df_indexed['ts'], unit='ms', utc=True)
    df_indexed.set_index('timestamp', inplace=True)
    
    # Test avec différentes configurations
    configs_to_test = [
        ('crypto', MARKET_CONFIGS['crypto']),
        ('stocks', MARKET_CONFIGS['stocks']), 
        ('forex', MARKET_CONFIGS['forex'])
    ]
    
    results = {}
    
    for market_name, config in configs_to_test:
        print(f"\n🧮 Test configuration {market_name} (T={config['limit_move_pct']*100:.0f}%):")
        
        # Calcul ASI
        result = asi_by_market(
            df=df_indexed,
            market=market_name,
            high_col='high',
            low_col='low', 
            close_col='close',
            open_col='open'
        )
        
        results[market_name] = result
        
        # Statistiques
        asi_values = result['ASI'].dropna()
        si_values = result['SI'].dropna()
        
        print(f"  ASI - Min: {asi_values.min():.2f}, Max: {asi_values.max():.2f}, Mean: {asi_values.mean():.2f}")
        print(f"  SI  - Min: {si_values.min():.2f}, Max: {si_values.max():.2f}, Mean: {si_values.mean():.2f}")
        print(f"  Signaux buy: {result['buy_signal'].sum()}, sell: {result['sell_signal'].sum()}")
    
    # Analyse comparative
    print(f"\n📈 Analyse comparative:")
    print("-" * 40)
    
    comparison_data = []
    for market_name, result in results.items():
        asi_values = result['ASI'].dropna()
        comparison_data.append({
            'Market': market_name,
            'ASI_Max': asi_values.max(),
            'ASI_Min': asi_values.min(), 
            'ASI_Range': asi_values.max() - asi_values.min(),
            'ASI_Std': asi_values.std(),
            'Buy_Signals': result['buy_signal'].sum(),
            'Sell_Signals': result['sell_signal'].sum()
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.round(4))
    
    # Visualisation
    try:
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Graphique prix et ASI
        ax1 = axes[0]
        ax1.plot(df_indexed.index, df_indexed['close'], label='Price', color='black', alpha=0.7)
        ax1.set_ylabel('Price')
        ax1.set_title(f'{symbol} - Price & ASI Comparison ({category})')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # ASI pour chaque configuration
        ax2 = axes[1]
        colors = ['blue', 'green', 'red']
        for i, (market_name, result) in enumerate(results.items()):
            ax2.plot(result.index, result['ASI'], 
                    label=f'ASI {market_name}', 
                    color=colors[i], 
                    alpha=0.8)
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_ylabel('ASI')
        ax2.set_xlabel('Time')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'data/outputs/asi_demo_{symbol}_{interval}_{category}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', 
                   dpi=150, bbox_inches='tight')
        print(f"\n📊 Graphique sauvegardé dans data/outputs/")
        
    except Exception as e:
        print(f"⚠️ Erreur visualisation: {e}")
    
    return results


def demo_asi_precision_test():
    """
    Test de précision avec données connues
    """
    print("\n🧪 Test de précision ASI")
    print("=" * 40)
    
    # Données de test simples
    test_data = pd.DataFrame({
        'open': [100.0, 102.0, 107.0, 104.0, 111.0, 109.0, 112.0],
        'high': [105.0, 110.0, 108.0, 112.0, 115.0, 113.0, 116.0], 
        'low': [95.0, 100.0, 98.0, 102.0, 105.0, 103.0, 108.0],
        'close': [100.0, 108.0, 105.0, 110.0, 113.0, 108.0, 114.0]
    })
    
    print("📊 Données de test:")
    print(test_data)
    
    # Test avec T fixe pour comparaison
    result_fixed = asi(
        high=test_data['high'],
        low=test_data['low'],
        close=test_data['close'],
        open_=test_data['open'],
        limit_move_value=10.0,  # T fixe
        offset=0
    )
    
    print(f"\n📈 Résultats avec T=10.0:")
    print(result_fixed[['ASI', 'SI']].round(4))
    
    # Test avec T automatique (crypto 10%)
    result_auto = asi_by_market(test_data, market='crypto')
    
    print(f"\n📈 Résultats avec T auto (crypto 10%):")
    print(result_auto[['ASI', 'SI']].round(4))
    
    return result_fixed, result_auto


def main() -> int:
    """
    Fonction principale simple - calcul ASI sur LINKUSDT perpétuel 2025 timeframe 1d
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="LINKUSDT")  # LINKUSDT comme demandé
    ap.add_argument("--category", default="linear")  # perpétuels uniquement
    ap.add_argument("--base-url", default="https://api.bybit.com")
    ap.add_argument("--interval", default="1D")  # timeframe 1D pour 2025
    ap.add_argument("--tail", type=int, default=20)  # n dernières valeurs par défaut
    ap.add_argument("--limit-move-value", type=float, default=1.0)
    ap.add_argument("--time-label", choices=["open", "close"], default="open")
    args = ap.parse_args()                                   

    # Récupération des données (maximum 1000 bougies = ~3 ans en daily)
    df = _fetch_bybit_klines_last_n(
        symbol=str(args.symbol),
        interval=_interval_to_bybit(str(args.interval)),
        limit=1000,  # Maximum Bybit pour couvrir 2025
        category=str(args.category),
        base_url=str(args.base_url),
        timeout_s=30.0,
    )

    if not len(df):
        raise RuntimeError("no klines fetched")

    print(f"Fetched {len(df)} klines for {args.symbol} {args.interval}")

    # Conversion en DataFrame avec index timestamp pour compatibilité
    df_indexed = df.copy()
    df_indexed['timestamp'] = pd.to_datetime(df_indexed['ts'], unit='ms', utc=True)
    df_indexed.set_index('timestamp', inplace=True)

    # Filtrer sur l'année 2025 uniquement
    start_2025 = pd.Timestamp('2025-01-01', tz='UTC')
    end_2025 = pd.Timestamp('2025-12-31', tz='UTC')
    df_2025 = df_indexed[(df_indexed.index >= start_2025) & (df_indexed.index <= end_2025)]
    
    if len(df_2025) == 0:
        print("No data for 2025")
        return 1
        
    print(f"Data for 2025: {len(df_2025)} days")

    result = asi(
        high=df_2025['high'],
        low=df_2025['low'],
        close=df_2025['close'],
        open_=df_2025['open'],
        limit_move_value=float(args.limit_move_value),
        offset=0,
    )

    # Afficher les n dernières valeurs ASI avec leurs dates
    asi_data = result[['ASI']].dropna()
    label_delta = pd.Timedelta(0)
    if str(args.time_label) == "close":
        label_delta = _interval_to_timedelta(str(args.interval))

    tail_n = int(args.tail)
    if len(asi_data) >= tail_n:
        last_n = asi_data.tail(tail_n)
        print(f"\n=== ASI VALUES (last {tail_n}) ===")
        for idx, row in last_n.iterrows():
            ts = idx + label_delta
            print(f"{ts.strftime('%Y-%m-%d %H:%M:%S UTC')} | ASI={row['ASI']:.4f}")
        
        # Afficher le plus petit et le plus grand ASI parmi ces n valeurs
        min_asi = last_n['ASI'].min()
        max_asi = last_n['ASI'].max()
        min_date = last_n['ASI'].idxmin()
        max_date = last_n['ASI'].idxmax()

        print(f"\n=== MIN/MAX ASI (last {tail_n}) ===")
        min_ts = min_date + label_delta
        max_ts = max_date + label_delta
        print(f"MIN: {min_ts.strftime('%Y-%m-%d %H:%M:%S UTC')} | ASI={min_asi:.4f}")
        print(f"MAX: {max_ts.strftime('%Y-%m-%d %H:%M:%S UTC')} | ASI={max_asi:.4f}")
    else:
        print(f"\n=== ASI VALUES (available: {len(asi_data)}) ===")
        for idx, row in asi_data.iterrows():
            ts = idx + label_delta
            print(f"{ts.strftime('%Y-%m-%d %H:%M:%S UTC')} | ASI={row['ASI']:.4f}")
        
    return 0


if __name__ == "__main__":
    # Création dossier outputs si nécessaire
    os.makedirs("data/outputs", exist_ok=True)
    exit(main())
