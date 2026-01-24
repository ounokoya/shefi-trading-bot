#!/usr/bin/env python3
"""
Analyse des performances des trades générés.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


def analyze_trades():
    """Analyse les trades et affiche les statistiques."""
    
    # Charger le fichier des trades
    trades_file = Path("data/processed/blocks/LINKUSDT_4h_2020-01-01_2025-12-31_trades.csv")
    
    if not trades_file.exists():
        print(f"❌ Fichier introuvable: {trades_file}")
        return
    
    df = pd.read_csv(trades_file)
    
    print(f"📊 ANALYSE DES TRADES")
    print(f"{'='*60}")
    print(f"📁 Fichier: {trades_file}")
    print(f"📈 Nombre total de trades: {len(df)}")
    print()
    
    # Analyser la colonne exit_pct (pourcentage de sortie)
    if 'exit_pct' in df.columns:
        exit_pct = pd.to_numeric(df['exit_pct'], errors='coerce')
        
        # Filtrer les valeurs valides
        valid_trades = exit_pct.dropna()
        print(f"✅ Trades valides: {len(valid_trades)}")
        
        # Séparer LONG et SHORT
        long_trades = valid_trades[df['side'] == 'LONG']
        short_trades = valid_trades[df['side'] == 'SHORT']
        
        print(f"📈 Trades LONG: {len(long_trades)}")
        print(f"📉 Trades SHORT: {len(short_trades)}")
        print()
        
        # Analyse des performances globales
        print(f"🎯 PERFORMANCE GLOBALE")
        print(f"{'='*40}")
        
        # Stats globales
        mean_return = valid_trades.mean()
        std_return = valid_trades.std()
        median_return = valid_trades.median()
        
        print(f"📊 Return moyen: {mean_return:.4f} ({mean_return*100:.2f}%)")
        print(f"📊 Écart-type: {std_return:.4f} ({std_return*100:.2f}%)")
        print(f"📊 Médiane: {median_return:.4f} ({median_return*100:.2f}%)")
        print()
        
        # Trades au-dessus des seuils
        thresholds = [0.01, 0.02, 0.05]  # 1%, 2%, 5%
        
        print(f"🎯 TRADES AU-DESSUS DES SEUILS")
        print(f"{'='*40}")
        
        for threshold in thresholds:
            above_threshold = (valid_trades.abs() >= threshold).sum()
            percentage = (above_threshold / len(valid_trades)) * 100
            
            # Séparer positifs et négatifs
            above_pos = (valid_trades >= threshold).sum()
            above_neg = (valid_trades <= -threshold).sum()
            
            print(f"📈 |{threshold*100:.0f}%|: {above_threshold} trades ({percentage:.1f}%)")
            print(f"   ├─ Positifs: {above_pos} trades")
            print(f"   └─ Négatifs: {above_neg} trades")
            print()
        
        # Analyse par direction
        print(f"📊 PERFORMANCE PAR DIRECTION")
        print(f"{'='*40}")
        
        for side, trades in [('LONG', long_trades), ('SHORT', short_trades)]:
            if len(trades) > 0:
                mean_side = trades.mean()
                above_1pct = (trades >= 0.01).sum()
                above_2pct = (trades >= 0.02).sum()
                above_5pct = (trades >= 0.05).sum()
                
                print(f"📈 {side}:")
                print(f"   ├─ Moyenne: {mean_side:.4f} ({mean_side*100:.2f}%)")
                print(f"   ├─ >1%: {above_1pct} trades ({above_1pct/len(trades)*100:.1f}%)")
                print(f"   ├─ >2%: {above_2pct} trades ({above_2pct/len(trades)*100:.1f}%)")
                print(f"   └─ >5%: {above_5pct} trades ({above_5pct/len(trades)*100:.1f}%)")
                print()
        
        # Distribution des returns
        print(f"📊 DISTRIBUTION DES RETURNS")
        print(f"{'='*40}")
        
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            value = valid_trades.quantile(p/100)
            print(f"Percentile {p:2d}%: {value:.4f} ({value*100:.2f}%)")
        print()
        
        # Meilleurs et pires trades
        print(f"🏆 MEILLEURS ET PIRES TRADES")
        print(f"{'='*40}")
        
        best_trades = valid_trades.nlargest(5)
        worst_trades = valid_trades.nsmallest(5)
        
        print(f"📈 Top 5 meilleurs trades:")
        for i, (idx, value) in enumerate(best_trades.items(), 1):
            side = df.loc[idx, 'side']
            print(f"   {i}. {side}: {value:.4f} ({value*100:.2f}%)")
        
        print(f"\n📉 Top 5 pires trades:")
        for i, (idx, value) in enumerate(worst_trades.items(), 1):
            side = df.loc[idx, 'side']
            print(f"   {i}. {side}: {value:.4f} ({value*100:.2f}%)")
        
    else:
        print("❌ Colonne 'exit_pct' introuvable")


if __name__ == "__main__":
    analyze_trades()
