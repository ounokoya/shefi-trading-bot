#!/usr/bin/env python3
"""
Analyse détaillée des trades en distinguant :
1. Trades extrême → extrême (t0 → t1)
2. Trades croisement → extrême (cross → t1)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


def analyze_trade_types():
    """Analyse les deux types de trades séparément."""
    
    # Charger le fichier des trades
    trades_file = Path("data/processed/blocks/LINKUSDT_4h_2020-01-01_2025-12-31_trades.csv")
    
    if not trades_file.exists():
        print(f"❌ Fichier introuvable: {trades_file}")
        return
    
    df = pd.read_csv(trades_file)
    
    print(f"📊 ANALYSE DÉTAILLÉE DES TYPES DE TRADES")
    print(f"{'='*70}")
    print(f"📁 Fichier: {trades_file}")
    print(f"📈 Nombre total de trades: {len(df)}")
    print()
    
    # Vérifier les colonnes nécessaires
    required_cols = ['exit_pct', 'cross_exit_pct_close', 'side']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Colonnes manquantes: {missing_cols}")
        return
    
    # Convertir en numérique
    df['exit_pct'] = pd.to_numeric(df['exit_pct'], errors='coerce')
    df['cross_exit_pct_close'] = pd.to_numeric(df['cross_exit_pct_close'], errors='coerce')
    
    # Filtrer les trades valides
    valid_extreme = df['exit_pct'].dropna()
    valid_cross = df['cross_exit_pct_close'].dropna()
    
    print(f"✅ Trades extrême→extrême valides: {len(valid_extreme)}")
    print(f"✅ Trades croisement→extrême valides: {len(valid_cross)}")
    print()
    
    # Séparer par direction
    long_df = df[df['side'] == 'LONG']
    short_df = df[df['side'] == 'SHORT']
    
    # ========================================
    # ANALYSE TRADES EXTREME → EXTREME
    # ========================================
    print(f"🎯 TYPE 1: TRADES EXTREME → EXTREME (t0 → t1)")
    print(f"{'='*50}")
    
    extreme_pct = df['exit_pct'].dropna()
    
    # Stats globales
    print(f"📊 Performance globale:")
    print(f"   ├─ Moyenne: {extreme_pct.mean():.4f} ({extreme_pct.mean()*100:.2f}%)")
    print(f"   ├─ Médiane: {extreme_pct.median():.4f} ({extreme_pct.median()*100:.2f}%)")
    print(f"   └─ Écart-type: {extreme_pct.std():.4f} ({extreme_pct.std()*100:.2f}%)")
    print()
    
    # Trades au-dessus des seuils
    thresholds = [0.01, 0.02, 0.05]
    print(f"📈 Trades > seuils (absolu):")
    for threshold in thresholds:
        above = (extreme_pct.abs() >= threshold).sum()
        pct = (above / len(extreme_pct)) * 100
        pos = (extreme_pct >= threshold).sum()
        neg = (extreme_pct <= -threshold).sum()
        print(f"   ├─ >{threshold*100:.0f}%: {above} trades ({pct:.1f}%)")
        print(f"   │  ├─ Positifs: {pos}")
        print(f"   │  └─ Négatifs: {neg}")
    
    print()
    
    # Par direction
    for side, side_df in [('LONG', long_df), ('SHORT', short_df)]:
        side_pct = side_df['exit_pct'].dropna()
        if len(side_pct) > 0:
            print(f"📊 {side}:")
            print(f"   ├─ Moyenne: {side_pct.mean():.4f} ({side_pct.mean()*100:.2f}%)")
            print(f"   ├─ >1%: {(side_pct.abs() >= 0.01).sum()} ({(side_pct.abs() >= 0.01).sum()/len(side_pct)*100:.1f}%)")
            print(f"   ├─ >2%: {(side_pct.abs() >= 0.02).sum()} ({(side_pct.abs() >= 0.02).sum()/len(side_pct)*100:.1f}%)")
            print(f"   └─ >5%: {(side_pct.abs() >= 0.05).sum()} ({(side_pct.abs() >= 0.05).sum()/len(side_pct)*100:.1f}%)")
    print()
    
    # ========================================
    # ANALYSE TRADES CROISEMENT → EXTREME
    # ========================================
    print(f"🎯 TYPE 2: TRADES CROISEMENT → EXTREME (cross → t1)")
    print(f"{'='*50}")
    
    cross_pct = df['cross_exit_pct_close'].dropna()
    
    # Stats globales
    print(f"📊 Performance globale:")
    print(f"   ├─ Moyenne: {cross_pct.mean():.4f} ({cross_pct.mean()*100:.2f}%)")
    print(f"   ├─ Médiane: {cross_pct.median():.4f} ({cross_pct.median()*100:.2f}%)")
    print(f"   └─ Écart-type: {cross_pct.std():.4f} ({cross_pct.std()*100:.2f}%)")
    print()
    
    # Trades au-dessus des seuils
    print(f"📈 Trades > seuils (absolu):")
    for threshold in thresholds:
        above = (cross_pct.abs() >= threshold).sum()
        pct = (above / len(cross_pct)) * 100
        pos = (cross_pct >= threshold).sum()
        neg = (cross_pct <= -threshold).sum()
        print(f"   ├─ >{threshold*100:.0f}%: {above} trades ({pct:.1f}%)")
        print(f"   │  ├─ Positifs: {pos}")
        print(f"   │  └─ Négatifs: {neg}")
    
    print()
    
    # Par direction
    for side, side_df in [('LONG', long_df), ('SHORT', short_df)]:
        side_pct = side_df['cross_exit_pct_close'].dropna()
        if len(side_pct) > 0:
            print(f"📊 {side}:")
            print(f"   ├─ Moyenne: {side_pct.mean():.4f} ({side_pct.mean()*100:.2f}%)")
            print(f"   ├─ >1%: {(side_pct.abs() >= 0.01).sum()} ({(side_pct.abs() >= 0.01).sum()/len(side_pct)*100:.1f}%)")
            print(f"   ├─ >2%: {(side_pct.abs() >= 0.02).sum()} ({(side_pct.abs() >= 0.02).sum()/len(side_pct)*100:.1f}%)")
            print(f"   └─ >5%: {(side_pct.abs() >= 0.05).sum()} ({(side_pct.abs() >= 0.05).sum()/len(side_pct)*100:.1f}%)")
    print()
    
    # ========================================
    # COMPARAISON DES DEUX TYPES
    # ========================================
    print(f"🔄 COMPARAISON DES DEUX TYPES")
    print(f"{'='*50}")
    
    comparison_data = []
    
    for metric_name, metric_func in [('Moyenne', np.mean), ('Médiane', np.median), ('Écart-type', np.std)]:
        extreme_val = metric_func(extreme_pct)
        cross_val = metric_func(cross_pct)
        
        comparison_data.append({
            'Métrique': metric_name,
            'Extrême→Extrême': f"{extreme_val:.4f} ({extreme_val*100:.2f}%)",
            'Croisement→Extrême': f"{cross_val:.4f} ({cross_val*100:.2f}%)",
            'Différence': f"{(extreme_val - cross_val):.4f} ({(extreme_val - cross_val)*100:.2f}%)"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    print()
    
    # Pourcentages au-dessus des seuils
    print(f"📈 COMPARAISON SEUILS (% trades)")
    print(f"{'='*50}")
    
    threshold_comparison = []
    for threshold in thresholds:
        extreme_above = (extreme_pct.abs() >= threshold).sum() / len(extreme_pct) * 100
        cross_above = (cross_pct.abs() >= threshold).sum() / len(cross_pct) * 100
        
        threshold_comparison.append({
            'Seuil': f">{threshold*100:.0f}%",
            'Extrême→Extrême': f"{extreme_above:.1f}%",
            'Croisement→Extrême': f"{cross_above:.1f}%",
            'Différence': f"{extreme_above - cross_above:+.1f}%"
        })
    
    threshold_df = pd.DataFrame(threshold_comparison)
    print(threshold_df.to_string(index=False))
    print()
    
    # Meilleurs trades par type
    print(f"🏆 MEILLEURS TRADES PAR TYPE")
    print(f"{'='*50}")
    
    # Type 1: Extrême → Extrême
    best_extreme = extreme_pct.nlargest(3)
    worst_extreme = extreme_pct.nsmallest(3)
    
    print(f"📈 Top 3 Extrême→Extrême:")
    for i, (idx, value) in enumerate(best_extreme.items(), 1):
        side = df.loc[idx, 'side']
        print(f"   {i}. {side}: {value:.4f} ({value*100:.2f}%)")
    
    print(f"\n📉 Pires 3 Extrême→Extrême:")
    for i, (idx, value) in enumerate(worst_extreme.items(), 1):
        side = df.loc[idx, 'side']
        print(f"   {i}. {side}: {value:.4f} ({value*100:.2f}%)")
    
    # Type 2: Croisement → Extrême
    best_cross = cross_pct.nlargest(3)
    worst_cross = cross_pct.nsmallest(3)
    
    print(f"\n📈 Top 3 Croisement→Extrême:")
    for i, (idx, value) in enumerate(best_cross.items(), 1):
        side = df.loc[idx, 'side']
        print(f"   {i}. {side}: {value:.4f} ({value*100:.2f}%)")
    
    print(f"\n📉 Pires 3 Croisement→Extrême:")
    for i, (idx, value) in enumerate(worst_cross.items(), 1):
        side = df.loc[idx, 'side']
        print(f"   {i}. {side}: {value:.4f} ({value*100:.2f}%)")


if __name__ == "__main__":
    analyze_trade_types()
