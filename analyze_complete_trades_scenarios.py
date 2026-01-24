#!/usr/bin/env python3
"""
Analyse complète des scénarios de trade avec drawdowns :
1. t0 → t1 (extrême → extrême)
2. t0 → t_fav (extrême → favorable)
3. cross → t_fav (croisement → favorable)
4. cross → t1 (croisement → extrême)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


def analyze_complete_scenarios():
    """Analyse tous les scénarios de trade avec drawdowns."""
    
    # Charger le fichier des trades
    trades_file = Path("data/processed/blocks/LINKUSDT_4h_2020-01-01_2025-12-31_trades.csv")
    
    if not trades_file.exists():
        print(f"❌ Fichier introuvable: {trades_file}")
        return
    
    df = pd.read_csv(trades_file)
    
    print(f"📊 ANALYSE COMPLÈTE DES SCÉNARIOS DE TRADE")
    print(f"{'='*80}")
    print(f"📁 Fichier: {trades_file}")
    print(f"📈 Nombre total de trades: {len(df)}")
    print()
    
    # Vérifier les colonnes nécessaires
    required_cols = [
        'exit_pct', 'cross_exit_pct_close', 'fav_pct', 
        'dd_max_trade_pct', 'cross_dd_max_trade_pct_entry_open', 'side',
        'dd_max_to_fav_pct', 'cross_dd_max_to_fav_pct_entry_open'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Colonnes manquantes: {missing_cols}")
        return
    
    # Convertir en numérique
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Séparer par direction
    long_df = df[df['side'] == 'LONG']
    short_df = df[df['side'] == 'SHORT']
    
    # Définir les scénarios
    scenarios = [
        {
            'name': 't0 → t1 (Extrême → Extrême)',
            'col': 'exit_pct',
            'dd_col': 'dd_max_trade_pct',
            'description': 'Trade complet du bloc'
        },
        {
            'name': 't0 → t_fav (Extrême → Favorable)',
            'col': 'fav_pct',
            'dd_col': 'dd_max_to_fav_pct',
            'description': 'Sortie au point optimal'
        },
        {
            'name': 'cross → t_fav (Croisement → Favorable)',
            'col': 'cross_fav_pct',
            'dd_col': 'cross_dd_max_to_fav_pct_entry_open',
            'description': 'Entrée tardive, sortie optimale'
        },
        {
            'name': 'cross → t1 (Croisement → Extrême)',
            'col': 'cross_exit_pct_close',
            'dd_col': 'cross_dd_max_trade_pct_entry_open',
            'description': 'Entrée et sortie tardives'
        }
    ]
    
    # ========================================
    # ANALYSE GLOBALE PAR SCÉNARIO
    # ========================================
    print(f"🎯 ANALYSE GLOBALE PAR SCÉNARIO")
    print(f"{'='*80}")
    
    results = []
    
    for scenario in scenarios:
        col = scenario['col']
        dd_col = scenario['dd_col']
        
        if col not in df.columns:
            print(f"⚠️  Colonne {col} non trouvée, calcul...")
            continue
            
        returns = df[col].dropna()
        dd = df[dd_col].dropna() if dd_col in df.columns else pd.Series()
        
        if len(returns) == 0:
            continue
            
        # Stats de performance
        mean_return = returns.mean()
        median_return = returns.median()
        std_return = returns.std()
        
        # Stats de drawdown
        mean_dd = dd.mean() if len(dd) > 0 else 0
        max_dd = dd.max() if len(dd) > 0 else 0
        
        # Trades au-dessus des seuils
        above_1pct = (returns.abs() >= 0.01).sum()
        above_2pct = (returns.abs() >= 0.02).sum()
        above_5pct = (returns.abs() >= 0.05).sum()
        
        # Trades positifs vs négatifs
        pos_trades = (returns > 0).sum()
        neg_trades = (returns < 0).sum()
        
        # Ratio de Sharpe approximatif
        sharpe = mean_return / std_return if std_return > 0 else 0
        
        results.append({
            'Scénario': scenario['name'],
            'Description': scenario['description'],
            'Trades': len(returns),
            'Mean %': f"{mean_return*100:.2f}%",
            'Median %': f"{median_return*100:.2f}%",
            'Std %': f"{std_return*100:.2f}%",
            'Sharpe': f"{sharpe:.3f}",
            'Mean DD %': f"{mean_dd*100:.2f}%",
            'Max DD %': f"{max_dd*100:.2f}%",
            '>1%': f"{above_1pct} ({above_1pct/len(returns)*100:.1f}%)",
            '>2%': f"{above_2pct} ({above_2pct/len(returns)*100:.1f}%)",
            '>5%': f"{above_5pct} ({above_5pct/len(returns)*100:.1f}%)",
            'Pos': f"{pos_trades} ({pos_trades/len(returns)*100:.1f}%)",
            'Neg': f"{neg_trades} ({neg_trades/len(returns)*100:.1f}%)"
        })
    
    # Afficher le tableau récapitulatif
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print()
    
    # ========================================
    # ANALYSE PAR DIRECTION (LONG/SHORT)
    # ========================================
    print(f"📊 ANALYSE PAR DIRECTION (LONG vs SHORT)")
    print(f"{'='*80}")
    
    for side, side_df in [('LONG', long_df), ('SHORT', short_df)]:
        print(f"\n🔴 {side} TRADES:")
        print("-" * 50)
        
        side_results = []
        
        for scenario in scenarios:
            col = scenario['col']
            dd_col = scenario['dd_col']
            
            if col not in side_df.columns:
                continue
                
            returns = side_df[col].dropna()
            dd = side_df[dd_col].dropna() if dd_col in side_df.columns else pd.Series()
            
            if len(returns) == 0:
                continue
                
            mean_return = returns.mean()
            mean_dd = dd.mean() if len(dd) > 0 else 0
            pos_trades = (returns > 0).sum()
            
            side_results.append({
                'Scénario': scenario['name'][:20] + '...',
                'Mean %': f"{mean_return*100:.2f}%",
                'DD Moy %': f"{mean_dd*100:.2f}%",
                'Pos %': f"{pos_trades/len(returns)*100:.1f}%",
                'Trades': len(returns)
            })
        
        side_df_results = pd.DataFrame(side_results)
        print(side_df_results.to_string(index=False))
    
    # ========================================
    # COMPARAISON DES PERFORMANCES
    # ========================================
    print(f"\n🔄 COMPARAISON DES PERFORMANCES")
    print(f"{'='*80}")
    
    # Extraire les moyennes pour comparaison
    comparison_data = []
    for result in results:
        scenario_name = result['Scénario']
        mean_pct = float(result['Mean %'].replace('%', ''))
        mean_dd_pct = float(result['Mean DD %'].replace('%', ''))
        
        comparison_data.append({
            'Scénario': scenario_name,
            'Performance': mean_pct,
            'DD Moyen': mean_dd_pct,
            'Ratio Perf/DD': mean_pct / mean_dd_pct if mean_dd_pct != 0 else 0
        })
    
    comp_df = pd.DataFrame(comparison_data)
    
    print("📈 Performance vs Drawdown:")
    for _, row in comp_df.iterrows():
        perf = row['Performance']
        dd = row['DD Moyen']
        ratio = row['Ratio Perf/DD']
        print(f"  {row['Scénario'][:30]:30} | {perf:+6.2f}% | DD: {dd:+6.2f}% | Ratio: {ratio:+.3f}")
    
    print()
    
    # ========================================
    # MEILLEURS ET PIRES TRADES PAR SCÉNARIO
    # ========================================
    print(f"🏆 MEILLEURS ET PIRES TRADES PAR SCÉNARIO")
    print(f"{'='*80}")
    
    for scenario in scenarios[:2]:  # Top 2 scénarios principaux
        col = scenario['col']
        
        if col not in df.columns:
            continue
            
        returns = df[col].dropna()
        if len(returns) == 0:
            continue
        
        best = returns.nlargest(3)
        worst = returns.nsmallest(3)
        
        print(f"\n📊 {scenario['name']}:")
        print(f"  Top 3:")
        for i, (idx, value) in enumerate(best.items(), 1):
            side = df.loc[idx, 'side']
            print(f"    {i}. {side}: {value:.4f} ({value*100:.2f}%)")
        
        print(f"  Pires 3:")
        for i, (idx, value) in enumerate(worst.items(), 1):
            side = df.loc[idx, 'side']
            print(f"    {i}. {side}: {value:.4f} ({value*100:.2f}%)")
    
    # ========================================
    # CONCLUSIONS
    # ========================================
    print(f"\n💡 CONCLUSIONS")
    print(f"{'='*80}")
    
    if len(results) > 0:
        best_scenario = max(results, key=lambda x: float(x['Mean %'].replace('%', '')))
        lowest_dd = min(results, key=lambda x: float(x['Mean DD %'].replace('%', '')))
        
        print(f"🏆 Meilleure performance: {best_scenario['Scénario']}")
        print(f"   → {best_scenario['Mean %']} moyen")
        print(f"🛡️  Drawdown minimum: {lowest_dd['Scénario']}")
        print(f"   → {lowest_dd['Mean DD %']} DD moyen")
        
        print(f"\n📊 Recommandations:")
        print(f"  • Le scénario optimal semble être: {best_scenario['Scénario']}")
        print(f"  • Éviter le scénario avec le plus grand drawdown")
        print(f"  • Considérer le ratio Performance/DD pour le risque")


if __name__ == "__main__":
    analyze_complete_scenarios()
