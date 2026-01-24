#!/usr/bin/env python3
"""
Analyse contextuelle spécifique pour t_fav (points de sortie optimaux)
"""

from __future__ import annotations

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from contextual_feature_explorer import ContextualFeatureExplorer

class TfavContextualAnalyzer(ContextualFeatureExplorer):
    """Analyseur contextuel spécialisé pour t_fav"""
    
    def __init__(self, trades_file: str, features_file: str, show_plots: bool = True):
        super().__init__(trades_file, features_file, show_plots=show_plots)
        
    def analyze_tfav_vs_t0_contexts(self):
        """Compare les contextes t_fav vs t0"""
        print("🔄 COMPARAISON CONTEXTES T_FAV vs T0")
        print("="*80)
        
        # Analyser t_fav
        tfav_results = self.analyze_contextual_patterns('tfav', top_n=10)
        
        # Analyser t0 (déjà fait mais on reprend pour comparaison)
        t0_results = self.analyze_contextual_patterns('t0', top_n=10)
        
        # Comparer les catégories
        print(f"\n📊 COMPARAISON DES TOP FEATURES PAR CATÉGORIE:")
        print("-" * 100)
        print(f"{'Catégorie':<15} {'t0 - Feature':<30} {'t_fav - Feature':<30} {'Différence'}")
        print("-" * 100)
        
        categories = ['window_3', 'window_6', 'window_12', 'normalized', 'evolution']
        
        for category in categories:
            if category in t0_results and category in tfav_results:
                t0_top = t0_results[category][0] if t0_results[category] else None
                tfav_top = tfav_results[category][0] if tfav_results[category] else None
                
                if t0_top and tfav_top:
                    t0_effect = abs(t0_top['cohens_d'])
                    tfav_effect = abs(tfav_top['cohens_d'])
                    diff = tfav_effect - t0_effect
                    
                    print(f"{category:<15} {t0_top['feature'][:28]:<30} {tfav_top['feature'][:28]:<30} {diff:+.3f}")
        
        return tfav_results, t0_results
    
    def create_tfav_detection_system(self):
        """Crée un système de détection spécialisé pour t_fav"""
        print(f"\n🎯 CRÉATION SYSTÈME DÉTECTION T_FAV")
        print("="*80)
        
        # Trouver les meilleures features pour t_fav
        best_tfav_features = self.find_contextual_combinations('tfav', max_features_per_category=3)
        
        # Générer les règles pour t_fav
        tfav_rules = self.generate_contextual_rules(best_tfav_features, 'tfav')
        
        # Tester les combinaisons pour t_fav
        tfav_combinations = self.test_contextual_combinations(tfav_rules, 'tfav', max_combination_size=3)
        
        return {
            'features': best_tfav_features,
            'rules': tfav_rules,
            'combinations': tfav_combinations
        }
    
    def create_entry_exit_system(self):
        """Crée un système complet entrée-sortie (t0 → t_fav)"""
        print(f"\n🔄 CRÉATION SYSTÈME COMPLET ENTRÉE-SORTIE")
        print("="*80)
        
        # Obtenir les meilleures règles pour t0 et t_fav
        t0_features = self.find_contextual_combinations('t0', max_features_per_category=2)
        t0_rules = self.generate_contextual_rules(t0_features, 't0')
        
        tfav_features = self.find_contextual_combinations('tfav', max_features_per_category=2)
        tfav_rules = self.generate_contextual_rules(tfav_features, 'tfav')
        
        # Prendre les meilleures règles
        best_t0_rule = t0_rules[0] if t0_rules else None
        best_tfav_rule = tfav_rules[0] if tfav_rules else None
        
        print(f"\n📋 SYSTÈME ENTRÉE-SORTIE OPTIMAL:")
        print("-" * 80)
        
        if best_t0_rule:
            print(f"🔴 ENTRÉE (t0):")
            print(f"   Feature: {best_t0_rule['feature']}")
            print(f"   Règle: {best_t0_rule['direction']} {best_t0_rule['threshold']:.6f}")
            print(f"   Précision: {best_t0_rule['precision']:.4f}")
        
        if best_tfav_rule:
            print(f"\n🟢 SORTIE (t_fav):")
            print(f"   Feature: {best_tfav_rule['feature']}")
            print(f"   Règle: {best_tfav_rule['direction']} {best_tfav_rule['threshold']:.6f}")
            print(f"   Précision: {best_tfav_rule['precision']:.4f}")
        
        # Calculer la performance théorique du système
        if best_t0_rule and best_tfav_rule:
            theoretical_performance = self._calculate_system_performance(best_t0_rule, best_tfav_rule)
            print(f"\n📈 PERFORMANCE THÉORIQUE DU SYSTÈME:")
            print(f"   Trades complets: {theoretical_performance['complete_trades']}")
            print(f"   Profit moyen: {theoretical_performance['avg_profit']:.4f} ({theoretical_performance['avg_profit']*100:.2f}%)")
            print(f"   Taux de réussite: {theoretical_performance['success_rate']:.2f}%")
        
        return {
            'entry_rule': best_t0_rule,
            'exit_rule': best_tfav_rule,
            'performance': theoretical_performance if best_t0_rule and best_tfav_rule else None
        }
    
    def _calculate_system_performance(self, entry_rule, exit_rule):
        """Calcule la performance théorique du système entrée-sortie"""
        print(f"\n🧪 CALCUL PERFORMANCE SYSTÈME...")
        
        # Simuler les signaux d'entrée
        if entry_rule['direction'] == '>':
            entry_signals = (self.features_df[entry_rule['feature']] > entry_rule['threshold']).astype(int)
        else:
            entry_signals = (self.features_df[entry_rule['feature']] < entry_rule['threshold']).astype(int)
        
        # Simuler les signaux de sortie
        if exit_rule['direction'] == '>':
            exit_signals = (self.features_df[exit_rule['feature']] > exit_rule['threshold']).astype(int)
        else:
            exit_signals = (self.features_df[exit_rule['feature']] < exit_rule['threshold']).astype(int)
        
        # Marquer les vrais points
        real_t0 = self.features_df['is_t0']
        real_tfav = self.features_df['is_tfav']
        
        # Calculer les métriques
        entry_tp = ((entry_signals == 1) & (real_t0 == 1)).sum()
        entry_fp = ((entry_signals == 1) & (real_t0 == 0)).sum()
        
        exit_tp = ((exit_signals == 1) & (real_tfav == 1)).sum()
        exit_fp = ((exit_signals == 1) & (real_tfav == 0)).sum()
        
        entry_precision = entry_tp / (entry_tp + entry_fp) if (entry_tp + entry_fp) > 0 else 0
        exit_precision = exit_tp / (exit_tp + exit_fp) if (exit_tp + exit_fp) > 0 else 0
        
        # Estimer le profit (basé sur les trades réels)
        complete_trades = min(entry_tp, exit_tp)
        
        # Récupérer les profits des trades où on a bien entré et sorti
        profits = []
        for _, trade in self.trades_df.iterrows():
            if self._ts_ms_to_idx is None:
                break

            t0_ms = trade.get('t0_ts_ms')
            tfav_ms = trade.get('tfav_ts_ms')
            if pd.isna(t0_ms) or pd.isna(tfav_ms):
                continue

            t0_idx = self._ts_ms_to_idx.get(int(t0_ms))
            tfav_idx = self._ts_ms_to_idx.get(int(tfav_ms))
            if t0_idx is None or tfav_idx is None:
                continue

            t0_detected = bool(entry_signals.iloc[t0_idx] == 1)
            tfav_detected = bool(exit_signals.iloc[tfav_idx] == 1)

            if t0_detected and tfav_detected:
                profits.append(trade['fav_pct'])
        
        avg_profit = np.mean(profits) if profits else 0
        success_rate = len([p for p in profits if p > 0]) / len(profits) * 100 if profits else 0
        
        return {
            'complete_trades': complete_trades,
            'avg_profit': avg_profit,
            'success_rate': success_rate,
            'entry_precision': entry_precision,
            'exit_precision': exit_precision,
            'total_profits': profits
        }
    
    def visualize_tfav_vs_t0_patterns(self):
        """Visualise les différences de patterns entre t_fav et t0"""
        print(f"\n📊 VISUALISATION PATTERNS T_FAV vs T0")
        print("="*80)
        
        # Créer une figure comparative
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Comparaison des features les plus importantes
        categories = ['evolution', 'normalized', 'window_3', 'window_6']
        
        for i, category in enumerate(categories):
            ax = axes[i//2, i%2]
            
            # Analyser les deux cibles
            t0_results = self.analyze_contextual_patterns('t0', top_n=5)
            tfav_results = self.analyze_contextual_patterns('tfav', top_n=5)
            
            if category in t0_results and category in tfav_results:
                t0_effects = [abs(r['cohens_d']) for r in t0_results[category][:3]]
                tfav_effects = [abs(r['cohens_d']) for r in tfav_results[category][:3]]
                
                x = np.arange(len(t0_effects))
                width = 0.35
                
                ax.bar(x - width/2, t0_effects, width, label='t0', alpha=0.7)
                ax.bar(x + width/2, tfav_effects, width, label='t_fav', alpha=0.7)
                
                ax.set_xlabel('Top Features')
                ax.set_ylabel("|Cohen's d|")
                ax.set_title(f'Comparaison {category.upper()}')
                ax.set_xticks(x)
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('tfav_vs_t0_comparison.png', dpi=300, bbox_inches='tight')
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        print("✅ Graphique sauvegardé: tfav_vs_t0_comparison.png")
    
    def run_complete_tfav_analysis(self):
        """Exécute l'analyse complète pour t_fav"""
        print("🚀 DÉMARRAGE ANALYSE COMPLÈTE T_FAV")
        print("="*80)
        
        # Charger les données
        self.load_data()
        
        # Identifier les features contextuelles
        self.identify_contextual_features()
        
        # Comparer t_fav vs t0
        print(f"\n" + "="*80)
        tfav_results, t0_results = self.analyze_tfav_vs_t0_contexts()
        
        # Créer le système t_fav
        print(f"\n" + "="*80)
        tfav_system = self.create_tfav_detection_system()
        
        # Créer le système entrée-sortie
        print(f"\n" + "="*80)
        entry_exit_system = self.create_entry_exit_system()
        
        # Visualiser les patterns
        print(f"\n" + "="*80)
        self.visualize_tfav_vs_t0_patterns()
        
        return {
            'tfav_patterns': tfav_results,
            't0_patterns': t0_results,
            'tfav_system': tfav_system,
            'entry_exit_system': entry_exit_system
        }


def main():
    """Fonction principale"""

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--trades",
        default="data/processed/blocks/LINKUSDT_4h_2020-01-01_2025-12-31_trades.csv",
    )
    ap.add_argument(
        "--features",
        default="data/processed/features/LINKUSDT_4h_2020-01-01_2025-12-31_with_rolling_quantiles.csv",
    )
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    trades_file = str(args.trades)
    features_file = str(args.features)
    
    # Vérifier les fichiers
    if not Path(trades_file).exists() or not Path(features_file).exists():
        print("❌ Fichiers introuvables")
        return
    
    show_plots = bool(os.environ.get('DISPLAY'))
    if args.show:
        show_plots = True
    if args.no_show:
        show_plots = False

    # Créer l'analyseur t_fav
    analyzer = TfavContextualAnalyzer(trades_file, features_file, show_plots=show_plots)
    results = analyzer.run_complete_tfav_analysis()
    
    print("\n🎉 ANALYSE T_FAV TERMINÉE")
    print("="*80)
    print("Système entrée-sortie contextuel prêt !")


if __name__ == "__main__":
    main()
