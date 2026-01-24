Stratégie de Trading R&D  8h 
1. Structure (The Window)
• Fenêtre : 270 périodes (tf : 8h / ~3 mois).
  • Notes :
    • Le TF est configurable (ex: 8h par défaut).
    • L’horizon (~3 mois) varie avec le TF (donc le nombre de bougies peut être ajusté).

• Niveaux : 10 pivots majeurs (Δ% ≥ M).

  • Zone pivot : ±1% autour du niveau.
    • Zone = [pivot*(1-0.01), pivot*(1+0.01)]

• Construction des pivots (uniquement ici on utilise macd_hist)
  • On segmente le prix en tranches macd_hist (signe constant).
  • Pour chaque tranche, on cherche un pivot candidat en prenant un extrême de prix sur la tranche.
    • Extrême intrabar = high/low.
    • close est utilisé seulement si high==close (sommet) ou low==close (creux).
  • Force du pivot par DMI (DX/DI) :
    • Dans une tranche, +DI et -DI peuvent s’inverser.
    • On calcule dx_max_tranche en prenant le max(dx) uniquement sur les bougies où les DI sont alignés avec le signe de la tranche.
      • Tranche haussière (macd_hist>0) : aligné si plus_di > minus_di.
      • Tranche baissière (macd_hist<0) : aligné si minus_di > plus_di.
    • Si aucune bougie n’est alignée dans la tranche : pas de pivot pour cette tranche.
    • Au moment où dx atteint ce max, on définit :
      • di_sup = max(plus_di, minus_di)
      • di_min = min(plus_di, minus_di)
    • Catégories :
      • Pivot fort : dx_max_tranche > di_sup
      • Pivot moyen : di_min <= dx_max_tranche <= di_sup
      • Pivot faible : dx_max_tranche < di_min
  • Pivot testé plusieurs fois :
    • Si le prix reteste la zone pivot au fil du temps, on conserve la dernière valeur dx_max associée (dx_max_last).
    • Le pivot garde la valeur dx_max_last (c’est elle qui sert à le classer).
  • Sélection des 10 pivots :
    • On trie les pivots par dx_max_last décroissant.
    • On prend les 10 plus forts (si moins de 10, on garde ce qu’on a).

2. États du DMI
• Sommeil : DX et ADX < DI+ et DI-. DMI ignoré.

• Éveil : DX et ADX > DI minimum. Force si DX > ADX.

• Règle d'or : Le retour de la Stochastique d'un extrême a priorité sur la force du DX.

3. Logique d'Exécution
• Signaux
  • Signal “prématuré” (en zone pivot)
    • Contexte : le prix est dans la zone pivot (±1%).
    • Conditions (ordre libre, peuvent être sur des bougies différentes) :
      • Contact zone pivot
      • Croisement Stoch (anticipation)
      • Sortie extrême Stoch %D
      • Retour CCI depuis son sommet/creux

  • Signal “validé” (hors zone pivot)
    • Contexte : le prix est hors zone pivot.
    • Conditions (ordre libre, peuvent être sur des bougies différentes) :
      • Sortie extrême Stoch %D
      • Croisement ADX par DX à la baisse (essoufflement)

4. Optimisation Robuste (Optuna)
• Objectif : limiter l’overfitting en évaluant chaque set de paramètres sur plusieurs régimes.

• 2 optimiseurs disponibles :
  • Multi-objectifs : scripts/44_optuna_new_strategie_flip_multi.py
    • maximise equity_end
    • minimise abs(max_dd)
    • maximise winrate
  • Score (objectif unique) : scripts/45_optuna_new_strategie_flip_score.py
    • score = equity_end - dd_weight * abs(max_dd) + winrate_weight * winrate

• Universe (multi-données) :
  • Multi-assets : SOLUSDT, ETHUSDT
  • Multi-timeframes : 4h, 6h
  • Multi-années : 2023, 2024, 2025
  • Chaque trial Optuna est évalué sur toutes les combinaisons (asset, TF, année).
  • Les métriques sont agrégées (moyenne sur les combinaisons).

• Walk-forward (forward test) :
  • Pour chaque année, on fait un split temporel :
    • train_pct = 0.7
    • test_pct = 0.3 (forward)
  • Le backtest / score est calculé sur la partie test (forward) uniquement.

• Config :
  • Multi-objectifs : configs/optuna_new_strategie_flip_multi_example.yaml
  • Score : configs/optuna_new_strategie_flip_score_example.yaml
  • Paramètres clés :
    • universe.symbols / universe.intervals / universe.years
    • walk_forward.mode=single_split, walk_forward.train_pct

• Exécution :
  • Multi-objectifs :
    • python scripts/44_optuna_new_strategie_flip_multi.py --config configs/optuna_new_strategie_flip_multi_example.yaml
  • Score :
    • python scripts/45_optuna_new_strategie_flip_score.py --config configs/optuna_new_strategie_flip_score_example.yaml

• Sorties :
  • Un fichier sqlite Optuna (storage) est créé dans output.out_dir.
  • optuna_trials.csv (tous les trials)
  • pareto_trials.csv (multi-objectifs)
  • best_by_*_backtest_config.yaml (config YAML rejouable avec scripts/43_backtest_new_strategie_flip_yaml.py)
