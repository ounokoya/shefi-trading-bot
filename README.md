# agent_economique_stable_py

## Description
Ce dépôt contient des fonctions d’indicateurs (inspirées du comportement TradingView) et un script de comparaison avec `pandas_ta` via des données Bybit.

## Installation
```bash
pip install -r requirements.txt
```

## Exécution
### Comparer les indicateurs TradingView vs pandas_ta (Bybit)
```bash
python compare_indicators_bybit.py --symbol BTCUSDT --interval 240 --limit 300
```

Le script génère un CSV et affiche un résumé des écarts.

## Structure
- `tv_indicators.py`
  Fonctions d’indicateurs (SMA/EMA/RMA/VWMA/MACD/CCI/MFI/ATR).
- `compare_indicators_bybit.py`
  Télécharge des chandeliers Bybit, calcule les indicateurs, compare avec `pandas_ta`, et exporte en CSV.
- `libs/`
  Package utilitaire interne (petites fonctions réutilisables), organisé en sous-dossiers.

## Notes
- Les scripts supposent un accès réseau sortant pour interroger l’API Bybit.


pour commencer on vas créé une optimisation avec optuna  pour trouver les bon parametre de macd qui donne le plus de trade avec des branche plus propre moin de dd entre ouverture et fermeture de position et  capt au moin +0.7% pour les long et -0.7% pour les short   

6, 13, 5
24, 52, 18


./venv_optuna/bin/python scripts/07_extract_legs_and_stats.py \
  --mode trades \
  --hx-enable \
  --hx-variant hxd \
  --in-csv data/processed/klines/LINKUSDT_5m_2020-01-01_2024-12-31_with_macd_12_26_9_with_tranches_and_blocks_first_candidate_vwma4_12_macd_align.csv \
  --stop-mode pct --stop-pct 0.01 --stop-fill stop_level


  ./venv_optuna/bin/python scripts/13_optuna_walk_forward.py   --config configs/optuna_walk_forward_example.yaml
trial=0 value=-3.452906 test_equity_sum=0.996192 test_max_dd_min=-0.246781 test_trades=142 test_winrate=0.4718
trial=1 value=-8.510011 test_equity_sum=-1.886802 test_max_dd_min=-0.377817 test_trades=969 test_winrate=0.8483
trial=2 value=-2.025386 test_equity_sum=-0.369565 test_max_dd_min=-0.056422 test_trades=36 test_winrate=0.5833
trial=3 value=-2.170449 test_equity_sum=-0.133106 test_max_dd_min=-0.123794 test_trades=31 test_winrate=0.3871
trial=4 value=-9.703635 test_equity_sum=-1.800220 test_max_dd_min=-0.347710 test_trades=316 test_winrate=0.3924
Optuna walk-forward optimization:
- study_name: walk_forward_opt
- best_value: -2.0253864527606655
- best_score_recomputed: -2.0253864527606655
Wrote: data/processed/backtests/optuna_walk_forward/optuna_trials.csv
Wrote: data/processed/backtests/optuna_walk_forward/best_backtest_config.yaml
Wrote: data/processed/backtests/optuna_walk_forward/best_walk_forward_folds.csv
Wrote: data/processed/backtests/optuna_walk_forward/optuna_summary.yaml .c'est c'est pas du tout bon des theta avectresmovaus equity end sont mailleur que ceux avecbon  en equity positif et bon ration c'est pas bon peut importe tot les critere un bon theta est un equity > 0 , et plus c'est grand et bon ration c'est meilleur .  



c'est possible de faire toute les combinanison de serie possible de 1 jusuqu'a n , apploquer le trade sur chacun sur chaque tf . une sorte d'espace de recherche sur période donné debu fin et classer selon les  les plus performant l'idée etand d'ajouter dd (perte total sur equity) et le max pnl négatif atteind depuis ouverture de chaque trade .. equity = somme de tout  les % de prix capté .  

wahou c'est super ca, est ce qu'on peut reutiliser les csv de 2025 déja telechargé ? on peux ajouter different distance de % entre extreme précedent et stop ? valeur : 2%, 5%, 10%, 15%, 20%, 25%, 30% ? 

