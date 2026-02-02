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

## Démo R&D : `scripts/99_demo_zerem.py`

Cette démo sert à explorer une stratégie basée sur :

- **[zone extrême CCI]** Une zone `creux/sommet` est détectée via `CCI`.
- **[structure (confluence de séries)]** Pendant la zone, on détecte une structure (ex: higher-low en creux / lower-high en sommet) sur une ou plusieurs séries.
- **[timing]** L’entrée est déclenchée par la validation du timing via `macd_hist` (cross/changement de signe), indépendamment du moment exact de la structure.
- **[confluence TF CCI (optionnel)]** Filtre multi-timeframe sur la *zone CCI uniquement* pour les entrées.

### Exécution (recommandée)

Utilise l’environnement virtuel fourni :

```bash
./venv_optuna/bin/python scripts/99_demo_zerem.py --help
```

### Ligne de commande

#### Options générales

- **`--symbol`** Symbole (ex: `BTCUSDT`, `LINKUSDT`). Défaut: `BTCUSDT`.
- **`--timeframe`** Timeframe (ex: `1m`, `5m`, `15m`, `1h`, `4h`, `1d`). Défaut: `5m`.
- **`--limit`** Nombre de bougies (si `--start-date/--end-date` ne sont pas fournis). Défaut: `1000`.
- **`--start-date`** Date de début `YYYY-MM-DD`.
- **`--end-date`** Date de fin `YYYY-MM-DD`.

#### Zone extrême (CCI)

- **`--cci-low`** Seuil bas CCI pour détecter un `creux` (zone extrême basse). Défaut: `-100.0`.
- **`--cci-high`** Seuil haut CCI pour détecter un `sommet` (zone extrême haute). Défaut: `100.0`.

#### Détection de structure

- **`--mode`** Mode structure.
  - Valeurs: `single`, `confluence`.
  - Défaut: `confluence`.
- **`--signal-from`** Série utilisée si `--mode=single`. Ex: `price`. Défaut: `price`.
- **`--series`** Séries utilisées si `--mode=confluence`.
  - Format: liste séparée par virgules.
  - Si absent: toutes les séries disponibles.

#### Simulation de trades (entrée/sortie)

- **`--simulate-trades`** Active la simulation bougie-par-bougie.
- **`--trade-direction`** Direction.
  - Valeurs: `long`, `short`, `both`.
  - Défaut: `both`.
- **`--trade-min-confluence`** Nombre minimum de séries en structure OK pour créer un signal d’entrée. Défaut: `1`.
- **`--use-fixed-stop`** Active un stop fixe basé sur l’extrême confirmé précédent (sur la série `price`).
- **`--stop-buffer-pct`** Buffer stop.
  - Notation: **fraction** (ex: `0.01` = `1%`, `0.02` = `2%`).
  - Défaut: `0.0`.
- **`--extreme-confirm-bars`** Confirme un extrême uniquement après `N` bougies clôturées après l’extrême.
  - `0` = désactivé.
  - Défaut: `0`.

#### Timing d’entrée via `macd_hist`

- **`--entry-no-hist-abs-growth`** Désactive la contrainte “croissance de `|macd_hist|`” sur l’entrée.

Si la contrainte est active (par défaut), l’entrée exige une **croissance relative** de `|macd_hist|` (sans seuil absolu fixe).

#### Confluence multi-timeframe (CCI) pour l’entrée (optionnel)

Ce filtre ne touche **que** la *zone* CCI (pas les structures séries, ni le timing).

- **`--entry-cci-tf-confluence`** Active le filtre.
- **`--entry-cci-tf-max-combo-size`** Taille max des combos timeframes testés.
  - Inclut le TF de simulation.
  - Exemple: `2` signifie “TF de base + 1 TF supérieur”.
  - Défaut: `1` (désactivé).
- **`--entry-cci-tf-confluence-tfs`** (hors grid-search) Liste de TF supplémentaires exigés en zone extrême CCI.
  - Format: `"1h,1d"`.
  - Contrainte: TF supplémentaires doivent être `>= --timeframe`.

#### Sortie B (après extrême)

- **`--exit-b-mode`** Mode de sortie B (cross “contre-mouvement” après extrême).
  - Valeurs: `macd`, `stoch`, `klinger`, `none`.
  - Défaut: `macd`.

#### Grid-search (R&D)

- **`--grid-search`** Active le grid-search.
- **`--grid-timeframes`** Liste des TF testés.
  - Format: `"5m,15m,1h,2h,6h,1d"`.
  - Défaut: `5m,15m,1h,2h,6h,1d`.
- **`--grid-year`** Raccourci: fixe automatiquement `--start-date` et `--end-date` pour une année.
- **`--grid-ensure-year-cache`** Construit le cache annuel CSV si absent.
- **`--grid-offline`** N’utilise que les CSV en cache (aucun appel API).
- **`--grid-allow-warmup-before-start`** Autorise le warmup d’indicateurs avant `start-date`.
- **`--grid-stop-buffers`** Liste des buffers stop.
  - Format:
    - `"1,2,5,10"` (valeurs **en %** si `>= 1` => `2` = `2%` = `0.02`)
    - ou `"0.02,0.05"` (valeurs **en fraction**)
  - Défaut: `1` (donc `1%` => `0.01`).
- **`--grid-series`** Sous-ensemble de séries pour le grid (sinon toutes).
- **`--grid-max-combo-size`** Taille max des combinaisons de séries testées. `0` = toutes.
- **`--grid-top`** Nombre de lignes affichées. Défaut: `30`.
- **`--grid-out-csv`** Chemin CSV de sortie.
- **`--cache-max-missing`** Tolérance: nb max de bougies manquantes dans un CSV cache avant suppression + refetch.
- **`--cache-no-validate`** Désactive la validation complétude des CSV cache.

### Score du grid-search

Le tri se fait sur `ratio_final`:

- **`ratio_dd`** = `equity / max_dd`
- **`ratio_mae`** = `equity / abs(worst_mae)`
- **`ratio_final`** = `(ratio_dd + ratio_mae) / 2`

### Cas d’utilisation (commandes utiles)

#### 1) Démo rapide (fetch réseau)

```bash
./venv_optuna/bin/python scripts/99_demo_zerem.py \
  --symbol BTCUSDT \
  --timeframe 5m \
  --limit 1500
```

#### 2) Simuler les trades (structure + timing)

```bash
./venv_optuna/bin/python scripts/99_demo_zerem.py \
  --symbol BTCUSDT \
  --timeframe 5m \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --simulate-trades \
  --mode confluence \
  --series price,asi,pvt \
  --trade-min-confluence 2 \
  --use-fixed-stop \
  --stop-buffer-pct 0.01
```

#### 3) Simuler les trades avec confluence CCI multi-TF (hors grid)

```bash
./venv_optuna/bin/python scripts/99_demo_zerem.py \
  --symbol BTCUSDT \
  --timeframe 5m \
  --start-date 2025-01-01 \
  --end-date 2025-03-01 \
  --simulate-trades \
  --mode confluence \
  --series price,asi,pvt \
  --trade-min-confluence 2 \
  --entry-cci-tf-confluence \
  --entry-cci-tf-confluence-tfs 1h,1d
```

#### 4) Grid-search (toutes combinaisons de séries)

```bash
./venv_optuna/bin/python scripts/99_demo_zerem.py \
  --grid-search \
  --symbol BTCUSDT \
  --grid-year 2025 \
  --grid-timeframes 5m,15m,1h \
  --grid-max-combo-size 3 \
  --grid-top 50
```

#### 5) Grid-search offline (cache uniquement)

```bash
./venv_optuna/bin/python scripts/99_demo_zerem.py \
  --grid-search \
  --symbol BTCUSDT \
  --grid-year 2025 \
  --grid-timeframes 5m,15m,1h \
  --grid-offline
```

#### 6) Grid-search + confluence CCI multi-TF (zone uniquement)

```bash
./venv_optuna/bin/python scripts/99_demo_zerem.py \
  --grid-search \
  --symbol BTCUSDT \
  --grid-year 2025 \
  --grid-timeframes 5m,15m,1h,1d \
  --entry-cci-tf-confluence \
  --entry-cci-tf-max-combo-size 2 \
  --grid-max-combo-size 2 \
  --grid-top 50
```

#### 7) Bornes d’optimisation (proposition à ajuster)

L’objectif ici est de définir un **espace de recherche** raisonnable pour éviter une explosion combinatoire, tout en couvrant les paramètres qui changent le plus le comportement.

Paramètres “axe” recommandés (bornes initiales):

- **Période**
  - `--grid-year`: une année complète (ex: `2025`) pour un premier tri.
  - Option: faire un second passage sur une sous-période (ex: 3 mois) si tu veux itérer vite.

- **Timeframes (univers TF)**
  - Petit espace (rapide): `--grid-timeframes 5m,15m,1h`
  - Espace moyen: `--grid-timeframes 5m,15m,1h,2h`
  - Espace large: `--grid-timeframes 5m,15m,1h,2h,6h,1d`

- **Stop buffer (notation)**
  - Proposition (notation en %): `--grid-stop-buffers 2,5,10,15,20,25,30`
  - Équivalence: `2` (grid) = `2%` = `0.02` (même buffer que `--stop-buffer-pct 0.02`)

- **Taille max des combos de séries**
  - `--grid-max-combo-size`: `1..4` (commencer par `3`)

- **Confluence CCI multi-TF (optionnel)**
  - Sans confluence: ne pas mettre `--entry-cci-tf-confluence`
  - Avec confluence: `--entry-cci-tf-confluence` et tester `--entry-cci-tf-max-combo-size` dans `2..3`
    - `2` = TF de base + 1 TF supérieur
    - `3` = TF de base + 2 TF supérieurs

- **Confirmation d’extrême (N bougies)**
  - `--extreme-confirm-bars`: tester `0..3` (ex: `0,1,2,3`)

- **Filtre timing MACD hist (abs growth)**
  - Actif (par défaut): la condition est une croissance relative de `|macd_hist|` (sans seuil)
  - Désactivé: `--entry-no-hist-abs-growth`

- **Mode de sortie B**
  - `--exit-b-mode`: tester `macd`, `stoch`, `klinger` (et `none` si tu veux comparer sans ce mécanisme)

Commande “base” (un run):

```bash
./venv_optuna/bin/python scripts/99_demo_zerem.py \
  --grid-search \
  --symbol BTCUSDT \
  --grid-year 2025 \
  --grid-timeframes 5m,15m,1h,2h \
  --grid-stop-buffers 2,5,10,15,20,25,30 \
  --grid-max-combo-size 3 \
  --extreme-confirm-bars 2 \
  --exit-b-mode macd \
  --grid-top 50 \
  --grid-out-csv data/processed/grid_zerem_bounds_run.csv
```

Commande “sweep” (balayage simple des bornes via bash):

```bash
for exit_b in macd stoch klinger; do
  for n in 0 1 2 3; do
    ./venv_optuna/bin/python scripts/99_demo_zerem.py \
      --grid-search \
      --symbol BTCUSDT \
      --grid-year 2025 \
      --grid-timeframes 5m,15m,1h,2h \
      --grid-stop-buffers 2,5,10,15,20,25,30 \
      --grid-max-combo-size 3 \
      --extreme-confirm-bars ${n} \
      --exit-b-mode ${exit_b} \
      --grid-top 50 \
      --grid-out-csv data/processed/grid_zerem_bounds_exit_${exit_b}_n_${n}.csv
  done
done
```


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


 axe : 1- las codition de deux cloture apres extreme est  optionel avec ces parametre et c'est seqetiel  on entre en extreme ensuite on a un extreme candidat s'il reste extreme des deux cloture selon activation et parametre apres lui et apres . réponse qu question 1-  l'extreme dois l'extre pour les deux vougie apres lui ex si plus petit par raport au bougie avant lui il dois l'etre pour les deux bougie cloturé apres lui (optioel et parametrable a  bougie apres lui. 2-  oui 3- ca veux dire le croisement est independant n'est pas obligé d'etre pendant l'extreme mais apres , pas de limit de bougie car le marché est dynamique 
