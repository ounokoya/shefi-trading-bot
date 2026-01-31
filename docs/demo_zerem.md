# Demo Zerem – Détection de structures multi-indicateurs

## Objectif

Script de démonstration qui :
- Télécharge les N dernières bougies pour une paire/timeframe via Bybit.
- Délimite les zones d’extrême via le CCI.
- Pour chaque série cible (prix, ASI, PVT, MACD line, DMI DX, Klinger KVO, Klinger signal, MFI, Stoch K, Stoch D) :
  - Détecte les structures haussières/baissières en utilisant les zones CCI.
  - Retourne les extrêmes retenus et les structures déclenchées à partir de l’alignement des extrêmes successifs.
- Produit une sortie agrégée selon un mode :
  - Mode `single` : n’utiliser qu’une seule série pour la détection de structure.
  - Mode `confluence` : lister les séries en confluence de structure au niveau d’une même zone CCI.

## Fonctions à implémenter

### `delimit_zones_by_cci(df, cci_col, low_threshold=-100, high_threshold=100)`
- Détecte les zones où CCI < low_threshold (creux) ou CCI > high_threshold (sommets).
- Retourne une liste de zones avec start_idx, end_idx, type (creux/sommet).

### `find_structures_price(df, zones, price_col='close')`
- Utilise les zones CCI pour extraire les prix extrêmes (min dans creux, max dans sommets).
- Détecte les structures haussières (creux_2 > creux_1) et baissières (sommets_2 < sommets_1).
- Retourne la liste des extrêmes par zone et, si applicable, la structure déclenchée à cette zone.
- Note : une structure est déclenchée à la zone courante en comparant l’extrême de cette zone avec l’extrême de la zone précédente du même type (creux->creux, sommet->sommet).

### `find_structures_asi(df, zones, asi_col='asi')`
- Même logique que price mais sur la série ASI.
- L’ASI (Accumulative Swing Index) est sensible aux swings et peut révéler des structures de momentum.

### `find_structures_pvt(df, zones, pvt_col='pvt')`
- Même logique sur PVT (Price-Volume Trend).
- Le PVT intègre volume et prix ; ses structures indiquent des changements de tendance avec confirmation volume.

### `find_structures_macd_line(df, zones, macd_line_col='macd_line')`
- Même logique sur la ligne MACD (pas l’histogramme).
- Structures sur la ligne MACD révèlent les accélérations/ralentissements du momentum.

### `find_structures_dmi_dx(df, zones, dx_col='dx')`
- Même logique sur le DX (Directional Movement Index) du DMI.
- Le DX mesure la force de la tendance ; ses structures montrent des consolidations/accélérations de force.

### `find_structures_klinger_kvo(df, zones, kvo_col='kvo')`
- Même logique sur le Klinger Volume Oscillator (KVO).
- Le KVO compare volume à la pression prix ; ses structures signalent des divergences volume/prix.

### `find_structures_klinger_signal(df, zones, ks_col='klinger_signal')`
- Même logique sur la ligne signal du Klinger.
- Structures sur la signal peuvent anticiper les retournements du KVO.

### `find_structures_mfi(df, zones, mfi_col='mfi')`
- Même logique sur MFI (Money Flow Index).
- Le MFI est un RSI pondéré volume ; ses structures indiquent des surachat/survente avec confirmation volume.

### `find_structures_stoch_k(df, zones, stoch_k_col='stoch_k')`
- Même logique sur le %K de Stochastique.
- Le %K réagit rapidement aux changements de prix ; ses structures révèlent les retournements courts.

### `find_structures_stoch_d(df, zones, stoch_d_col='stoch_d')`
- Même logique sur le %D de Stochastique.
- Le %D lissé donne des structures plus fiables mais moins rapides.

## Sorties attendues

- Pour chaque zone CCI et chaque série cible :
  - Un extrême : date (dt), type (creux/sommet), série, valeur extrême.
  - Si applicable, une structure déclenchée à cette zone : `structure_haussiere` (zones creux) ou `structure_baissiere` (zones sommet).
- Sortie `single` : liste chronologique des zones où la série choisie déclenche une structure, avec le détail des deux extrêmes (précédent + courant).
- Sortie `confluence` : pour chaque zone CCI, liste des séries qui déclenchent une structure dans cette zone (confluence par zone).

## Implémentation

- Réutiliser `libs/data_loader.py` pour télécharger les données Bybit.
- Réutiliser les implémentations d’indicateurs existantes dans `libs/indicators/`.
- Script principal dans `scripts/XX_demo_zerem.py`.

## Paramètres indicateurs par timeframe

Les paramètres CCI et Stochastique sont choisis automatiquement en fonction de `--timeframe` pour couvrir un horizon cohérent :

- `5m` : horizon ~8h (1 session)
  - `CCI period = 96`
  - `Stoch = (96, 2, 3)`
- `15m` : horizon ~24h (1 jour)
  - `CCI period = 96`
  - `Stoch = (96, 2, 3)`
- `1h` : horizon ~1 semaine
  - `CCI period = 168`
  - `Stoch = (168, 2, 3)`
- `2h` : horizon ~1 semaine
  - `CCI period = 84`
  - `Stoch = (84, 2, 3)`
- `6h` : horizon ~1 mois
  - `CCI period = 120`
  - `Stoch = (120, 2, 3)`
- `1d` : horizon ~3 mois
  - `CCI period = 90`
  - `Stoch = (90, 2, 3)`



## Confluence par zone CCI

La confluence n’est pas évaluée « au même timestamp », mais au niveau d’une même zone CCI.

Pour une zone donnée :
- Chaque série produit un seul extrême (min ou max) dans la zone.
- Une série est dite « en confluence de structure » sur cette zone si elle déclenche une structure au moment de cet extrême, en comparant avec l’extrême précédent du même type.

## Séries disponibles

Les séries utilisables via `--signal-from` (mode `single`) et `--series` (mode `confluence`) sont :

- `price`
- `asi`
- `pvt`
- `macd_line`
- `dmi_dx`
- `klinger_kvo`
- `klinger_signal`
- `mfi`
- `stoch_k`
- `stoch_d`

## Modes et paramètres

- Mode `single` :
  - Paramètre : `--signal-from <serie>` (ex : `price`, `asi`, `mfi`...)
  - Sortie : uniquement les structures de cette série.

- Mode `confluence` :
  - Paramètres :
    - `--series <liste>` optionnel : liste des séries à inclure dans la confluence (si absent : toutes les séries).
  - Sortie : par zone CCI, la liste des séries déclenchant une structure dans la zone.

## Backtest / trading (optionnel)

Le script peut aussi simuler un trading **bougie-par-bougie** basé sur :
- Les extrêmes (définis par les zones CCI).
- La structure (haussière/baissière) calculée à partir des extrêmes.
- Le timing (déclenchement) fourni par un changement de signe de `macd_hist`.

### États d’extrême

- **Extrême potentiel** : extrême courant dans une zone CCI non encore clôturée (il peut évoluer tant que la zone continue).
- **Extrême confirmé** : quand la zone CCI se ferme, l’extrême potentiel devient confirmé.

La structure est évaluée **pendant la zone** en comparant l’extrême potentiel courant avec le dernier extrême confirmé du même type (creux->creux, sommet->sommet).

### Timing MACD

L’exécution d’un signal (ou flip de position) se fait quand `macd_hist` change de signe :
- Passage `<= 0` -> `> 0` : timing haussier.
- Passage `>= 0` -> `< 0` : timing baissier.

### Paramètres CLI

- `--simulate-trades` : active la simulation de trades.
- `--trade-direction` : `long`, `short` ou `both`.
- `--trade-min-confluence` : seuil minimal de confluence (nombre de séries validant la structure) pour qu’un signal soit exécutable.
  - En mode `confluence`, le seuil s’applique sur la liste `--series`.
  - En mode `single`, le seuil effectif est `1` (une seule série).
- `--use-fixed-stop` : active un stop fixe.
- `--stop-buffer-pct` : buffer du stop en pourcentage (ex: `0.001` = 0.1%).

### Sortie trades

Le script affiche :
- Un résumé : nombre de trades, winrate, moyenne et somme des %.
- La liste chronologique des trades : side, entry, exit, % capté, zone_id, et `series_ok`.

### Stop fixe (optionnel)

Quand `--use-fixed-stop` est activé, un stop est placé **juste derrière l’extrême price confirmé précédent** qui sert à valider la structure (le `prev_confirmed` de la série `price`).

- Pour un `LONG` : stop sous le précédent creux confirmé (`stop = prev_creux * (1 - stop_buffer_pct)`).
- Pour un `SHORT` : stop au-dessus du précédent sommet confirmé (`stop = prev_sommet * (1 + stop_buffer_pct)`).

Le stop est déclenché si le `low` (pour LONG) ou le `high` (pour SHORT) casse ce niveau.

Exemple :

```bash
./venv_optuna/bin/python scripts/99_demo_zerem.py \
  --symbol LINKUSDT --timeframe 5m --limit 1000 \
  --mode confluence --series asi,pvt,price \
  --simulate-trades --trade-direction both --trade-min-confluence 3
```

## Grid-search (backtest multi-TF + combinaisons de séries)

Le mode grid-search teste automatiquement :
- Plusieurs `timeframes`
- Toutes les combinaisons de séries (de taille `1..K`)
- En appliquant la simulation de trades (structure + timing MACD + stop fixe)

### Principe

- Pour chaque couple `(timeframe, combo)` :
  - Mode utilisé : `confluence`
  - `min_confluence = len(combo)`
  - Stop fixe activé avec buffer `1%` (basé sur la série `price`)

Le classement se fait par **ratio** :

- `ratio = equity / max_dd`

Avec :
- `equity` = somme des `%` captés sur tous les trades
- `max_dd` = max drawdown sur la courbe d’equity (en “points de %”)

### Sortie

Le script affiche :
- Un **TOP N par timeframe** (trié par `ratio`), avec 3 sous-blocs :
  - `ALL combos`
  - `SINGLE (k=1)`
  - `CONFLUENCE (k>=2)`
- Un **TOP global** (tous timeframes), avec les mêmes 3 sous-blocs.

Chaque ligne affiche :
- `tf=<timeframe>`
- `combo=<liste séries>`
- `min_conf=<k>` (confluence = taille du combo)
- `equity`, `dd`, `ratio`, `worst_mae`, `winrate`, `n`

### Cache CSV (Bybit)

Les données Bybit sont mises en cache dans :

- `data/raw/klines_cache/`

Format du fichier :

- `{symbol}_{timeframe}_{start_date}_{end_date}.csv`

### Paramètres CLI

- `--grid-search` : active le grid-search.
- `--grid-timeframes <list>` : liste séparée par virgule (ex: `5m,15m,1h,2h,6h,1d`).
- `--grid-series <list>` : sous-ensemble de séries (sinon toutes).
- `--grid-max-combo-size <int>` : taille max des combos (0 = toutes).
- `--grid-top <int>` : top N affiché par TF + global.
- `--grid-out-csv <path>` : export CSV contenant tous les résultats.
- `--grid-year <YYYY>` : raccourci pour tester toute l’année (ex: `2025` = `2025-01-01..2025-12-31`).
- `--grid-ensure-year-cache` : construit le cache `{symbol}_{tf}_{YYYY-01-01}_{YYYY-12-31}.csv` si absent.

### Exemple (année 2025, tous TF, top 10)

```bash
./venv_optuna/bin/python scripts/99_demo_zerem.py \
  --grid-search \
  --symbol LINKUSDT \
  --grid-year 2025 \
  --grid-timeframes 5m,15m,1h,2h,6h,1d \
  --grid-max-combo-size 3 \
  --grid-top 10 \
  --grid-ensure-year-cache \
  --grid-out-csv /tmp/zerem_grid_2025.csv
```

## Format de sortie (exemple)

Zone creux [start_dt -> end_dt]
Séries en confluence haussière : price, mfi, pvt
  price : creux=...  structure_haussiere  (prev_creux=...)
  mfi   : creux=...  structure_haussiere  (prev_creux=...)
