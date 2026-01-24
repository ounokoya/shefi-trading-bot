# Backtest "price action sur tranche"

## Objectif

Cette stratégie est un backtest séparé du backtest confluence existant.

Principe:

- Un événement "tranche/extreme/confluence" déclenche un signal de type trigger.
- Une fois le trigger détecté, le moteur passe en état **armed** (armé) dans le sens du trigger (LONG/SHORT).
- L’entrée n’est exécutée que lorsque les filtres de **price action** (VWMA / Stochastic / MACD) valident la direction armée.
- La sortie par signal (optionnelle) est déclenchée par les filtres de **price action** côté inverse (conditions bearish pour sortir un LONG, et inversement).

## Fichiers

- `libs/backtest_price_action_tranche/config.py`
- `libs/backtest_price_action_tranche/indicators.py`
- `libs/backtest_price_action_tranche/signals.py`
- `libs/backtest_price_action_tranche/engine.py`
- Runner: `scripts/12_backtest_price_action_tranche_yaml.py`
- Exemple config: `configs/backtest_price_action_tranche_example.yaml`

## Entrées / sorties (timing)

Le moteur exécute les décisions selon un schéma "close -> open".

- Les signaux sont évalués sur la clôture de la bougie `i` en utilisant une fenêtre `window_size`.
- Si une entrée/sortie est validée à `close(i)`, elle est planifiée et exécutée au `open(i+1)`.
- Les sorties TP/SL (si activées) sont évaluées en intrabar sur la bougie courante.

## Trigger (tranche/extreme/confluence)

Le trigger réutilise la logique existante de tranche/extreme/confluence via:

- `get_current_tranche_extreme_zone_confluence_signal`
- `get_current_tranche_extreme_zone_confluence_tranche_last_signal`

Configuration: `signals.trigger`.

Paramètres principaux:

- `confluence_type`
  - `instant`
  - `tranche_last`
- `mode`
  - `long` (LOW uniquement)
  - `short` (HIGH uniquement)
  - `both`
- `min_confirmed`
  - `null` => toutes les séries du preset
  - `N` => au moins N séries confirment
- `series.add` / `series.exclude`
  - permet d’ajouter/retirer des colonnes de confirmation.
  - alias supportés:
    - `cci_fast`, `cci_medium`, `cci_slow` => résolus vers `cci_<indicators.cci.*>`
    - `vwma_fast`, `vwma_medium` => résolus vers `vwma_<indicators.vwma.*>`

Seuils CCI:

- `cci_thresholds.fast|medium|slow` (optionnel)
  - si non défini, les seuils du preset sont utilisés.

### Filtre “mémoire temporelle” (pivots) — optionnel

Objectif : confirmer qu’un trigger se produit sur un niveau “solide” (pivot répété dans le temps), en consultant une mémoire de pivots construite sur la fenêtre glissante.

Activation : bloc YAML `pivot_temporal_memory`.

- `enabled` (bool)
- `radius_pct` (float) : rayon inclusif autour du niveau du trigger
- `min_fast`, `min_medium`, `min_slow` : seuils de solidité (1 slow **OU** 2 medium **OU** 4 fast)
- `max_events` : limite d’events considérés (triés du plus jeune au plus vieux)

Implémentation :

- Au moment où un trigger est détecté (`is_zone=True` + `open_side` valide), le moteur construit un `PivotRegistry` runtime à partir des extrêmes de la fenêtre glissante (1 event par tranche).
- On calcule la solidité via `PivotRegistry.temporal_memory_solidity(...)` en cherchant les pivots historiques dans la bande ±`radius_pct`.
- Si `is_solid=False`, le trigger est rejeté (la stratégie ne s’arme pas).
- Les métriques sont ajoutées dans `TriggerDecision.meta["pivot_temporal_memory"]`.

### Conditions du trigger (détaillées)

Le trigger est calculé à partir de:

- la **tranche** courante (déduite du signe de `macd_hist`)
- la détection d’un **extrême confirmé** sur une ou plusieurs séries
- une règle de **confluence** (`min_confirmed`)

#### Définition d’une tranche

Pour chaque bougie, on observe le signe de `macd_hist`:

- si `macd_hist > 0` alors signe = `+`
- si `macd_hist < 0` alors signe = `-`
- si `macd_hist == 0` alors le signe conserve la dernière valeur non nulle

La tranche courante commence au dernier changement de signe (le début de la séquence continue de `+` ou de `-`).

Conséquences:

- tranche `-` (hist négatif) => contexte “LOW” et `open_side=LONG`
- tranche `+` (hist positif) => contexte “HIGH” et `open_side=SHORT`

#### Définition d’un “extrême confirmé” (par série)

Pour une série donnée (`close`, `vwma_4`, `cci_30`, etc.), l’extrême est évalué sur la bougie candidate `cand = n-2` (avant-dernière bougie), et la confirmation utilise la bougie `now = n-1`.

- En tranche `-` (LOW / open_side LONG):
  - `cand` doit être un **nouveau plus bas** (record) depuis le début de tranche
  - puis `now` doit confirmer en étant **>=** à `cand`
- En tranche `+` (HIGH / open_side SHORT):
  - `cand` doit être un **nouveau plus haut** (record) depuis le début de tranche
  - puis `now` doit confirmer en étant **<=** à `cand`

Si ces conditions sont remplies, la série est “confirmée maintenant” (`is_extreme_confirmed_now=True`) avec un `extreme_ts` égal au timestamp de `cand`.

#### `mode` (long/short/both)

`mode` filtre le type d’extrême autorisé sur la série `close`:

- `long` => exige `close_extreme_kind=LOW` (donc tranche `-`, open_side LONG)
- `short` => exige `close_extreme_kind=HIGH` (donc tranche `+`, open_side SHORT)
- `both` => pas de filtre; LONG ou SHORT possibles

#### `cci_thresholds` (filtre CCI)

Les seuils ne s’appliquent qu’aux colonnes CCI (`cci_<fast|medium|slow>`), si elles font partie de `series_cols`.

- LONG: exige qu’il existe **au moins une bougie dans la tranche courante** telle que `cci <= -threshold`
- SHORT: exige qu’il existe **au moins une bougie dans la tranche courante** telle que `cci >= +threshold`

Si `threshold` est `null` ou `<= 0`, le filtre est ignoré.

#### `trend_filter` (filtre tendance)

Le filtre tendance (si activé) est évalué sur la bougie `now = n-1`:

- `vortex`: compare `vi_plus` vs `vi_minus`
- `dmi`: compare `di_plus` vs `di_minus`
- `both`: exige l’accord des deux

Le trigger est bloqué si la tendance ne va pas dans le sens de `open_side`.

#### `confluence_type=instant`

Dans ce mode, le trigger “zone” est vrai si:

- `close` a un extrême confirmé **sur `cand`**
- `close_extreme_kind` respecte `mode` (si `mode != both`)
- chaque autre série confirmée:
  - confirme aussi **sur le même `cand_ts`**
  - a la même direction `open_side`
  - respecte le filtre CCI si applicable
- `confirmed_count >= min_confirmed` (ou toutes les séries si `min_confirmed=null`)
- `trend_filter` (si activé) autorise la direction

Remarque importante: ce mode est **synchrone sur la bougie candidate** (timestamp `cand_ts = ts[n-2]`) et **pas** sur “toute la tranche”.

 Exception: les colonnes CCI (si incluses dans `series_cols`) sont validées via `cci_thresholds` sur **toute la tranche courante** (hit du seuil au moins une fois), et ne sont donc pas contraintes à confirmer sur `cand_ts`.

Remarque importante: ce mode produit un trigger au moment où l’extrême de `close` est confirmé (puis confluence sur les autres séries au même instant).

#### `confluence_type=tranche_last`

Dans ce mode, on travaille sur toute la tranche en cours.

Pour chaque série, on cherche la première position dans la tranche où elle “confirme” (record + bougie suivante qui confirme), puis:

- on trie ces positions de confirmation
- le trigger se déclenche exactement lorsque la `min_confirmed`-ième série (le N-ième élément) devient confirmée

Pour les colonnes CCI (si incluses dans `series_cols`), le filtre `cci_thresholds` est évalué sur **toute la tranche courante** (hit du seuil au moins une fois). La “position” retenue pour l’ordre de confluence est la première bougie de la tranche où le seuil est atteint.

Autrement dit, le trigger est vrai si:

- au moins `min_confirmed` séries ont une confirmation dans la tranche
- la bougie courante est précisément celle où la N-ième confirmation se produit
- `trend_filter` (si activé) autorise la direction

Conséquence pratique: `tranche_last` tend à être plus “rare” mais vise à attendre une confluence progressive dans la tranche.

## État "armed"

- Tant qu’aucune position n’est ouverte, un trigger valide (zone + direction LONG/SHORT) arme la stratégie.
- Une fois armée, la stratégie n’entre pas immédiatement.
- L’entrée se produit seulement si les filtres `signals.entry` valident la direction armée.

Dans l’implémentation actuelle, l’état armé persiste tant qu’une entrée n’a pas été exécutée.

### Filtre tranche "tendance / saine" (contrarien)

Optionnel (entrée): `signals.entry.params.price_action.tranche_hist_trend_mode`

Objectif: filtrer les triggers en ne gardant que ceux dont la tranche est dans une **tendance MACD cohérente**, puis prendre un trade **contrarien** à cette tendance.

Définition (sur les bougies de la tranche):

- On regarde les signes de `macd_line`, `macd_signal`, `macd_hist`.
- Une bougie est “alignée” si `sign(macd_line) == sign(macd_signal) == sign(macd_hist)` et que le signe est non nul.

Modes:

- `none`: pas de filtre.
- `trend`: accepte la tranche si au moins une bougie est alignée.
- `healthy`: accepte la tranche seulement si toutes les bougies de la tranche sont alignées.

Direction:

- `tranche_hist_trend_side` est déduite de l’alignement (LONG si signe +, SHORT si signe -).
- La stratégie prend une position **contrarienne**: `open_side` doit être l’opposé de `tranche_hist_trend_side`.

### Workflow entrée / sortie (résumé)

#### Workflow entrée

- À `close(i)`:
  - Calcul du **trigger** (tranche/extreme/confluence).
  - Si trigger valide => `armed_side` devient `LONG` ou `SHORT`.
  - Si `armed_side` défini =>
    - si `signals.entry.params.price_action.entry_mode=simple`: évaluation directe des filtres `signals.entry`.
    - si `entry_mode=vwma_break`: workflow séquentiel ci-dessous.
  - Si **tous** les critères de l’entry_mode actif sont `True` => entrée planifiée.
- À `open(i+1)`:
  - Exécution de l’entrée.

#### Entry mode `vwma_break` (workflow)

Paramètres:

- `entry_mode: vwma_break`
- `vwma_break_max_bars: Y` (entier)

Workflow:

- Le trigger arme une direction `armed_side`.
- Ensuite, on attend une **cassure VWMA fast** sur clôture:
  - LONG: `close[t-1] <= vwma_fast[t-1]` et `close[t] > vwma_fast[t]`
  - SHORT: `close[t-1] >= vwma_fast[t-1]` et `close[t] < vwma_fast[t]`
- Quand la cassure est détectée:
  - elle démarre une fenêtre de validation de longueur max `Y` bougies.
  - on peut valider immédiatement sur la bougie de cassure si les filtres le permettent.
- Pendant la fenêtre (tant que `close` reste du bon côté de VWMA fast), on valide avec les filtres `signals.entry` (ex: pente VWMA, MACD accel, couleur bougie…)
- Si une **cassure inverse** se produit avant validation, on **reset** l’état “cassure” et on ré-attend une nouvelle cassure (le trigger reste armé).
- Si la fenêtre dépasse `Y` bougies sans validation, on reset la cassure (le trigger reste armé).

#### Workflow sortie

- À chaque bougie:
  - TP/SL (si activés) peuvent sortir **intrabar**.
- À `close(i)` si `exit_policy.allow_exit_signal=true`:
  - Évaluation des filtres `signals.exit`.
  - Important: pour les filtres directionnels VWMA/Stoch, le moteur utilise la **direction inverse** de la position (ex: sortir un LONG => confirmation "bear").
  - Si **tous** les filtres activés sont `True` => sortie planifiée.
- À `open(i+1)`:
  - Exécution de la sortie.

## Filtres "price action"

Les filtres sont paramétrés séparément pour:

- `signals.entry.params.price_action`
- `signals.exit.params.price_action`

### Mécanisme add/exclude

Chaque phase a une liste de filtres par défaut, puis:

- `filters.add`: ajoute des filtres
- `filters.exclude`: retire des filtres

Filtres disponibles (noms exacts):

- `vwma_fast_confirm`
- `vwma_medium_confirm`
- `vwma_fast_slope`
- `stoch_cross`
- `macd_hist_slope`
- `macd_hist_sign` (entry uniquement)
- `candle_color`
- `macd_hist_sign_change` (entry/exit)

### VWMA slope

Filtre: `vwma_fast_slope`

- LONG: `vwma_fast[t] - vwma_fast[t-1] > 0`
- SHORT: `vwma_fast[t] - vwma_fast[t-1] < 0`

Pour la sortie, la direction est inversée comme pour les autres filtres directionnels.

### Bougie alignée (couleur)

Filtre: `candle_color`

- LONG: `close > open` (bougie haussière)
- SHORT: `close < open` (bougie baissière)

Pour la sortie, la direction est inversée.

### VWMA confirmation

- `vwma_confirm_bars: N` (entier > 0)

Règle:

- Entry LONG: `close > vwma_fast` sur les `N` dernières bougies
- Entry SHORT: `close < vwma_fast` sur les `N` dernières bougies

Pour la sortie, la direction est inversée:

- Sortie d’un LONG: conditions "bear" (comme un SHORT entry)
- Sortie d’un SHORT: conditions "bull" (comme un LONG entry)

`vwma_medium_confirm` est identique mais utilise `vwma_medium`.

### Stochastic cross

Colonnes requises:

- `stoch_k`, `stoch_d`

Règle:

- Entry LONG: `stoch_k > stoch_d`
- Entry SHORT: `stoch_k < stoch_d`

Pour la sortie, la direction est inversée.

### MACD histogram slope

Colonne requise:

- `macd_hist`

Paramètre:

- `macd_hist_slope_mode`
  - `delta`: dérivée 1
  - `accel`: dérivée 2

Paramètre additionnel (si `macd_hist_slope_mode=accel`):

- `macd_hist_accel_mode`
  - `diff2`: dd = (hist[t]-hist[t-1]) - (hist[t-1]-hist[t-2])
  - `mono`: hist monotone sur 3 points (LONG: `hist[t] > hist[t-1] > hist[t-2]`, SHORT inverse)

Règles:

- Mode `delta`
  - Entry LONG: `hist[t] - hist[t-1] > 0`
  - Entry SHORT: `hist[t] - hist[t-1] < 0`
  - Exit LONG: `hist[t] - hist[t-1] < 0`
  - Exit SHORT: `hist[t] - hist[t-1] > 0`

- Mode `accel`
  - Compare la variation des deltas entre `t-2`, `t-1`, `t`.
  - Entry LONG: accélération positive
  - Entry SHORT: accélération négative
  - Exit LONG: décélération (accélération négative)
  - Exit SHORT: décélération (accélération positive)

### MACD histogram sign (entry)

Filtre: `macd_hist_sign`

- Entry LONG: `macd_hist > 0`
- Entry SHORT: `macd_hist < 0`

### MACD histogram sign change (entry/exit)

Filtre: `macd_hist_sign_change`

Paramètre:

- `exit_hist_sign_change_mode`
  - `cross`: vrai cross de signe entre `t-1` et `t`
  - `sign`: seulement le signe final (plus permissif)

Note: malgré le nom, `exit_hist_sign_change_mode` est utilisé aussi pour l’**entry** si tu actives `macd_hist_sign_change` côté entry.

Règles:

#### Conditions entry

- Entry LONG:
  - `cross`: `hist[t-1] <= 0` et `hist[t] > 0`
  - `sign`: `hist[t] > 0`
- Entry SHORT:
  - `cross`: `hist[t-1] >= 0` et `hist[t] < 0`
  - `sign`: `hist[t] < 0`

#### Conditions exit

- Sortie LONG:
  - `cross`: `hist[t-1] >= 0` et `hist[t] < 0`
  - `sign`: `hist[t] < 0`
- Sortie SHORT:
  - `cross`: `hist[t-1] <= 0` et `hist[t] > 0`
  - `sign`: `hist[t] > 0`

Remarque: si tu actives à la fois `macd_hist_sign` et `macd_hist_sign_change` sur entry, les deux doivent passer (donc `macd_hist_sign_change` rend souvent `macd_hist_sign` redondant).

## Money management / Risk

### Fees

- `backtest.fee_rate` est un fee par côté.
- Le PnL net soustrait `2 * fee_rate`.

### Exit policy

- `exit_policy.allow_exit_signal`
  - `true`: sortie sur signal `signals.exit` autorisée
  - `false`: sorties uniquement via TP/SL (ou EOD)

### TP

- `tp.mode`
  - `none`
  - `fixed_pct`
  - `pivot_grid`
- `tp.tp_pct`
  - requis si `fixed_pct`
  - optionnel si `pivot_grid` (si défini, le TP est pris comme le premier pivot au-delà de la cible `%`)

Si `tp.mode=pivot_grid`, le moteur utilise la table de pivots multi‑TF pondérée (confluence) et maintient un cache runtime :

- la table est rafraîchie uniquement si nécessaire (sinon réutilisation de la table précédente)
- TP = **premier pivot** du bon côté de la cible `%TP`
  - LONG: pivot `>= entry_price * (1 + tp_pct)`
  - SHORT: pivot `<= entry_price * (1 - tp_pct)`

Configuration: `pivot_grid.*`

- `pivot_grid.enabled`: doit être `true`
- `pivot_grid.symbol`
- `pivot_grid.registries.{5m,1h,4h}`: chemins vers les JSON `PivotRegistry`
- `pivot_grid.mode`: `grid` ou `zones`
- `pivot_grid.grid_pct`: requis si `mode=grid`
- `pivot_grid.zones_cfg`: requis si `mode=zones` (mêmes paramètres que la démo)
- `pivot_grid.min_supports` / `pivot_grid.min_resistances`: règle de réutilisation du cache

### SL

- `sl.mode`
  - `none`
  - `fixed_pct`
  - `trailing_pct`
  - `atr`

Paramètres:

- `fixed_pct`: `sl.sl_pct`
- `trailing_pct`: `sl.trail_pct`
- `atr`: `sl.atr_mult` (+ optionnel `sl.atr_len`)

## Lancer le backtest

Avec le venv du projet:

```bash
/root/projects/trading_space/windsurf_space/harmonie_60_space/agent_economique_stable_py/venv_optuna/bin/python3 \
  scripts/12_backtest_price_action_tranche_yaml.py \
  --config configs/backtest_price_action_tranche_example.yaml \
  --out-dir data/processed/backtests/_run_price_action_tranche
```

## Outputs

Le runner écrit dans `output.out_dir` (ou `--out-dir`):

- `trades.csv`
- `equity.csv`
- `equity.png` (si `output.png: true` et si matplotlib est disponible)

Champs importants dans `trades.csv`:

- `entry_ts`, `exit_ts`, `entry_price`, `exit_price`, `exit_reason`
- `gross_ret`, `net_ret`
- `mfe`, `mae`
