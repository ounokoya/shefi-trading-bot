# zereme-session-asi-only-exit-klinger-v1

## Identité du bot

- **Horizon temporel couvert (données d’optimisation)**: année 2025
- **Timeframe d’exécution**: `5m`
- **Série(s) de signal**: `asi` uniquement
- **Méthode de sortie**: `klinger` (cross `kvo` vs `klinger_signal`)
- **Version**: `v1`

Source de sélection: export Optuna top10 (ratio) – **trial 939**.

## Résumé performance (selon export trial 939)

- **objective_mode**: `annual`
- **objective (ratio)**: `203.91760250471663`
- **equity**: `6745.855521728981`
- **max_dd**: `-42.54797635051358`
- **worst_mae**: `-27.060471975370614`
- **n_trades**: `6982`
- **winrate**: `87.12403322830134`

> Notes importantes:
> - Ce bot est “**asi-only**” car `max_combo_size=1` dans le trial exporté ⇒ une seule série est réellement utilisée.
> - Les colonnes `p:series_combo_k2` / `p:series_combo_k3` dans le CSV ne changent pas le comportement si `max_combo_size=1`.

---


# Spécification algorithmique (language-agnostic)

L’objectif de cette section est que tu puisses implémenter le bot dans **n’importe quel langage**.

## Entrées

- Flux de bougies OHLCV à intervalle constant (ici `5m`):
  - `timestamp` (UTC)
  - `open`, `high`, `low`, `close`, `volume`

## Indicateurs requis

- **CCI** sur OHLC (période dépendante du timeframe; pour `5m` utiliser `period = 96`).
- **ASI** (Accumulative Swing Index) calculé sur OHLC.
- **MACD** sur `close` avec paramètres `fast=12`, `slow=26`, `signal=9`:
  - `macd_line`, `macd_signal`, `macd_hist = macd_line - macd_signal`
- **Klinger Oscillator** (KVO) + ligne signal:
  - `kvo`, `klinger_signal` avec `fast=34`, `slow=55`, `signal=13`

L’implémentation exacte de ces indicateurs est **hors scope de ce document** (elle existe déjà dans la doc indicateurs). Ici, on ne fait que spécifier comment le bot consomme ces séries.

## États internes (mémoire)

- **Zones CCI**: une zone est active quand CCI est en extrême.
  - `zone_type ∈ {creux, sommet}`
  - `zone_start_index`
  - `zone_id` (incrémenté à chaque nouvelle zone)

- **Extremums confirmés par série** (ici 2 séries sont suivies):
  - Série de signal: `asi`
  - Série de stop/référence: `price` (le `close`)
  - Pour chaque série `s` et pour chaque type (`creux`/`sommet`), on mémorise le dernier extrême confirmé: `last_confirmed[s][type]`.

- **Extremum potentiel courant dans la zone** (par série):
  - pour un `creux`: on cherche le minimum dans la zone
  - pour un `sommet`: on cherche le maximum dans la zone

- **Signal en attente**:
  - `pending_bull` / `pending_bear` quand une structure est détectée:
    - pendant la zone CCI (évaluation continue), ou
    - à la sortie de zone CCI (confirmation de fin de zone).

- **Armement de la sortie**:
  - `exit_arm_index` (initialement vide)
  - la sortie B (klinger) n’est autorisée qu’après armement (voir plus bas).

## Paramètres (valeurs de ce bot)

- `cci_extreme_abs = 20.0`
- `cci_low = -20.0`
- `cci_high = +20.0`
- `extreme_confirm_bars = 0`
- `entry_require_hist_abs_growth = false`
- `entry_cci_tf_confluence = false`
- `use_fixed_stop = true`
- `stop_buffer_pct = 0.01`
- `exit_b_mode = klinger`
- `trade_direction = both`

## Contrat d’entrée/sortie (ce que ton implémentation doit produire)

### Entrée (données)

À chaque bougie `i`, ton moteur doit disposer au minimum de:

- OHLCV: `open[i], high[i], low[i], close[i], volume[i]`
- Indicateurs: `cci[i], asi[i], macd_hist[i], kvo[i], klinger_signal[i]`

### Sortie (trades)

Le bot doit produire une liste de trades, chaque trade contenant au minimum:

- `side` ∈ {`LONG`, `SHORT`}
- `entry_index`, `entry_time`, `entry_price`
- `exit_index`, `exit_time`, `exit_price`
- `exit_reason` ∈ {`stop`, `exit_klinger`, `eod`}
- (optionnel) `stop_price`

### Hypothèses d’exécution (backtest / production)

- **Une seule position à la fois** (pas de pyramiding).
- **Entrée au prix de clôture** de la bougie `i` (ou ton équivalent “market at close”).
- **Stop**: déclenché intrabar via `low/high`:
  - LONG: si `low[i] <= stop_price` ⇒ sortie au `stop_price`
  - SHORT: si `high[i] >= stop_price` ⇒ sortie au `stop_price`
- **Sortie klinger**: déclenchée sur le close de la bougie `i` quand la condition de cross est vraie (après armement).

Initialisation / warmup:

- Ne pas autoriser d’ordres tant que les indicateurs ne sont pas “valides” (ex: CCI/MACD/Klinger non calculables faute d’historique).
- Minimum recommandé avant trading:
  - `>= 3 * CCI_period` bougies pour stabiliser CCI (ici `>= 288` bougies)
  - `>= max(MACD_slow + MACD_signal, Klinger_slow + Klinger_signal)` bougies (ici ordre de grandeur `>= 68`)

Ces conventions doivent rester **cohérentes** entre ton backtest et ta prod.

---

# Signaux de trading (résumé opérationnel)

Cette section résume **exactement** ce qui déclenche une **ouverture** et une **fermeture**.

## Signal d’ouverture (entrée en position)

Pré-requis généraux:

- Pas de position ouverte.
- Indicateurs valides (warmup terminé).
- Un **signal en attente** existe (`pending_bull` ou `pending_bear`).

Note importante:

- L’entrée n’est **jamais rétroactive**: on n’ouvre pas “au point extrême”.
- Le point extrême (dans la zone CCI) sert uniquement à:
  - valider la structure (higher-low / lower-high) et créer `pending_*`
  - calculer le stop fixe (référence sur un extrême `price` confirmé précédent)
  - puis l’entrée se fait sur la bougie courante quand le gating MACD est vrai.

Note sur l’implémentation (important):

- Il y a **2 chemins** de création de `pending_*`:
  - **Chemin A (pendant la zone)**: la structure est évaluée en continu, et `pending_*` peut être créé/mis à jour tant que la zone CCI est active.
  - **Chemin B (à la sortie de zone)**: si la zone se termine, le code confirme les extrêmes de zone et peut créer `pending_*` à ce moment.
- `pending_*` peut aussi être **effacé** pendant la zone si la structure n’est plus satisfaite au bar courant.
- Donc l’ouverture peut arriver **avant** OU **après** la sortie de zone, selon quand le gating MACD devient vrai.

Ordre exact d’évaluation au bar `i` (non ambigu):

- Si une position est ouverte:
  - le STOP est évalué en premier (si `stop_price` existe).
  - ensuite la sortie B est évaluée (si `exit_arm_index` existe et `i > exit_arm_index`).
- Ensuite, la logique de zones CCI est évaluée:
  - d’abord les transitions de zone (démarrage de zone si `not in_zone`, ou fin de zone si `in_zone` et CCI sort de l’extrême).
  - si une fin de zone est détectée au bar `i`, le code peut créer `pending_*` dans le chemin “sortie de zone”, puis la zone est clôturée (`in_zone=false`).
  - dans ce cas précis, le chemin “pendant zone” n’est pas évalué au bar `i` (car il nécessite `in_zone=true`).
- Ensuite, les crosses MACD hist sont mis à jour.
- Enfin, si `pending_*` existe et que le gating MACD est vrai, l’entrée peut être déclenchée au bar `i`.

Détails exacts des différences entre les 2 chemins (tel que codé):

- **Filtre confluence CCI multi-TF** (si activé):
  - Chemin A (pendant zone) vérifie la confluence au bar `i`.
  - Chemin B (sortie de zone) vérifie la confluence au bar `i-1` (dernier bar encore “dans la zone”).
  - Dans ce bot: `entry_cci_tf_confluence = false` ⇒ le filtre est inactif.
- **Confirmation N-bar (`extreme_confirm_bars`)**:
  - Chemin A: si `extreme_confirm_bars > 0`, la structure ne peut devenir vraie qu’après que l’extrême potentiel ait “vieilli” d’au moins `N` bougies (sinon l’extrême est ignoré).
  - Chemin B: à la sortie de zone, le code confirme l’extrême courant de la zone et peut déclencher une structure **même si `N>0`** (donc la sortie de zone contourne la contrainte N-bar).
- **Effacement**:
  - Chemin A: si à un bar donné `n_ok < min_confluence` (ou confluence multi-TF échoue), alors `pending_*` est remis à `null`.
  - Chemin B: si la sortie de zone n’atteint pas `min_confluence` (ou confluence multi-TF échoue), alors il ne crée pas de `pending_*` (et il n’en recrée pas à ce moment).
- **trade_direction**:
  - Si `trade_direction = long`: quand un signal bear est prêt, le code efface `pending_bear` sans ouvrir (anti-short).
  - Si `trade_direction = short`: quand un signal bull est prêt, le code n’ouvre pas, et (nuance) il ne force pas systématiquement l’effacement de `pending_bull`.
  - Si `trade_direction = both`: pas de restriction.

### Ouverture LONG (entrée acheteuse)

1. **Création de `pending_bull`** (2 possibilités):
   - **A (pendant zone creux)**: si la zone est active (`cci[i] < cci_low`), et que la structure ASI est vraie au bar `i`, alors `pending_bull` est créé/mis à jour.
   - **B (sortie de zone creux)**: quand la zone creux se termine (CCI repasse `>= cci_low`), le code confirme l’extrême de zone et peut créer `pending_bull`.

2. **Timing MACD hist (gating)**:
   - on mémorise `last_bull_hist_cross_index` quand `macd_hist` croise de `<=0` vers `>0`.
   - l’entrée LONG est autorisée si:
     - `pending_bull != null`
     - ET `macd_hist[i] > 0`
     - ET `last_bull_hist_cross_index >= es`
       - où `es = pending_bull.extreme_start_i` si présent, sinon `pending_bull.created_i`.

3. **Stop fixe (si disponible)**:
   - si `use_fixed_stop=true` et si un extrême `price` confirmé précédent existe, alors:
     - `stop_price = prev_confirmed_price_creux * (1 - stop_buffer_pct)`
   - sinon `stop_price = null` (le code actuel autorise l’entrée même si le stop n’est pas calculable).

4. **Entrée**:
   - ouvrir LONG au prix de clôture `close[i]`
   - stocker `stop_price`
   - effacer `pending_bull`

### Ouverture SHORT (entrée vendeuse)

Symétrique:

1. **Création de `pending_bear`** (2 possibilités):
   - **A (pendant zone sommet)**: si la zone est active (`cci[i] > cci_high`), et que la structure ASI est vraie au bar `i`, alors `pending_bear` est créé/mis à jour.
   - **B (sortie de zone sommet)**: quand la zone sommet se termine (CCI repasse `<= cci_high`), le code confirme l’extrême de zone et peut créer `pending_bear`.

2. **Timing MACD hist (gating)**:
   - on mémorise `last_bear_hist_cross_index` quand `macd_hist` croise de `>=0` vers `<0`.
   - l’entrée SHORT est autorisée si:
     - `pending_bear != null`
     - ET `macd_hist[i] < 0`
     - ET `last_bear_hist_cross_index >= es`
       - où `es = pending_bear.extreme_start_i` si présent, sinon `pending_bear.created_i`.

3. **Stop fixe (si disponible)**:
   - si `use_fixed_stop=true` et si un extrême `price` confirmé précédent existe, alors:
     - `stop_price = prev_confirmed_price_sommet * (1 + stop_buffer_pct)`
   - sinon `stop_price = null`.

4. **Entrée**:
   - ouvrir SHORT au prix de clôture `close[i]`
   - stocker `stop_price`
   - effacer `pending_bear`

## Signal de fermeture (sortie de position)

Il y a 2 familles de sorties: **STOP** (prioritaire) et **exit_klinger** (sortie “B”, seulement après armement).

### Fermeture par STOP (prioritaire)

À chaque bougie, si une position est ouverte et que `stop_price` existe:

- LONG:
  - si `low[i] <= stop_price` ⇒ fermer au `stop_price` (raison `stop`)
- SHORT:
  - si `high[i] >= stop_price` ⇒ fermer au `stop_price` (raison `stop`)

### Fermeture par Klinger (exit_klinger)

La sortie Klinger est **verrouillée** tant qu’on n’a pas armé la sortie.

1. **Armement (exit_arm_index)**:
   - LONG: armer quand un nouvel extrême `price` de type `sommet` est détecté **après l’entrée**
   - SHORT: armer quand un nouvel extrême `price` de type `creux` est détecté **après l’entrée**
   - avec `extreme_confirm_bars=0`, l’armement arrive dès que l’extrême est observé.
2. **Condition de cross Klinger** (uniquement si `exit_arm_index` est défini et `i > exit_arm_index`):
   - LONG: sortir si `kvo` croise **sous** `klinger_signal`
     - `kvo[i-1] >= klinger_signal[i-1]` ET `kvo[i] < klinger_signal[i]`
   - SHORT: sortir si `kvo` croise **au-dessus** `klinger_signal`
     - `kvo[i-1] <= klinger_signal[i-1]` ET `kvo[i] > klinger_signal[i]`
3. **Exécution de la sortie**:
   - fermer au prix de clôture `close[i]` (raison `exit_klinger`)

### Fermeture fin d'historique (eod)

Si une position est encore ouverte à la dernière bougie, le code ferme à `close[last]` avec la raison `eod`.

## Définition “structure” (ASI)

On travaille sur la série `asi`.

- En zone **creux** (CCI < `cci_low`), on veut détecter un **higher low** sur ASI:
  - `asi_creux_confirmé_actuel > asi_creux_confirmé_précédent`
- En zone **sommet** (CCI > `cci_high`), on veut détecter un **lower high** sur ASI:
  - `asi_sommet_confirmé_actuel < asi_sommet_confirmé_précédent`

Si la structure est vraie au moment où la zone se termine, on génère un signal en attente:
- fin de zone `creux` ⇒ `pending_bull`
- fin de zone `sommet` ⇒ `pending_bear`

La structure peut aussi être évaluée et déclencher un `pending_*` pendant qu’une zone est encore active, si la condition de structure est satisfaite sur un extrême potentiel (ou “ready” si `extreme_confirm_bars>0`).

Remarque: la stratégie est “asi-only” car `min_confluence=1` et la seule série structurale utilisée est `asi`.

## Gating par MACD histogram

On n’entre en position que si le signal en attente est validé par le **timing MACD hist**:

- Définition de cross:
  - bull cross: `macd_hist` passe de `<= 0` à `> 0`
  - bear cross: `macd_hist` passe de `>= 0` à `< 0`

Le cross doit se produire **après** le début de l’extrême (index `zone_start_index`), et l’entrée se fait quand:
- `pending_bull` existe ET `macd_hist > 0` ET (dernier bull cross index `>= zone_start_index`)
- `pending_bear` existe ET `macd_hist < 0` ET (dernier bear cross index `>= zone_start_index`)

`entry_require_hist_abs_growth=false` signifie qu’on **n’exige pas** que `|macd_hist|` augmente d’une bougie à l’autre.

## Stop fixe (basé sur le dernier extrême price confirmé précédent)

Le stop est calculé à l’entrée, à partir du **dernier extrême confirmé** de la série `price` (close) du même type que la zone de signal:

- Si entrée LONG (signal issu d’un `creux`):
  - référence = dernier `price creux` confirmé précédent
  - `stop = ref * (1 - stop_buffer_pct)`

- Si entrée SHORT (signal issu d’un `sommet`):
  - référence = dernier `price sommet` confirmé précédent
  - `stop = ref * (1 + stop_buffer_pct)`

Si la référence n’existe pas (ex: tout début d’historique), le code actuel calcule `stop_price = null` et peut quand même ouvrir une position (donc: **stop désactivé** pour ce trade).

## Armement de la sortie

La sortie “B” (klinger) n’est pas activée immédiatement.
Elle est **armée** quand un nouvel extrême **price** est détecté **après l’entrée**:

- position LONG: armement lors d’un nouvel extrême `sommet` sur `price` (dans une zone sommet)
- position SHORT: armement lors d’un nouvel extrême `creux` sur `price` (dans une zone creux)

Avec `extreme_confirm_bars=0`, l’armement se fait dès que l’extrême est observé (pas d’attente N bougies).

## Sortie (exit_b_mode = klinger)

Une fois `exit_arm_index` défini (armé), on peut sortir sur le **cross Klinger**:

- LONG: sortir si `kvo` croise **sous** `klinger_signal`
  - condition: `kvo[i-1] >= klinger_signal[i-1]` ET `kvo[i] < klinger_signal[i]`
- SHORT: sortir si `kvo` croise **au-dessus** `klinger_signal`
  - condition: `kvo[i-1] <= klinger_signal[i-1]` ET `kvo[i] > klinger_signal[i]`

---

# Détails des composants (spécification plus stricte)

## 1) Détection des zones CCI

On définit 2 états: `NOT_IN_ZONE` et `IN_ZONE`.

- Entrée en zone creux:
  - condition: `cci[i] < cci_low` ET (`cci[i-1] >= cci_low` ou `i==0` ou `cci[i-1] non valide`)
- Entrée en zone sommet:
  - condition: `cci[i] > cci_high` ET (`cci[i-1] <= cci_high` ou `i==0` ou `cci[i-1] non valide`)

Sortie de zone:
- fin d’une zone `creux` quand `cci[i] >= cci_low`
- fin d’une zone `sommet` quand `cci[i] <= cci_high`

Note d’ordre d’évaluation:

- Au bar où une fin de zone est détectée, le code évalue le chemin “sortie de zone” et clôture la zone.
- Par conséquent, le chemin “pendant zone” n’est pas évalué sur ce même bar (il ne s’exécute que si `in_zone` reste vrai).

À l’entrée dans une zone, on initialise:
- `zone_start_index = i`
- `zone_id += 1`
- les extrêmes potentiels (pour `asi` et `price`)

## 2) Extrêmes “potentiels” et “confirmés”

On suit 2 séries:
- `price` (référence stop et armement)
- `asi` (structure)

Pour chaque zone:
- en `creux`:
  - extrême potentiel = **minimum** observé dans la zone
- en `sommet`:
  - extrême potentiel = **maximum** observé dans la zone

Confirmation / “ready” (non ambigu):

- À la fin de la zone, le code confirme l’extrême courant de la zone (le dernier `potential` connu) et met à jour `last_confirmed[...]`.
  - Cela arrive même si `extreme_confirm_bars > 0`.
- Pendant une zone, si `extreme_confirm_bars > 0`, un extrême potentiel n’est considéré exploitable (“ready”) qu’à partir de la bougie `i` telle que `i >= potential.idx + extreme_confirm_bars`.
  - Cette notion “ready” affecte:
    - la création/maintenance de `pending_*` pendant zone,
    - l’armement de la sortie.
  - Elle n’empêche pas la confirmation de fin de zone.

À la fin de la zone, on met à jour:
- `last_confirmed[asi][zone_type]`
- `last_confirmed[price][zone_type]`

## 3) Calcul de la structure ASI

Définition de la structure (série `asi`):

- Zone `creux`: structure bullish si `asi_extreme_actuel > last_confirmed_asi_creux_précédent`.
- Zone `sommet`: structure bearish si `asi_extreme_actuel < last_confirmed_asi_sommet_précédent`.

Deux moments possibles d’évaluation (les deux peuvent créer un `pending_*`, mais pas sur le même bar si la zone se termine):

- Pendant la zone:
  - `asi_extreme_actuel` est l’extrême potentiel de la zone (ou l’extrême “ready” si `extreme_confirm_bars>0`).
  - Si la structure est vraie et la confluence requise est satisfaite, le code peut créer/mettre à jour `pending_*`.
- À la fin de zone:
  - `asi_extreme_actuel` est l’extrême confirmé de fin de zone.
  - Si la structure est vraie et la confluence requise est satisfaite, le code peut créer `pending_*`.

## 4) Timing MACD hist (gating)

On mémorise:
- `last_bull_hist_cross_index`
- `last_bear_hist_cross_index`

Détection cross:
- bull cross si `macd_hist[i-1] <= 0` ET `macd_hist[i] > 0`
- bear cross si `macd_hist[i-1] >= 0` ET `macd_hist[i] < 0`

Condition d’entrée LONG:
- `pending_bull` existe
- `macd_hist[i] > 0`
- `last_bull_hist_cross_index >= pending_bull.zone_start_index`

Condition d’entrée SHORT:
- `pending_bear` existe
- `macd_hist[i] < 0`
- `last_bear_hist_cross_index >= pending_bear.zone_start_index`

Option désactivée dans ce bot:
- `entry_require_hist_abs_growth=false` ⇒ pas de condition `abs(macd_hist[i]) > abs(macd_hist[i-1])`.

## 5) Armement de la sortie

But: éviter une sortie “trop tôt” sans nouveau swing.

Une fois en position, on cherche un extrême `price` dans la direction opposée:
- LONG: armement sur un `price sommet` détecté après l’entrée
- SHORT: armement sur un `price creux` détecté après l’entrée

Quand la condition est vraie:
- `exit_arm_index = i` (ou l’index de confirmation, selon ton implémentation N-bar)

## 6) Sortie Klinger (après armement)

Cross Klinger:
- `cross_down(a,b)` vrai si `a[i-1] >= b[i-1]` et `a[i] < b[i]`
- `cross_up(a,b)` vrai si `a[i-1] <= b[i-1]` et `a[i] > b[i]`

Sortie:
- LONG: si `exit_arm_index` défini ET `i > exit_arm_index` ET `cross_down(kvo, klinger_signal)`
- SHORT: si `exit_arm_index` défini ET `i > exit_arm_index` ET `cross_up(kvo, klinger_signal)`

---

# Configuration (schéma language-neutral)

Cette section définit **ce que tu dois exposer comme paramètres** dans ton bot, quel que soit le langage.

## Paramètres de marché

- `symbol` (string)
- `timeframe` (string) = `5m`

## Paramètres de stratégie (ce bot)

- `cci_extreme_abs` (float) = `20.0`
- `cci_low` (float) = `-20.0`
- `cci_high` (float) = `+20.0`
- `extreme_confirm_bars` (int) = `0`
- `entry_require_hist_abs_growth` (bool) = `false`
- `entry_cci_tf_confluence` (bool) = `false`
- `use_fixed_stop` (bool) = `true`
- `stop_buffer_pct` (float) = `0.01`
- `exit_b_mode` (enum) = `klinger`
- `trade_direction` (enum) = `both` | `long` | `short`

## Paramètres indicateurs (à transmettre à ta lib indicateurs)

- CCI: `period = 96` (pour TF 5m)
- MACD: `fast = 12`, `slow = 26`, `signal = 9`
- Klinger: `fast = 34`, `slow = 55`, `signal = 13`

---

# Points de vigilance (avant mise en prod)

- **Sur-optimisation**: bot sélectionné sur un historique (ici 2025). Tester sur une autre fenêtre avant déploiement réel.
- **Stop fixed**: stop calculé depuis un extrême `price` confirmé précédent (pas depuis l’entrée). Donc worst MAE peut dépasser `stop_buffer_pct`.
- **Frais/slippage**: fréquence de trade élevée sur `5m` ⇒ intégrer fees + slippage dans le backtest du langage cible.

