# Stratégie (étude) : zones d’accumulation / distribution via Klinger + CCI + DMI

Ce document décrit une **méthode d’étude** (pas un signal d’entrée/sortie automatique) pour détecter des **zones d’accumulation / distribution** à partir de l’alignement chronologique :

- **Klinger Oscillator** : `kvo` + `kvo_signal`
- **CCI confluence** : 2/2 (fast+medium) ou 3/3 (fast+medium+slow)
- **DMI** : filtrage sur la catégorie **`impulsion`** au moment de l’extrême Klinger

L’objectif est de répondre à la question : **“À quel moment peut-on commencer à accumuler ?”** en phase short (CCI extrêmes négatifs), sans se baser directement sur le prix.

---

## 1) Briques logiques utilisées

### 1.1 Tranches Klinger
On segmente la série en **tranches** entre deux croisements de `kvo_diff` :

- `kvo_diff = kvo - kvo_signal`
- un changement de signe de `kvo_diff` délimite une tranche

Pour chaque tranche :
- on détermine `side` :
  - `LONG` si `kvo_diff` est positif après le début de tranche
  - `SHORT` si `kvo_diff` est négatif
- on détecte l’**extrême Klinger** de tranche :
  - `SHORT` : minimum de `kvo`
  - `LONG` : maximum de `kvo`

### 1.2 DMI au moment de l’extrême
On calcule DMI/ADX/DX par bougie et on classe la tranche à **l’instant de l’extrême Klinger** :

- **`sans_force`** si ADX au moment de l’extrême est sous un seuil
- sinon :
  - **`respiration`** si `DX <= ADX`
  - **`impulsion`** si `DX > ADX`

Dans cette étude “zone”, on se concentre sur :
- `dmi_category = impulsion`

### 1.3 Confluence CCI (2/2)
On calcule deux CCI (par défaut dans l’étude actuelle) :

- fast = **CCI(30)**
- medium = **CCI(90)**

Confluence 2/2 sur une tranche `SHORT` :
- `CCI(30) <= -level` **ET** `CCI(90) <= -level` (au moins une fois dans la tranche)

Niveau par défaut : `level = 100`.

---

## 2) Définition d’un “événement” étudié

Un **événement** `E` est une tranche qui vérifie :

- `side = SHORT`
- `cci_confluence_ok = True` (confluence 2/2)
- `dmi_category = impulsion` (au moment de l’extrême Klinger)

Chaque événement a :
- `E.kvo_ext` = `kvo` à l’extrême de tranche
- `E.sig_ext` = `kvo_signal` au même instant
- `E.dt` = date de l’extrême

### 2.1 Définition de “consécutif”
Dans l’étude, **consécutif** signifie :

- `E2` est le **prochain événement** après `E1` qui vérifie **le même filtre**
  - (les tranches inverses, ou non impulsion, ou sans confluence ne sont pas comptées)

---

## 3) Pattern ACCUM (déclencheur de “zone d’accumulation”)

Entre deux événements consécutifs `E1 -> E2` (toujours `SHORT + confluence + impulsion`) :

- **ACCUM** si :
  - `E2.kvo_ext > E1.kvo_ext` (KVO remonte, donc *moins négatif*)
  - **ET** `E2.sig_ext > E1.sig_ext` (signal remonte)

Interprétation :
- malgré des CCI toujours en extrêmes négatifs, le couple `kvo/kvo_signal` cesse de s’aggraver → **accumulation**

### 3.1 Moment pour “commencer à accumuler”
Dans cette règle, le point opérationnel est :

- **Start accumulation = date de `E2`** (le 2e événement du pattern ACCUM)

---

## 4) Validation de la zone : structure en “V” (sur Klinger)

### 4.1 V-bottom (minimum sur la période de formation)
On définit le bottom du V sur toute la formation `E1 -> E2` :

- `Vbottom_kvo = min(kvo)` sur la période indexée `[E1_pos..E2_pos]`
- `Vbottom_sig = min(kvo_signal)` sur la même période

### 4.2 Condition “ne pas casser le V par le bas”
On utilise le V-bottom comme **référence structurelle** pour savoir si une campagne d’accumulation reste “valide”.

La logique actuelle ne s’appuie plus sur une fenêtre fixe `30/60/90j`.

À la place, on raisonne en **événements** :
- tant que Klinger ne casse pas (selon la règle choisie) le bottom de référence, la zone continue d’être observée
- l’invalidation devient un **événement de cassure** (voir section 6)

---

## 5) Validation événementielle post-formation (post_ok)

Après `E2`, on regarde les **2 événements suivants** `E3` et `E4` (mêmes filtres).

On invalide (post_ok = False) si :

- les extrêmes de la **CCI de référence** deviennent **moins négatifs** deux fois de suite :
  - `CCIref(E3) > CCIref(E2)` et `CCIref(E4) > CCIref(E3)`
- mais en même temps Klinger **rechute** sur ces deux événements :
  - `kvo_ext(E4) < kvo_ext(E3)`
  - `sig_ext(E4) < sig_ext(E3)`

Cela correspond à :
- “CCI s’affaiblit mais Klinger redégrade en 2 temps” → la zone est jugée fragile.

Dans les sorties :
- `accum_post_ok_2ev` est le ratio moyen de `post_ok` (uniquement quand on a assez d’événements après).

---

## 6) Validation “profit” et campagnes (nouvelle logique)

Cette étude est centrée sur la question opérationnelle :

- Une zone d’accumulation est-elle **capable de produire un profit minimal** (ex: `+2%`) avant que Klinger n’invalide la structure ?

### 6.1 Confirmation par % de prix (profit)
Après un start d’accumulation (date de `E2`), on mesure si le prix atteint un ou plusieurs seuils :

- `+2%`, `+5%`, `+10%`, `+15%`, `+20%`, `+25%`, `+30%`

Dans les sorties, ça correspond à :
- `accum_reached_2p`, `accum_reached_5p`, ...
- et les délais moyens : `accum_avg_days_to_2p`, ...

### 6.2 Campagne d’accumulation (fusion de plusieurs ACCUM)
Si plusieurs signaux `ACCUM` apparaissent avant d’avoir atteint le profit minimal (ex: `+2%`), on ne les compte pas comme des zones indépendantes.

On construit une **campagne** :
- début : 1er `ACCUM` (date de son `E2`)
- absorbtion : tout nouveau `ACCUM` avant confirmation/invalidation est absorbé
- confirmation : dès qu’un seul seuil (ex: `+2%`) est atteint, la campagne est **confirmée**

Dans les sorties :
- `campaigns_total`
- `campaigns_confirmed`
- `campaigns_no_confirm_retest` (campagnes qui n’atteignent jamais `+2%` avant invalidation)

### 6.3 Prix moyen (DCA) utilisé pour le %
Quand une campagne absorbe plusieurs signaux `ACCUM`, on calcule un **prix moyen** d’entrée :

- `entry_avg = moyenne des close(E2) absorbés` (poids égal par signal)

Les % (`+2%`, `+5%`, ...) sont évalués **par rapport à `entry_avg`**.

### 6.4 Invalidation décalée : bottoms successifs (max-bottom-breaks)
Une campagne peut “retester”/casser le bottom et **reconstruire** plus bas.

On paramètre le nombre de cassures autorisées :

- `--max-bottom-breaks = 0` : cassure du bottom => campagne terminée
- `--max-bottom-breaks = 1` (défaut) : on autorise 1 cassure, le bottom est mis à jour plus bas
- `--max-bottom-breaks = 2/3/...` : idem, avec plus de reconstructions

Une campagne n’est classée *sans confirmation* que si elle dépasse ce quota sans jamais atteindre le profit minimal.

---

## 7) Première version de “règle zone” (pas une stratégie)

Cette section formalise une règle **de zone d’accumulation** (détection + validation) sans définir une exécution complète.

### 7.1 Détection de zone
- détecter un pattern `ACCUM` sur deux événements consécutifs `E1 -> E2` (section 3)
- start de zone = `E2.dt`

### 7.2 Validation de zone (profit minimal)
- construire une **campagne** qui regroupe les signaux `ACCUM` qui s’alignent
- calculer `entry_avg` (prix moyen par signaux absorbés)
- confirmer la zone si le prix atteint au moins `+2%` (ou un autre seuil) avant invalidation

### 7.3 Invalidation de zone (structure)
- utiliser le V-bottom comme référence
- autoriser `N` reconstructions via `--max-bottom-breaks`
- au-delà de `N`, si le profit minimal n’a jamais été atteint, la campagne est classée *sans confirmation*

---

## 8) Implémentation et fichiers

### 6.1 Détecteur (logique principale)
- `libs/strategies/klinger_cci_extremes/detector.py`
  - calcule KVO/KVO signal, tranches, extrêmes
  - calcule CCI
  - calcule DMI et `dmi_category` **au moment de l’extrême**

### 6.2 Script d’étude / analyse
- `scripts/37_verify_klinger_divergence_link_2024_2025.py`

Fonctions clés ajoutées côté script :
- `_fetch_bybit_klines_range(...)` : pagination Bybit pour couvrir une période > 1000 bougies
- `_run_one_symbol(...)` : exécute l’étude sur un symbole, retourne des métriques + dates `ACCUM_START_POINTS`

Ajouts récents :
- cache CSV des klines : `--cache-dir`, `--force-refresh`
- campagnes + prix moyen + bottom shifts : `--max-bottom-breaks`
- logs : `--print-campaigns` (`none|failed|all`), `--print-start-points` (`no|yes`)

---

## 9) Utilisation (CLI)

### 7.1 Multi-actifs, période large (exemple)
```bash
venv_optuna/bin/python scripts/37_verify_klinger_divergence_link_2024_2025.py \
  --symbols LINKUSDT,SOLUSDT,ETHUSDT,BTCUSDT \
  --interval 1d \
  --year-start 2020-01-01 --year-end 2025-12-31 \
  --confluence-mode 2 \
  --require-dmi-category impulsion \
  --cci-fast 30 --cci-medium 90 \
  --cci-extreme 100 \
  --max-bottom-breaks 1 \
  --print-campaigns failed \
  --print-start-points no
```

### 9.2 Sortie attendue
Par symbole :
- `bars` : nombre de bougies 1D dans la période
- `events` : nombre d’événements (SHORT + confluence + impulsion)
- `accum_pairs` : nombre de patterns ACCUM
- `campaigns_total/campaigns_confirmed/campaigns_no_confirm_retest` : métriques principales (profit minimal avant invalidation)
- `accum_reached_2p/5p/10p/...` : % d’ACCUM qui atteignent un seuil avant invalidation
- `accum_avg_days_to_2p/5p/...` : délais moyens
- `accum_post_ok_2ev` : ratio de validation événementielle (étude secondaire)

En option (logs) :
- `CAMPAIGNS_FAILED_DETAIL` : campagnes sans confirmation avec détails (dates, breaks, entry_avg)
- `ACCUM_START_POINTS` : liste des dates `E2` (si `--print-start-points yes`)

---

## 10) Notes / limites

- Cette méthode décrit une **zone** (contexte), pas une stratégie complète.
- Les seuils de profit (`2/5/10/...%`) sont des **paramètres d’observation**.
- La sélection d’actifs (univers) a un impact fort : cette étude sert à comparer la qualité des zones par actif.
- Le modèle “campagne + bottom shifts” vise à coller à une logique d’accumulation progressive ; d’autres règles peuvent être ajoutées.
