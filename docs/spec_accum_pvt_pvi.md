# Brouillon — Spécification stratégie ACCUM (PVT / PVT&PVI)

Ce document formalise la logique de la stratégie **ACCUM** telle qu’elle est utilisée dans le R\&D actuel, en se limitant aux flows :

- **Variante A** : PVT
- **Variante B** : PVT \& PVI en confluence ("PVT ET PVI")
 - **Variante C** : ASI
 - **Variante D** : ASI \& PVT en confluence stricte (extrêmes au même instant)

 Objectif : définir **sans ambiguïté** la chaîne complète :

 1. Segmentation en tranches via `MACD_hist`.
 2. Construction des **évènements** (extrêmes) sur les tranches.
 3. Filtrage des évènements via **CCI confluence** et optionnellement via **DMI catégorie**.
 4. Détection des signaux **ACCUM** sur basculement de phase du cycle (section 10).
 5. Émission d’évènements/signaux : `ACCUM`, `RETEST_EXTREME`, `BOTTOM_BREAK`, `REBOTTOM`.
 6. Construction d’un **prix moyen** basé sur les extrêmes validés (prix moyen de bottom/top selon le côté).

---

## 0) Données

Sur chaque bougie clôturée t :

- `close(t)`
- `high(t)` / `low(t)` (pour DMI)
- `volume(t)` (ou `turnover(t)` si choisi comme source)

Dans cette stratégie, le **prix** intervient de deux façons :

- `close` sert à définir l’**extrême de prix** dans chaque tranche (point temporel de l’évènement).
- `close` sert à construire un **prix moyen** basé sur les extrêmes (utile pour la suite : R\&D, ou usage par un moteur d’exécution).

Remarque : la confirmation "profit" (seuils %) était un mécanisme de R\&D. Elle n’est pas partie de la stratégie décrite dans ce document.

---

## 1) Segmentation : tranches MACD_hist

### 1.1 Définition

On calcule `MACD_hist(t)`.

Une **tranche** est un intervalle maximal \[a..b\] tel qu’il n’y a pas de changement de signe de `MACD_hist` à l’intérieur.

- tranche **LONG** : `MACD_hist` strictement positif après le début de tranche.
- tranche **SHORT** : `MACD_hist` strictement négatif après le début de tranche.

Remarque : `MACD_hist == 0` n’a pas de signe. Il est considéré comme appartenant aux deux côtés :

- à la fin d’une tranche, les zéros font partie de la tranche qui se termine
- au début de la tranche suivante, ces mêmes zéros font partie de la tranche qui commence

Les tranches sont donc délimitées par les changements de signe observés sur les valeurs non-nulles.

---

## 2) Flows

### 2.1 PVT

PVT est une série cumulée construite à partir du prix et du volume.

Dans cette spécification, **PVT est supposé déjà calculé** par le module indicateurs (formule hors périmètre de ce document). La stratégie attend une colonne `pvt(t)` en entrée.

Dans cette stratégie :

- on utilise **le mouvement** de PVT entre évènements (comparaison d’extrêmes),
- et **la valeur** de PVT (pour définir les bottoms de structure).

### 2.2 PVI

PVI est une série cumulée qui n’évolue que sur les bougies où le volume augmente.

Dans cette spécification, **PVI est supposé déjà calculé** par le module indicateurs (formule hors périmètre de ce document). La stratégie attend une colonne `pvi(t)` en entrée (variante B uniquement).

Dans la variante "PVT ET PVI" :

- PVT et PVI sont traités comme **deux flows distincts**,
- la confluence est un **ET logique** sur leur mouvement (et potentiellement sur leurs conditions de structure) : il n’existe **pas** de combinaison mathématique de PVT et PVI en un seul indicateur.

### 2.3 ASI

ASI (Accumulative Swing Index) est une série cumulée construite à partir de `open/high/low/close`.

Dans cette spécification, **ASI est supposé déjà calculé** par le module indicateurs. La stratégie attend une colonne `asi(t)` en entrée.

Dans cette stratégie :

- on utilise le mouvement de ASI entre évènements (comparaison d’extrêmes)
- et, dans les variantes ASI, l’instant de l’évènement peut être défini par l’extrême de ASI (voir section 3.2)

### 2.4 ASI & PVT (confluence stricte)

Dans la variante "ASI ET PVT" :

- ASI et PVT sont traités comme **deux flows distincts**
- la confluence est un **ET logique** sur leur mouvement
- et l’extrême d’évènement est **confluent strictement** : l’index de l’extrême ASI et l’index de l’extrême PVT dans la tranche doivent être **identiques** (sinon la tranche est rejetée)

---

## 3) Construction des évènements (extrêmes) par tranche

Un **évènement** est un point temporel associé à une tranche et à un extrême.

### 3.1 Tranche support

La stratégie construit des évènements sur :

- les tranches `SHORT` pour l’**accumulation côté LONG**
- les tranches `LONG` pour l’**accumulation côté SHORT**

### 3.2 Instant `t*` de l’évènement (selon le flow)

Pour chaque tranche \[a..b\], on définit un instant `t*` (un **instant unique**) selon la variante de flow :

- Variante A (PVT) et Variante B (PVT\&PVI) :
  - tranche `SHORT` : `t*` est l’instant où `close` est **minimum** sur \[a..b\]
  - tranche `LONG` : `t*` est l’instant où `close` est **maximum** sur \[a..b\]

- Variante C (ASI) :
  - tranche `SHORT` : `t*` est l’instant où `asi` est **minimum** sur \[a..b\]
  - tranche `LONG` : `t*` est l’instant où `asi` est **maximum** sur \[a..b\]

- Variante D (ASI\&PVT confluent strict) :
  - on calcule l’index de l’extrême ASI et l’index de l’extrême PVT dans \[a..b\]
  - si ces 2 index sont identiques, alors `t*` est cet index
  - sinon la tranche est rejetée (pas d’évènement)

### 3.3 Valeurs stockées par évènement

Chaque évènement E contient au minimum :

- `E.t` : instant de l’évènement (extrême de prix dans la tranche)
- `E.tranche` : \[a..b\]
- `E.price_extreme_close` : close à `E.t` (extrême de prix)

Chaque indicateur de flow possède son propre extrême sur la tranche. Ces extrêmes sont ceux qui sont utilisés pour l’analyse ACCUM :

- `E.price_extreme_close_in_tranche` : extrême de `close` sur \[a..b\]
  - tranche `SHORT` : `min(close)`
  - tranche `LONG` : `max(close)`

- `E.pvt_extreme_in_tranche` : extrême de PVT sur \[a..b\]
  - tranche `SHORT` : `min(PVT)`
  - tranche `LONG` : `max(PVT)`
- `E.pvi_extreme_in_tranche` : extrême de PVI sur \[a..b\] (si variante B)
  - tranche `SHORT` : `min(PVI)`
  - tranche `LONG` : `max(PVI)`

- `E.asi_extreme_in_tranche` : extrême de ASI sur \[a..b\] (si variante C ou D)
  - tranche `SHORT` : `min(ASI)`
  - tranche `LONG` : `max(ASI)`

---

## 4) Filtre CCI : confluence d’extrêmes dans la tranche

On calcule plusieurs CCI (ex : fast/medium, éventuellement slow) et un niveau `L`.

Pour une tranche \[a..b\], la confluence est OK si :

- tranche `SHORT` : pour chaque CCI activé, il existe au moins une bougie dans la tranche telle que `CCI(t) <= -L`
- tranche `LONG` : pour chaque CCI activé, il existe au moins une bougie dans la tranche telle que `CCI(t) >= +L`

Si un seul CCI n’atteint pas le seuil attendu dans la tranche, l’évènement est rejeté.

---

## 5) Filtre DMI : catégorie / filtre / force brute au moment de l’évènement

On calcule ADX, +DI, -DI, DX.

Paramètres :

- seuil ADX `T`

Au point de l’évènement, on calcule 3 champs distincts :

### 5.1 `dmi_category`

- `plat` si `ADX(E.t) < T`
- `tendenciel` sinon

### 5.2 `dmi_filter`

Ce filtre est indépendant de `dmi_category` :

- `impulsion` si `DX(E.t) > ADX(E.t)`
- `respiration` sinon

### 5.3 `dmi_force_brute`

Ce champ mesure une force brute au moment de l’évènement en comparant `DX(E.t)` aux 2 DI :

- `tres_fort` si `DX(E.t) > max(+DI(E.t), -DI(E.t))`
- `petite_force` si `DX(E.t) < min(+DI(E.t), -DI(E.t))`
- `moyen_fort` sinon

Par défaut, aucun filtrage DMI n’est imposé : on conserve toutes les catégories.

Un paramètre peut imposer de ne garder que certaines catégories.

---

## 6) Détection du signal ACCUM (bascule de phase du cycle)

 Un signal `ACCUM` est déclenché par un **basculement de phase** du cycle de tendance défini en section 10 (HH/HL vs LH/LL) sur les extrêmes de flow par tranche.

 Les évènements considérés sont ceux qui passent les filtres (section 4 et section 5).

 Définition :

 - `ACCUM_LONG` : bascule baissière → haussière (un **BOTTOM** est identifié au point d’intersection)
 - `ACCUM_SHORT` : bascule haussière → baissière (un **TOP** est identifié au point d’intersection)

 L’instant opérationnel du signal est l’instant de l’extrême structurel défini en section 10.5 :

 - `ACCUM_LONG.t = BOTTOM.t`
 - `ACCUM_SHORT.t = TOP.t`

 Remarque : en pratique, la détection d’un TOP/BOTTOM ne peut être confirmée qu’après observation du basculement (apparition des extrêmes suivants). Le signal est donc **daté** sur `TOP/BOTTOM.t` mais **connu** au moment où le basculement devient détectable.

---

## 7) Émission d’évènements / signaux (pas de money management)

La stratégie émet un flux d’évènements. Un moteur d’exécution peut choisir de les ignorer ou de les utiliser.

 Évènements principaux (par côté) :

- `ACCUM` : un basculement de phase du cycle est détecté (section 6 et section 10). Le signal est daté sur l’extrême structurel `TOP/BOTTOM` (section 10.5).
- `BOTTOM_BREAK` : le **flow** casse l’extrême de flow de référence ("bottom/top" de structure) du côté en cours (nouvel extrême plus bas en LONG, nouvel extrême plus haut en SHORT).
- `REBOTTOM` : après cassure, on identifie/fixe un **nouvel extrême de flow** (nouveau bottom/top de structure), et on actualise le prix moyen associé.

 ### 7.1 Symétrie ACCUM / DISTRIB (interprétation côté opposé)

La stratégie peut émettre des signaux `ACCUM_LONG` / `ACCUM_SHORT`.

Par construction, un même phénomène peut être interprété des deux côtés :

- `ACCUM_LONG` (accumulation pour LONG) est équivalent, du point de vue du camp opposé, à une **distribution côté SHORT** : `DISTRIB_SHORT`.
- `ACCUM_SHORT` (accumulation pour SHORT) est équivalent, du point de vue du camp opposé, à une **distribution côté LONG** : `DISTRIB_LONG`.

Dans cette section, `DISTRIB_*` n’est pas un évènement distinct : c’est la **relecture côté opposé du même évènement**, décrivant des **actions opposées entre camps** (accumuler vs distribuer) au sein d’une même zone.

Autrement dit, si le consommateur souhaite travailler avec un vocabulaire `ACCUM` et `DISTRIB`, il peut dériver :

- `DISTRIB_SHORT = ACCUM_LONG`
- `DISTRIB_LONG = ACCUM_SHORT`

Règle de base : il n’existe pas de "nouvelle campagne" indépendante tant que l’état de côté en cours n’a pas été invalidé/renversé.

---

## 8) Bottom (flow), break (flow), rebottom et prix moyen

`BOTTOM_BREAK` concerne le **bottom/top de structure** (référence de **flow**) et est évalué sur les **extrêmes de flow par évènement**.

Le "bottom/top" de structure ne concerne **pas** le prix :

- il correspond à une valeur d’extrême de **flow** (PVT, ASI, ou confluences),
- il est utilisé comme référence de structure,
- il est (re)défini lors du `REBOTTOM`.

On maintient un **prix moyen** basé sur les extrêmes de prix validés (et non sur une logique d’exécution).

 ### 8.1 Prix moyen de référence (par côté)

- côté LONG : `MeanBottomPrice` = moyenne arithmétique des extrêmes de prix `SHORT` retenus depuis le dernier `REBOTTOM`.
- côté SHORT : `MeanTopPrice` = moyenne arithmétique des extrêmes de prix `LONG` retenus depuis le dernier `REBOTTOM`.

 ### 8.2 Référence de structure (bottom/top) selon la variante de flow
 
 La référence de structure est fixée au moment où un **BOTTOM/TOP** est identifié à l’intersection de phase (section 10.5) :
 
 - côté LONG : `bottom_ref_*` correspond au **BOTTOM**
 - côté SHORT : `top_ref_*` correspond au **TOP**
 
 Le type de référence dépend de la variante :
 
 - Variante A (PVT) : une seule valeur `bottom_ref_pvt` / `top_ref_pvt`
- Variante C (ASI) : une seule valeur `bottom_ref_asi` / `top_ref_asi`
- Variante B (PVT\&PVI) : une **confluence** de 2 valeurs (`bottom_ref_pvt` ET `bottom_ref_pvi`, idem pour top)
- Variante D (ASI\&PVT) : une **confluence** de 2 valeurs (`bottom_ref_pvt` ET `bottom_ref_asi`, idem pour top)

 ### 8.3 Cassure (`BOTTOM_BREAK`) et rebottom (`REBOTTOM`)
 
 Une cassure est évaluée sur l’extrême de **flow** (pas sur le prix) :
 
 - côté LONG :
  - `BOTTOM_BREAK` si le(s) flow(s) font un nouvel extrême **plus bas** que la (les) référence(s)
- côté SHORT :
  - `BOTTOM_BREAK` si le(s) flow(s) font un nouvel extrême **plus haut** que la (les) référence(s)

Conditions par variante :
 
 - Variante A (PVT) :
  - LONG : `pvt_extreme_in_tranche(E) < bottom_ref_pvt`
  - SHORT : `pvt_extreme_in_tranche(E) > top_ref_pvt`

- Variante C (ASI) :
  - LONG : `asi_extreme_in_tranche(E) < bottom_ref_asi`
  - SHORT : `asi_extreme_in_tranche(E) > top_ref_asi`

- Variante B (PVT\&PVI) (confluence stricte, ET logique) :
  - LONG : `pvt_extreme_in_tranche(E) < bottom_ref_pvt` **ET** `pvi_extreme_in_tranche(E) < bottom_ref_pvi`
  - SHORT : `pvt_extreme_in_tranche(E) > top_ref_pvt` **ET** `pvi_extreme_in_tranche(E) > top_ref_pvi`

- Variante D (ASI\&PVT) (confluence stricte, ET logique) :
  - LONG : `pvt_extreme_in_tranche(E) < bottom_ref_pvt` **ET** `asi_extreme_in_tranche(E) < bottom_ref_asi`
  - SHORT : `pvt_extreme_in_tranche(E) > top_ref_pvt` **ET** `asi_extreme_in_tranche(E) > top_ref_asi`

Règle : un statut `BOTTOM_BREAK` n’est valide que si le (ou les) flow(s) font effectivement un extrême au niveau ou au-delà de la référence (dans le bon sens) sur un évènement.

Après cassure, si `REBOTTOM` est activé :
 
 - le bottom/top de structure (référence de flow) est mis à jour au nouvel extrême,
 - le prix moyen (section 8.1) repart à partir de ce nouveau régime.

 Définition de la référence de flow (bottom/top de structure) : elle correspond à la valeur de flow au point structurel **TOP/BOTTOM** défini à l’intersection de phase (section 10.5). Elle est donc associée au basculement de phase (section 6).
 
 ### 8.4 Principe de validation des cassures
 
 Une cassure de la référence de flow n'est valide que **si et seulement si** un extrême de flow est formé au niveau ou au-delà de la référence, dans le sens attendu :
 
 - côté LONG : formation d'un nouveau minimum sur le(s) flow(s) au-delà de la(les) référence(s) (voir section 8.3)
 - côté SHORT : formation d'un nouveau maximum sur le(s) flow(s) au-delà de la(les) référence(s) (voir section 8.3)

Sans cette formation d'extrême validée, il n'y a pas de cassure reconnue par la stratégie.
 
 ### 8.5 Rebottom après cassure validée

Lorsqu'une cassure est validée (avec formation d'extrême), si `REBOTTOM` est activé :

- le bottom/top de structure (référence de flow) est mis à jour au nouvel extrême validé,
- le prix moyen (section 8.1) est recalculé à partir de ce nouveau régime.

Le rebottom n'est possible que **après** une cassure validée selon le principe de la section 8.3.

---

## 9) Glossaire

- **Tranche** : segment de temps défini par le signe de `MACD_hist`.
- **Evènement** : point d’extrême dans une tranche (après filtres).
- **Séquence** : évènements consécutifs.
- **ACCUM_LONG / ACCUM_SHORT** : basculement de phase du cycle (section 6 et section 10).
- **REBOTTOM** : réinitialisation de la référence de structure (bottom/top) sur le **flow** et du prix moyen associé après cassure.

---

## 10) Cycle de tendance (phases haussière / baissière) basé sur les extrêmes de flow

Cette section définit une lecture "cycle de tendance" construite à partir des **extrêmes de flow** définis par tranches.

### 10.1 Sommets et creux de flow

Sur chaque tranche \[a..b\] on dispose d’un extrême de flow (section 3.3) :

- tranche `LONG` (`tranche_sign=+`) : l’extrême de flow correspond à un **sommet** (peak)
- tranche `SHORT` (`tranche_sign=-`) : l’extrême de flow correspond à un **creux** (trough)

Pour chaque variante, les valeurs à considérer sont :

- Variante A (PVT) : `pvt_extreme_in_tranche`
- Variante C (ASI) : `asi_extreme_in_tranche`
- Variante B (PVT\&PVI) : le couple (`pvt_extreme_in_tranche`, `pvi_extreme_in_tranche`)
- Variante D (ASI\&PVT) : le couple (`pvt_extreme_in_tranche`, `asi_extreme_in_tranche`)

### 10.2 Définition d’une phase haussière saine (sur flow)

On dit qu’on est en **phase haussière** (bullish) lorsque les deux structures suivantes sont vraies :

- **Sommets croissants** (HH) : les **2 derniers sommets** (peaks) de flow sont croissants
- **Creux croissants** (HL) : les **2 derniers creux** (troughs) de flow sont croissants

Pour une variante à deux flows (B ou D), la phase haussière est valide uniquement si le mouvement est **synchronisé** :

- chaque condition (HH et HL) doit être vraie **pour chacun des 2 flows**
- il n’y a pas de compensation : c’est un **ET logique**

### 10.3 Définition d’une phase baissière saine (sur flow)

On dit qu’on est en **phase baissière** (bearish) lorsque :

- **Sommets décroissants** (LH) : les **2 derniers sommets** (peaks) de flow sont décroissants
- **Creux décroissants** (LL) : les **2 derniers creux** (troughs) de flow sont décroissants

Pour une variante à deux flows (B ou D), la phase baissière est valide uniquement si le mouvement est **synchronisé** sur les 2 flows (ET logique).

### 10.4 Transition immédiate et durée d’une phase

La transition de phase est **immédiate** : une phase peut ne durer que **deux extrêmes** avant de s’inverser.

On ne cherche pas de confirmation supplémentaire au-delà des règles HH/HL (bullish) ou LH/LL (bearish).

### 10.5 Définition de TOP et BOTTOM (à l’intersection de phase)

On définit les points structurels sur le **flow** à l’intersection d’un changement de phase :

- **TOP** : l’extrême (sommet/peak) au sommet d’une phase haussière, situé à l’intersection où la phase bascule haussière → baissière.
- **BOTTOM** : l’extrême (creux/trough) au plus bas d’une phase baissière, situé à l’intersection où la phase bascule baissière → haussière.

 Pour lever l’ambiguïté "U/M" : on note les extrêmes de flow alternés dans le temps :

- `P1, P2, P3, ...` : sommets (peaks) sur tranches `LONG`
- `T1, T2, T3, ...` : creux (troughs) sur tranches `SHORT`

 Le basculement de phase est détecté quand les règles HH/HL (bullish) ou LH/LL (bearish) changent d’état (section 10.2 et 10.3).

 Définition opérationnelle de l’intersection : lorsqu’un basculement est détecté, on considère une fenêtre minimale d’extrêmes autour de l’intersection comprenant :

 - les **2 derniers sommets** et les **2 derniers creux** qui valident la phase sortante
 - les **2 premiers sommets** et les **2 premiers creux** qui valident la phase entrante

 Cela représente typiquement **4 sommets** et **4 creux** autour de l’intersection.

 Règle (basée sur les valeurs, pas sur la position) :

 - bascule haussière → baissière : **TOP** = le sommet (peak) de plus grande valeur parmi les sommets de la fenêtre d’intersection.
 - bascule baissière → haussière : **BOTTOM** = le creux (trough) de plus petite valeur parmi les creux de la fenêtre d’intersection.

 En cas d’égalité (plusieurs sommets au même maximum ou plusieurs creux au même minimum), on retient l’extrême **le plus récent** (index temporel le plus grand).

 Pour les variantes à deux flows, TOP/BOTTOM exigent :

 - que le basculement de phase soit valide en **synchronisation** (les 2 flows satisfont le basculement de phase)
 - et que l’extrême sélectionné (TOP ou BOTTOM) corresponde au **même évènement** (même index) pour les 2 flows
