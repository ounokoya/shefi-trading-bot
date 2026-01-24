# SPEC — Registres S/R Multi‑TF (5m, 1h, 4h) + Confluences Pairwise & Triple (5m_1h_4h) (v1)

Ce document détaille une architecture **Multi‑Timeframe (MTF)** basée sur des registres S/R **par TF** puis des registres **de confluence** entre TF :
- confluence **5m↔1h**
- confluence **1h↔4h**
- confluence **triple 5m↔1h↔4h** (5m_1h_4h)

L’objectif est de **ne pas mélanger** la détection locale de chaque TF, tout en disposant d’une couche “meta” de niveaux **plus lourds** (multi‑TF).

---

## 1) Invariants validés

### 1.0 Règle fondamentale (runtime)
À l’instant **t** (prix courant `current`) :
- **résistance** = pivot / niveau **au‑dessus** du prix courant
- **support** = pivot / niveau **en‑dessous** du prix courant

Le `role` historique (persisté dans les registres locaux) **ne doit pas** influencer l’état présent : le rôle est **dérivé à runtime** par comparaison au prix courant.

### 1.1 Une zone locale (par TF) ne dépend que de :
- proximité de prix dans un **seuil inclusif** `ε_tf`

✅ Une zone locale peut contenir :
- plusieurs catégories (`fast`, `medium`, `slow`) (selon votre modèle)
- plusieurs occurrences dans le temps

### 1.2 Stockage normalisé (anti‑redondance)
- Les **events** sont stockés une seule fois (base + confluence instantanée)
- Les **zones** stockent uniquement des **IDs d’events** (mémoire temporelle)  
- Les métriques vs prix courant (`d_pct`, `bars_ago`, quantiles…) sont **calculées à la demande**

---

## 2) Registres locaux par TF

On maintient un registre **indépendant** pour chaque timeframe :

### 2.1 `registry_5m`
- `events_5m`
- `zones_5m`
- `meta_5m`

### 2.2 `registry_1h`
- `events_1h`
- `zones_1h`
- `meta_1h`

### 2.3 `registry_4h`
- `events_4h`
- `zones_4h`
- `meta_4h`

Chaque registre local utilise ses propres paramètres :
- `ε_tf` (tolérance inclusive pour les zones)
- stratégie de mise à jour incrémentale (bootstrap + delta)

---

## 3) Registres de confluence Pairwise (2 TF)

L’idée : créer un registre “meta” qui lie des **zones locales** quand elles sont :
- du même côté du prix courant (rôle dérivé à runtime : support/résistance)
- dans une même bande de prix (matching inclusif)

### 3.1 `mtf_5m_1h`
Liaison entre `zones_5m` et `zones_1h`.

### 3.2 `mtf_1h_4h`
Liaison entre `zones_1h` et `zones_4h`.

> Optionnel : `mtf_5m_4h` (non requis si vous construisez le triple via intersection).

---

## 4) Registre de confluence Triple : `mtf_5m_1h_4h`

Une zone triple représente une bande de prix (par `role`) où :
- une zone **4h** existe
- une zone **1h** existe
- une zone **5m** existe
et elles se recouvrent selon un seuil de matching.

---

## 5) Définition d’un “match” de confluence (entre zones)

### 5.1 Critère minimal (validé)
- même rôle **à runtime** (support/résistance dérivé du prix courant)
- proximité de prix avec seuil inclusif `ε_mtf`

### 5.2 Référence prix d’une zone (sans duplication)
Une zone locale doit exposer une valeur de référence “prix représentatif” :
- soit via `selected_event_id` (référence = `events[selected_event_id].level`)
- soit via une statistique dérivée (`mean/median` des events de la zone)

Le **match** MTF se teste avec :
- `abs(zoneA.ref_price - zoneB.ref_price) <= ε_mtf` (inclusif)
ou variante intervalle si vous stockez des bornes.

---

## 6) Structure des registres MTF (anti‑redondance)

### 6.1 Zone MTF pairwise (ex : mtf_5m_1h)
Une entrée MTF **ne copie pas** les zones/events ; elle stocke des références :

- `role`
- `eps_mtf`
- `members` :
  - `tf_low.zone_id`
  - `tf_high.zone_id`
- `updated_dt` (optionnel)
- `strength` (optionnel, dérivé) : ex. nb touches, âge, confluence instantanée dominante, etc.

### 6.2 Zone MTF triple (mtf_5m_1h_4h)
- `role`
- `eps_mtf3`
- `members` :
  - `tf5m.zone_id`
  - `tf1h.zone_id`
  - `tf4h.zone_id`
- `updated_dt` (optionnel)
- `strength` (optionnel, dérivé)

**Evidence (optionnel, sans copie lourde)**
- `selected_event_ids` :
  - `tf5m.selected_event_id`
  - `tf1h.selected_event_id`
  - `tf4h.selected_event_id`

> Important : on garde le principe “références uniquement”.

---

## 7) Construction du registre Triple (méthode recommandée)

### 7.1 Approche A — Intersection via la zone 1h (recommandée)
1) Construire `mtf_5m_1h`
2) Construire `mtf_1h_4h`
3) Construire `mtf_5m_1h_4h` par **intersection** :

Pour une entrée de `mtf_1h_4h` (qui lie une zone `1h` et `4h`) :
- vérifier si cette même zone `1h` apparaît dans une entrée de `mtf_5m_1h`
- si oui, créer (ou mettre à jour) une entrée triple contenant :
  - `zone_5m` + `zone_1h` + `zone_4h`

Avantages :
- réutilise les calculs pairwise
- évite un matching direct “3‑directions”
- plus stable et plus simple à maintenir

### 7.2 Approche B — Matching direct 4h→1h→5m (alternative)
- partir des zones 4h
- trouver match 1h
- puis match 5m
Plus complexe, mais possible.

---

## 8) Mise à jour incrémentale des registres MTF

### 8.1 Quand reconstruire / mettre à jour MTF ?
- après un bootstrap local (TF)
- après insertion de nouveaux events/zones (delta) sur l’un des TF concernés
- selon fréquence (ex : au moment où vous rafraîchissez `registry_1h`)

### 8.2 Stratégie simple
- mettre à jour pairwise (5m↔1h, 1h↔4h)
- puis recalculer l’intersection triple

---

## 9) Pourquoi la confluence multi‑TF est importante

### 9.1 5m↔1h
- niveaux “travaillés” intraday
- utiles pour timing, stops locaux, TP1

### 9.2 1h↔4h
- niveaux “majeurs”
- utiles pour structure globale, zones de retournement / frein

### 9.3 Triple 5m↔1h↔4h
- souvent “mur majeur”
- très utile pour :
  - prudence maximale à l’approche
  - TP/SL/trailing par niveaux
  - filtrer des entrées “collées” à un mur multi‑TF

---

## 10) Runtime (calcul à la demande au moment d’un signal)

Au moment d’un signal (ex : en 5m), vous pouvez tagger :
- `mtf_pair_5m_1h = true/false`
- `mtf_pair_1h_4h = true/false`
- `mtf_triple_5m_1h_4h = true/false`

Puis calculer à la demande :
- distance du prix courant à la zone MTF (via ref_price)
- “room” avant contact
- position dans un corridor (quantile) par catégorie si nécessaire

---

## 11) Champs minimaux à persister (checklist)

### Local TF (x3)
- `events_tf`
- `zones_tf`
- `meta_tf`

### MTF
- `mtf_5m_1h` (références vers zone_id)
- `mtf_1h_4h` (références vers zone_id)
- `mtf_5m_1h_4h` (références vers zone_id)

---

Fin.
