# SPEC — Registre S/R normalisé (Events + Zones) + Confluences + Mises à jour incrémentales (v1)

Ce document formalise **exactement** le workflow et la structure de données validés pour le module **Support/Résistance (S/R)**, afin d’être transmis à Windsurf.

---

## 1) Objectif

Construire et maintenir un registre S/R **robuste et exploitable en trading**, en séparant :

- **Stockage persistant** (JSON / DB) : niveaux historiques + regroupement en zones.
- **Calcul à la demande** (runtime) : distances au prix courant, âge en bougies, quantiles, sélection “niveau proche”, etc.

L’objectif est d’obtenir, autour d’un signal, une **carte de contexte** (niveaux proches, importance, mémoire) sans redondance de données.

---

## 2) Principes validés (invariants)

### 2.0 Règle fondamentale (runtime)
À l’instant **t** (prix courant `current`) :
- **résistance** = pivot / niveau **au‑dessus** du prix courant
- **support** = pivot / niveau **en‑dessous** du prix courant

Le `role` historique (persisté) **ne doit pas** influencer l’état présent : le rôle est **dérivé à runtime** par comparaison au prix courant.

### 2.1 Zone = proximité de prix (pas catégorie)
Une **zone** regroupe des niveaux :
- **proches en prix** selon un **seuil inclusif** `ε` (ex. “dans ±ε%”)

✅ Une zone peut contenir :
- plusieurs catégories (`fast`, `medium`, `slow`)
- plusieurs niveaux d’une même catégorie
- des occurrences à des dates différentes

### 2.2 Un niveau stocké = 1 event par tranche + confluence CCI de tranche
Lorsqu’un niveau est identifié (par la méthode de détection existante), on stocke **une seule fois** un **Event**.

Spécification : un niveau correspond à **1 extrême par tranche** (donc **1 event par tranche**).

L’event stocke :
- **champs de base**
- **confluence CCI de tranche** (CCI présents + forces), calculée sur les bougies de la tranche (pas une lecture “instantanée” sur une seule bougie).

### 2.3 Confluence temporelle = mémoire de zone (liste d’IDs)
La **mémoire temporelle** (confluence temporelle) d’une zone est :
- une **liste chronologique** d’IDs d’events appartenant à cette zone
- on ne recopie pas les events dans la zone (structure normalisée)

### 2.4 Les métriques “par rapport au prix courant” ne sont pas stockées
Les champs suivants sont **calculés uniquement au besoin** (runtime) :
- `d_price`, `d_pct`, `d_abs`
- `bars_ago`
- quantiles “support → résistance” par catégorie
- sélection du “niveau le plus jeune / plus proche” selon rayon

---

## 3) Modèle de données normalisé

### 3.1 `events` — table des niveaux individuels (source de vérité)
Clé : `event_id` (unique)

Valeur : **base + confluence CCI de tranche**

**Champs de base (validés via fonction existante)**
- `dt` (timestamp UTC)
- `level` (prix)
- `cat` : `fast|medium|slow`
- `role` : `support|resistance` (historique / à titre indicatif)
- `kind` : `LOW|HIGH`

> Rappel : au moment d’un signal, le rôle “support/résistance” est déterminé par la règle fondamentale (niveau vs prix courant), indépendamment du `role` historique.

**Confluence CCI de tranche (obligatoire)**
- `tranche.tags_ccis` : ensemble `{fast?, medium?, slow?}`
  (si plusieurs CCI franchissent les seuils d’extrême pendant la tranche, ils sont **tous** inclus)
- `tranche.strength_fast` (nullable)
- `tranche.strength_medium` (nullable)
- `tranche.strength_slow` (nullable)
- `tranche.sync_bars` (optionnel si calculé)

#### Recommandation d’ID (anti-collision)
Pour éviter les collisions (même dt possible) :
- `event_id = "{dt_ms}:{cat}:{role}:{kind}"`  
ou équivalent stable.

---

### 3.2 `zones` — regroupement par rôle + bande de prix
Clé : `zone_id` (unique)

Valeur minimale :
- `role` : `support|resistance` (optionnel / indicatif)
- `eps` : seuil de zone (inclusif) — même logique que votre “rayon”
- `event_ids` : liste d’IDs **triée chronologiquement (dt croissant)**

Optionnel (si besoin, sinon dérivable depuis `event_ids`) :
- `created_dt`
- `updated_dt`
- `selected_event_id` (si vous voulez pointer un “représentant” persistant)
- `notes` / `version`

---

### 3.3 Index en mémoire (pour éviter de scanner toutes les zones)
Comme vous n’êtes pas dans une DB relationnelle, il faut un **index** pour rattacher un nouvel event à une zone sans boucle sur toutes les zones.

**Index recommandé : bucket prix**
- `zone_index[role][bucket_key] -> [zone_id, ...]`

Le bucket sert uniquement à réduire la recherche. La validation finale reste :
- **même role**
- `abs(level - zone_ref_price) <= eps_inclusive`  (ou équivalent %)

> La référence `zone_ref_price` peut être soit un champ stocké, soit reconstruite à partir d’un `selected_event_id` (via `events[selected_event_id].level`), soit une statistique (médiane/moyenne) recalculée.

---

## 4) Confluences (données attachées et reconstruisibles)

### 4.1 Confluence CCI de tranche (attachée à un event)
Un `event` emporte sa **confluence CCI de tranche** :
- quels CCI ont été en extrême pendant la tranche (`tags_ccis`)
- forces par catégorie (`strength_*`)
- optionnel : synchronicité (`sync_bars`)

### 4.2 Confluence temporelle (attachée à une zone)
Une `zone` emporte une **mémoire temporelle** via :
- `event_ids = [event_id_0, event_id_1, ...]` (ordre chronologique)

✅ À tout moment, on peut reconstruire la table complète de la zone :
- `memory = [events[id] for id in zone.event_ids]`

**Indicateurs dérivés possibles (non stockés par défaut)**
- `touches = len(event_ids)`
- `first_dt`, `last_dt`
- `span_bars` (durée en bougies)
- `gaps_bars` (stats d’espacement)

---

## 5) Workflow validé (déclenchements + incrémental)

### 5.1 Bootstrap initial (seule analyse “complète” sur fenêtre)
Déclenché une seule fois par TF, sur les limites choisies (exemple validé) :
- TF 5m : 7 jours
- TF 15m : 14 jours
- TF 30m / 1h : 30 jours
- TF 4h : 3 mois
- TF 8h : 6 mois
- TF 1d : 1 an 6 mois

Étapes :
1) Charger l’historique selon la fenêtre TF
2) Produire les `events` (base + confluence CCI de tranche) via la méthode de détection
3) Construire les `zones` (par `role` + seuil de prix inclusif `eps`)
4) Construire/mettre à jour `zone_index` (bucket) en mémoire
5) Persister `events + zones + meta`

### 5.2 Redémarrage (compléter, sans refaire l’analyse complète)
Déclenché au redémarrage du service :
1) Charger `events + zones + meta`
2) Récupérer uniquement les nouvelles données depuis `meta.last_ts`
3) Produire uniquement les **nouveaux events** (dt > last_ts_event)
4) Rattacher chaque event à une zone existante via `zone_index`, sinon créer une nouvelle zone
5) Mettre à jour `meta.last_ts` et persister

✅ Pas de réanalyse de toute la fenêtre.

### 5.3 Mise à jour “slow” (compléter ce qui manque)
Déclenchée :
- quand un nouvel événement “slow” apparaît
- ou à une fréquence décidée

Principe :
- même logique que redémarrage : **delta depuis la dernière date persistée**
- les `events` et `zones` sont **complétés**, pas reconstruits depuis zéro

---

## 6) Calculs runtime (à la demande) autour du prix courant

### 6.1 Distances et âge (non stockés)
À partir d’un event sélectionné (ou d’une zone + logique de sélection) :
- `d_price = level - current`
- `d_pct = (level - current) / current`
- `d_abs = abs(d_pct)`
- `bars_ago` (distance en bougies depuis `dt`)

### 6.2 Quantile “support → résistance” par catégorie
À l’instant t, vous sélectionnez (déjà validé : “plus jeune + rayon”):
- `S_cat` = support de catégorie `cat` (event)
- `R_cat` = résistance de catégorie `cat` (event)

Quantile :
- `q_cat = (current - S_cat.level) / (R_cat.level - S_cat.level)` (si R>S)

Interprétation :
- `q≈0` proche support
- `q≈1` proche résistance

Côté position :
- LONG : risque côté résistance → indicateur = `1 - q_cat`
- SHORT : risque côté support → indicateur = `q_cat`

### 6.3 Validation “mémoire temporelle” d’un trigger (solidité)

Objectif : lorsqu’un trigger (extrême confirmé + confluence CCI de tranche) se produit à un niveau `level`, on veut vérifier que ce niveau correspond à un **pivot solide dans le temps**.

Principe (runtime) :

1) Sélectionner dans le registre tous les `events` historiques dont le `level` est dans la bande **inclusive** :
   - `[level*(1-radius_pct), level*(1+radius_pct)]`

2) Filtrer par type d’extrême cohérent avec la direction :
   - trigger LONG => pivots `kind=LOW`
   - trigger SHORT => pivots `kind=HIGH`

3) Trier les events sélectionnés **du plus jeune au plus vieux** (dt décroissant).

4) Calculer :
   - `n_fast`, `n_medium`, `n_slow` = nombre d’occurrences (par catégorie CCI primaire)
   - `score` = somme des poids par event, avec pondération :
     - `fast = 1`
     - `medium = 2`
     - `slow = 3`

Critère de solidité (validation) :
- `is_solid = (n_slow >= 1) OR (n_medium >= 2) OR (n_fast >= 4)`

Remarques :
- On exclut l’event de la **tranche courante** (le trigger ne doit pas “se valider lui-même”).
- Les métriques détaillées (liste des events + distances) restent **runtime** et ne sont pas persistées.

---

## 7) Liaison events ↔ zones (comment on “fait le lien”)

### 7.1 À l’insertion d’un event
Input : `event(level, role, cat, dt, ...)`

1) Calculer `bucket_key` depuis `level` (et éventuellement paramètres TF)
2) Récupérer une petite liste de zones candidates :
   - `candidates = zone_index[role][bucket_key]`
   - + éventuellement buckets adjacents
3) Tester seulement ces zones :
   - condition finale : `abs(level - zone_ref_price) <= eps` (inclusif)
4) Si match :  
   - `zones[zone_id].event_ids.append(event_id)` (en ordre chrono)
   - (option) `event.zone_id = zone_id` (utile debug)
5) Sinon :
   - créer une nouvelle `zone_id`
   - initialiser `event_ids=[event_id]`
   - enregistrer dans `zone_index`

---

## 8) Avantages (raison d’être de cette architecture)

- **Aucune redondance** : un niveau est stocké une fois (base + confluence CCI de tranche)
- **Mémoire temporelle légère** : zone = liste d’IDs
- **Reconstructions faciles** : table/print d’une zone = join en mémoire `events[id]`
- **Scalable sans DB relationnelle** : index bucket évite le scan complet
- **Compatible live** : calculs au prix courant faits à la demande

---

## 9) Champs minimaux à persister (checklist)

### JSON / DB
- `events`: dict `event_id -> event_object`
- `zones`: dict `zone_id -> {role, eps, event_ids}`
- `meta`:
  - `last_ts` (dernier timestamp traité)
  - `tf`, `symbol`, version, paramètres `eps`, etc.

### En mémoire (runtime)
- `zone_index` (bucket)
- cache des zones/événements (dict) pour reconstruction rapide

---

Fin.
