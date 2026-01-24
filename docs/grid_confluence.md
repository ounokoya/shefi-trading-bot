# SPEC — Confluence Multi‑TF par “Grille” / “Zones” (5m, 1h, 4h) (v2)

Ce document décrit deux modes :

- **mode `grid`** : bucketisation log par pas en % (cellules discrètes)
- **mode `zones`** : regroupement en **zones** (clustering) avec bornes réelles + zones imbriquées par TF (macro → context → execution)

---

## 1) Règle fondamentale (runtime)

À l’instant **t** (prix courant `current`) :
- **résistance** = pivot / niveau **au‑dessus** du prix courant
- **support** = pivot / niveau **en‑dessous** du prix courant

Le `role` persisté (historique) ne doit pas influencer le présent : le rôle est **dérivé à runtime** par comparaison au prix courant.

---

## 2) Objectif

Construire une structure de confluence multi‑TF structurée par :
- un paramètre en % (selon le mode : pas de grille ou rayon)
- une **importance globale** (dépendante du TF)
- une **importance locale** (confluence CCI)

et identifier en priorité :
- les cellules de grille où il existe une confluence **5m + 1h + 4h** (même côté du prix courant)
- puis sélectionner les pivots les plus “jeunes” (abscisse = `bars_ago`) dans le rayon inclusif de la grille.

---

## 3) Mode `grid` — Définition de la grille (bucket)

La grille est définie de manière multiplicative (échelle log) avec un pas `(1 + grid_pct)`.

### 3.1 Anchor

On ancre la grille sur le prix courant `current_price`.

### 3.2 Clé de grille

Pour un niveau `level`, on calcule une clé entière `grid_key` :

- `x = level / current_price`
- `grid_key = round( log(x) / log(1 + grid_pct) )`

La cellule a un niveau central :

- `grid_level = current_price * (1 + grid_pct) ** grid_key`

### 3.3 Rayon inclusif

Un pivot “appartient” à une cellule si :

- `grid_d_abs = abs(level / grid_level - 1)`
- `grid_d_abs <= grid_pct` (inclusif)

---

## 4) Sources

Entrées : 3 `PivotRegistry` (5m, 1h, 4h).

Chaque zone locale fournit un pivot représentatif via :
- `selected_event_id` sinon
- dernier event de `event_ids` (plus “jeune”)

---

## 5) Importance

### 5.1 Importance globale (TF)

Poids fixes par TF (exemple v1) :
- 5m → 1
- 1h → 2
- 4h → 3

### 5.2 Importance locale (CCI)

Poids local = nombre de tags CCI présents dans l’event :
- `cci_weight = len(set(event.instant.tags_ccis))`

---

## 6) Cellule de confluence

Une cellule est considérée “confluente” si :
- elle contient au moins un pivot de chaque TF : 5m, 1h, 4h
- et que tous ces pivots sont du **même rôle à runtime** (support ou résistance), déterminé par `level` vs `current_price`.

---

## 7) Sélection “2 plus jeunes” (v1)

Dans chaque cellule confluent :
- on trie les pivots 5m par `bars_ago` croissant (plus petit = plus jeune)
- on garde les **2 plus jeunes pivots 5m**
- pour chacun, on associe un pivot 1h et 4h “le plus proche en prix” à l’intérieur de la cellule (tie‑break: plus jeune)

Scores :
- `importance_global = importance_tf(5m) + importance_tf(1h) + importance_tf(4h)`
- `importance_local = cci_weight(5m) + cci_weight(1h) + cci_weight(4h)`

---

## 8) Mode `zones` — Zones imbriquées par TF (macro → context → execution)

### 8.1 Idée

Au lieu de créer des cellules `grid_key`, on regroupe les pivots en **zones** avec :

- un **centre** (prix de référence, calculé en moyenne pondérée)
- des **bornes réelles** `[lower, upper]` (min/max des pivots contenus)
- une **liste des pivots** contenus (membres)

Puis on organise ces zones en hiérarchie par TF :

- **macro** (TF haut, ex: 4h)
  - **context** (TF moyen, ex: 1h)
    - **execution** (TF bas, ex: 5m)

### 8.2 Paramètres

Chaque niveau a ses paramètres :

- `radius_pct` : rayon inclusif utilisé pour regrouper les pivots (clustering)
- `padding_pct` : espacement minimum entre deux zones (si trop proche → merge)

En mode `zones`, `grid_pct` n’est **pas** le paramètre principal :

- si tu fournis `radius_pct`/`padding_pct` par niveau (macro/context/execution), `grid_pct` n’intervient pas
- sinon, `grid_pct` peut servir de **fallback** (compat) pour remplir les valeurs manquantes

### 8.3 Clustering + merge

Règle de proximité (inclusif) :

- deux pivots sont “proches” si `abs(a/b - 1) <= radius_pct`

Puis les zones résultantes sont triées par centre et on merge des zones voisines si :

- `abs(center1/center2 - 1) <= padding_pct`

### 8.4 Bornes réelles

Les bornes d’une zone sont :

- `lower = min(levels)`
- `upper = max(levels)`

### 8.5 Confluence & picks

Les picks (2 plus jeunes 5m) sont construits au niveau **execution** :

- on prend les 2 pivots 5m les plus jeunes de la zone execution
- pour chacun on associe :
  - un pivot 1h “le plus proche” dans la zone context (avec max distance `context.radius_pct`)
  - un pivot 4h “le plus proche” dans la zone macro (avec max distance `macro.radius_pct`)

---

## 9) Format de sortie

La fonction renvoie un JSON qui dépend du `mode`.

### 9.1 Sortie `mode=grid`

- `meta`:
  - `symbol`
  - `grid_pct`
  - `current_price`
  - `now_ts_ms`
  - `tfs`
  - `tf_importance`

- `cells[]`:
  - `role` (runtime)
  - `grid_key`
  - `grid_level`
  - `grid_pct`
  - `picks[]`:
    - `members.{5m|1h|4h}` avec champs: `tf`, `zone_id`, `event_id`, `dt_ms`, `dt`, `bars_ago`, `level`, `cci_tags`, `cci_weight`, `grid_key`, `grid_level`, `grid_d_abs`
    - `score.importance_global`
    - `score.importance_local`

### 9.2 Sortie `mode=zones`

- `meta`:
  - `symbol`
  - `mode` = `"zones"`
  - `current_price`
  - `now_ts_ms`
  - `tf_importance`
  - `zones_cfg.macro|context|execution` (`tf`, `radius_pct`, `padding_pct`)
  - `keep_top2_5m`

- `zones[]` (macro zones) :
  - `role`
  - `tf` (ex: `4h`)
  - `center_level`
  - `bounds.lower|upper` (bornes réelles)
  - `members[]` (pivots macro)
  - `subzones[]` (context)
    - (mêmes champs)
    - `subzones[]` (execution)
      - (mêmes champs)
      - `picks[]` (2 plus jeunes 5m)

---

## 10) Lib + Démo

### 9.1 Lib

- `libs/pivots/grid_confluence.py`
  - `build_grid_confluence(..., mode="grid"|"zones", zones_cfg=...)`

### 9.2 Démo (Bybit)

Le calcul est intégré dans :
- `scripts/16_demo_multi_tf_pivot_confluence_bybit.py`

Exemple :

### 10.2.1 Exemple mode `grid`

```bash
./venv_optuna/bin/python scripts/16_demo_multi_tf_pivot_confluence_bybit.py \
  --symbol LINKUSDT \
  --triad 5m_1h_4h \
  --limit 1000 \
  --grid-mode grid \
  --grid-pct 0.05
```

### 10.2.2 Exemple mode `zones` (hiérarchique)

```bash
./venv_optuna/bin/python scripts/16_demo_multi_tf_pivot_confluence_bybit.py \
  --symbol LINKUSDT \
  --triad 5m_1h_4h \
  --limit 1000 \
  --grid-mode zones \
  --grid-pct 0.05 \
  --zones-macro-radius-pct 0.10 \
  --zones-macro-padding-pct 0.05 \
  --zones-context-radius-pct 0.05 \
  --zones-context-padding-pct 0.03 \
  --zones-exec-radius-pct 0.02 \
  --zones-exec-padding-pct 0.01
```

Le fichier est écrit par défaut sous :

- mode `grid` : `data/processed/pivots/grid_{SYMBOL}_{TRIAD}_{N}pct.json`
- mode `zones` : `data/processed/pivots/zones_{SYMBOL}_{TRIAD}_M{M}C{C}E{E}.json` (M/C/E = radius% macro/context/exec)
