# Spec stratégie — Identification des extrêmes

Ce document décrit les façons disponibles dans le projet pour identifier un **extrême** (ou candidat d’extrême) en s’appuyant sur la **segmentation en tranches MACD_hist**.

## Définition commune : tranche MACD_hist

Une **tranche** est un segment continu où le signe effectif de `macd_hist` est constant.

- `macd_hist > 0` => tranche `+`
- `macd_hist < 0` => tranche `-`
- `macd_hist == 0` => prend le **signe effectif précédent**

Implémentation :
- `libs/blocks/segment_macd_hist_tranches_df.py` (produit `tranche_id`, `tranche_sign`, `tranche_start_ts`, `tranche_end_ts`, `tranche_len`, etc.)

Mapping standard (utilisé ci-dessous) :
- tranche `-` => extrêmes **LOW** => biais `LONG`
- tranche `+` => extrêmes **HIGH** => biais `SHORT`

## Données attendues (minimum)

- `ts` : timestamp en millisecondes UTC
- `close`
- `macd_hist`

Selon les méthodes, des colonnes supplémentaires peuvent être requises (ex: `high/low`, `vwma_*`, `atr_*`).

---

# Combien de façons d’identifier un extrême ?

On a actuellement **3 façons** principales (2 “core” + 1 variante “candidats filtrés”).

---

# Méthode A — Live / streaming : extrême confirmé à i+1 (sans lookahead)

## Objectif
Identifier un **candidat extrême** sur la bougie `i` mais **uniquement confirmable à `i+1`**.

C’est la version conçue pour le **trading live** sur un flux de klines.

## Règle (close)
Sur la tranche en cours (non achevée) :

- On considère `cand_i = n-2` et `now_i = n-1` (dans le dataset actuel)
- tranche `-` (LOW)
  - `close[cand_i]` doit être un **nouveau record low strict** depuis le début de tranche
  - la confirmation se fait **uniquement** à `now_i` si `close[now_i] >= close[cand_i]` (la bougie après n’est pas plus low)
  - si confirmé => `open_side = LONG`
- tranche `+` (HIGH)
  - `close[cand_i]` doit être un **nouveau record high strict** depuis le début de tranche
  - confirmation à `now_i` si `close[now_i] <= close[cand_i]`
  - si confirmé => `open_side = SHORT`

⚠️ Conséquence : la dernière bougie (`now_i`) ne peut jamais être « l’extrême ». Elle sert à **valider** l’extrême de l’avant-dernière.

## Implémentation
- `libs/blocks/get_current_tranche_extreme_signal.py`
  - retourne un dict runtime:
    - `is_extreme_confirmed_now`
    - `extreme_ts`, `extreme_close` (pointent sur `cand_i`)
    - `open_side`
    - infos tranche (`tranche_sign`, `tranche_start_ts`, `tranche_len`)

## Démo
- `scripts/08_demo_extreme_candidates_dec2025.py`
  - dataset : LINKUSDT 5m décembre 2025
  - fenêtre glissante 600
  - export : `data/processed/klines/candidates_extremes_LINKUSDT_5m_2025-12.csv`

---

# Méthode B — Analyse / offline : 1 extrême parfait par tranche (lookahead autorisé)

## Objectif
Sur un dataset complet, une tranche étant entièrement connue, extraire **un seul extrême “parfait” par tranche**.

C’est la version conçue pour l’**analyse** (pas pour le live), car elle utilise l’information future à l’intérieur de la tranche.

## Règle (close)
Pour chaque `tranche_id`:

- tranche `-` => extrême = `min(close)` sur toute la tranche => `open_side = LONG`
- tranche `+` => extrême = `max(close)` sur toute la tranche => `open_side = SHORT`

## Implémentation
- `libs/blocks/extract_tranche_perfect_close_extremes_df.py`
  - sortie : 1 ligne par tranche avec:
    - `tranche_id`, `tranche_sign`, `tranche_start_ts`, `tranche_end_ts`, `tranche_len`
    - `extreme_kind`, `extreme_ts`, `extreme_close`, `open_side`
    - `extreme_row_index`

## Démo
- `scripts/09_demo_tranche_perfect_extremes_dec2025.py`
  - export : `data/processed/klines/tranche_perfect_extremes_LINKUSDT_5m_2025-12.csv`

---

# Méthode C — Analyse / engineering : candidats extrêmes basés sur la forme du MACD_hist

## Objectif
Identifier un **premier extrême candidat** dans une tranche via la structure locale de `macd_hist` (retournements de pente), avec des options de filtrage (VWMA align) et de “stop invalidation” (pct/ATR).

C’est utile pour des variantes de stratégie où l’extrême n’est pas simplement “record close” mais un point de retournement détecté sur l’indicateur.

## Règle (résumé)
Dans chaque tranche, on regarde `macd_hist`:
- on calcule `d = diff(macd_hist)` et son signe
- on repère des points de retournement:
  - `- -> +` : minima de l’histogramme
  - `+ -> -` : maxima de l’histogramme
- selon `tranche_sign`, on sélectionne des candidats puis on peut:
  - filtrer (VWMA4 / VWMA12 / MACD align)
  - invalider des candidats selon un stop (pct ou ATR)

## Implémentation
- `libs/blocks/add_tranche_hist_extreme_candidates_df.py`
  - calcule des colonnes “candidat extrême” par tranche, dont:
    - `tranche_cand_extreme_ts`, `tranche_cand_extreme_price`
    - `tranche_cand_is_event`, `tranche_cand_rank`, etc.

Consommation dans la construction de blocks:
- `libs/blocks/add_blocks_multislot_df.py` (option `use_tranche_candidate_extremes=True`)

---

# Résumé rapide

- **A (live)**: `get_current_tranche_extreme_signal` => extrême confirmé strictement à `i+1` (pas de lookahead)
- **B (offline)**: `extract_tranche_perfect_close_extremes_df` => 1 extrême parfait par tranche (lookahead)
- **C (offline/feature)**: `add_tranche_hist_extreme_candidates_df` => candidats basés sur la forme de `macd_hist` (+ filtres/stop)
