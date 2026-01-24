Voici la **spécification** de la nouvelle hypothèse (à envoyer à Windsurf).

---

# Hypothèse Hx — Filtre “Tendance + Triple CCI” + Trades uniquement dans le sens de la tendance

## 1) Objectif

Améliorer la qualité des trades issus des **candidats d’extrêmes** (MACD_hist + MACD line) en :

* filtrant **le contexte** (tendance confirmée),
* filtrant **l’état momentum** (triple CCI),
* et en ne gardant que les **jambes alignées** avec la tendance.

Attendus :

* baisse des faux signaux (losers),
* baisse du DD (surtout jambe B),
* hausse du `winrate_ge_target` (≥ 0.7%),
* au prix d’un `n_trades` plus faible (filtre dur).

---

## 2) Contexte “Tendance” (Gate global)

On définit un **trend_side** à chaque timestamp (au minimum au moment `t_entry` de la jambe).

### 2.1 Indicateurs

* **Vortex** : `VI+`, `VI-` (ex: période 300)
* **DMI** : `DI+`, `DI-` (ex: période 300)

### 2.2 Règles de tendance

* **Tendance LONG** si : `VI+ > VI-` **ET** `DI+ > DI-`
* **Tendance SHORT** si : `VI+ < VI-` **ET** `DI+ < DI-`
* Sinon : **NEUTRE** (pas de trade)

---

## 3) Trades autorisés : uniquement dans le sens de la tendance

Pour chaque jambe (trade candidat) :

* On évalue `trend_side` à `t_entry` (défaut).
* On conserve la jambe **uniquement si** `leg_side == trend_side`.

Sinon : jambe ignorée (pas de trade).

---

## 4) Filtre “Triple CCI” (conditions d’entrée/sortie)

### 4.1 Indicateurs

* `CCI(30)`
* `CCI(100)`
* `CCI(300)`

### 4.2 Long (trend_side = LONG)

#### Condition entrée (au t0 candidat / t_entry)

* `CCI100 < 0`
* `CCI30 < -100`

#### Condition sortie (au tfav candidat / t_exit)

* `CCI100 > 100`
* `CCI30 > 100`
* `CCI300 > 100`

### 4.3 Short (trend_side = SHORT)

(symétrique)

#### Entrée

* `CCI100 > 0`
* `CCI30 > +100`

#### Sortie

* `CCI100 < -100`
* `CCI30 < -100`
* `CCI300 < -100`

---

## 5) Placement dans le pipeline existant (important)

Cette hypothèse ne remplace pas la détection de candidats existante, elle ajoute des **gates**.

### 5.1 Détection candidats

Conserver la détection actuelle :

* candidats basés sur changement de pente de `macd_hist` et `macd_line` (selon implémentation actuelle).

### 5.2 Gate “trend_side”

Avant d’accepter un trade, exiger `trend_side` non NEUTRE.

### 5.3 Gate “leg_side == trend_side”

Ne prendre que les jambes dans le sens trend.

### 5.4 Gate “triple CCI”

Appliquer :

* au moment de l’entrée (t_entry) : conditions CCI30/CCI100
* au moment de la sortie (t_exit/tfav) : conditions CCI30/CCI100/CCI300

---

## 6) Sorties / métriques à comparer (vs baseline)

Comparer aux baselines déjà mesurées (perfect, candidate alignés, candidate + stop) :

* `n_trades`
* `n_losers`, `sum_losers`
* `n_target_winners`, `winrate_ge_target`
* `capture_sum`
* `sum_target_winners`, `sum_small_winners`
* `dd_max`
* `n_stop_exits`, `sum_stop_exits` (si mode trades + stop actif)

---

## 7) Notes de design (points à trancher si besoin)

* Tendance évaluée à `t_entry` (défaut). Option : exiger tendance inchangée jusqu’à `t_exit` (plus strict).
* Si sortie CCI300 trop restrictive : possible d’évaluer un assouplissement (ex: CCI300 > 0) mais **pas dans cette hypothèse**.

---
