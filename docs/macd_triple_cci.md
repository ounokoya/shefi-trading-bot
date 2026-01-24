Voici le **résumé** à transmettre à Windsurf.

---

# Hypothèse Hx-t0 — “t0 = vrai pivot prix + pivot CCI (30/100/300)”

## Objectif

Renforcer la validité du **t0** : il doit être un **extrême de prix** et un **pivot momentum** confirmé par les **CCI**, pour éviter les entrées trop tôt (source de gros DD).

---

## 1) Règle t0 (LONG)

On part d’un **candidat t0** déjà détecté (MACD/hist).

### A) t0 extrême de prix

* `t0` doit être un **plus bas prix (close)** sur la zone de référence (selon le mode : tranche / fenêtre candidate).

### B) t0 “extrême CCI” au moment t0

* `CCI100 < 0`
* `CCI30 < -100`

### C) Après t0 : inversion CCI rapides, CCI300 plat ou début d’inversion

Dans une fenêtre **[t0 ; t0+K]** :

* `CCI30` doit **s’inverser à la hausse** (décroissant → croissant)
* `CCI100` doit **s’inverser à la hausse**
* `CCI300` doit devenir **plat** ou **commencer à s’inverser** (pas obligé de croître fortement)

Interprétation :

* CCI30/100 = “rebond démarre vraiment”
* CCI300 = “la pression long-terme arrête de s’aggraver (plat) ou tourne”

---

## 2) Règle t0 (SHORT) — symétrique

### A) t0 extrême de prix

* `t0` = **plus haut close** sur la zone de référence.

### B) t0 “extrême CCI” au t0

* `CCI100 > 0`
* `CCI30 > +100`

### C) Après t0 : inversion à la baisse

Dans **[t0 ; t0+K]** :

* `CCI30` s’inverse **à la baisse**
* `CCI100` s’inverse **à la baisse**
* `CCI300` devient **plat** ou commence à s’inverser **à la baisse**

---

## 3) Paramètres à fixer

* `K` = nb max de bougies après t0 pour observer l’inversion (ex: 3/6/12).
* Définition de **CCI300 “plat”** :

  * soit `abs(slope_CCI300) < eps` sur M bougies,
  * soit `abs(ΔCCI300) < eps` sur M bougies.

---

## 4) Effet attendu (métriques)

* `n_trades` ↓ (filtre plus strict)
* `dd_max` ↓, `n_losers` ↓ (moins d’entrées trop tôt)
* `winrate_ge_target` ↑ probable (trades plus propres)

---
