# Récap validé — Blocs, Tranches, Jambes, Features (v1)

## 1) Base & alignement des données (validé)
- **1 ligne = 1 bougie**, clé `ts`.
- À chaque bougie : **OHLCV + indicateurs + features** (instant + window) + **métas tranches** (runs MACD hist) + **métas blocs** (si la bougie appartient à un bloc).
- Les features sont calculées **sur toutes les bougies**, **indépendamment** des blocs.

---

## 2) Tranches MACD histogramme (validé)
- Une **tranche** = suite de bougies consécutives où **MACD_hist garde le même signe** :
  - `hist+` si hist > 0
  - `hist−` si hist < 0
- Une tranche se termine au **flip** de signe.
- Chaque bougie peut porter : `tranche_id`, `tranche_sign`, `start/end`, `len`, etc.

> Précision intégrée : pour la définition des points t0/tfav/t1, on se base sur des **extrêmes de prix de clôture (close)** dans chaque tranche (plateau possible au sommet/creux), pas sur une logique “tranche figée uniquement à la fin” côté trading.

---

## 3) Bloc = 3 tranches consécutives (validé)
Fenêtre glissante de **3 tranches** :

- **Bloc LONG** : `hist− (avant)` → `hist+ (milieu)` → `hist− (après)`
- **Bloc SHORT** : `hist+ (avant)` → `hist− (milieu)` → `hist+ (après)`

**Matérialisation dataset** : un bloc est “entier” quand les **3 tranches sont connues** (utile pour labelliser proprement).

---

## 4) Points t0 / tfav / t1 (validé + précision)
### 4.1 LONG (− / + / −)
- `t0, p0` = **plus bas close** dans `hist− avant`
- `tfav, pfav` = **plus haut close** dans `hist+ milieu`
- `t1, p1` = **plus bas close** dans `hist− après`

### 4.2 SHORT (+ / − / +)
- `t0, p0` = **plus haut close** dans `hist+ avant`
- `tfav, pfav` = **plus bas close** dans `hist− milieu`
- `t1, p1` = **plus haut close** dans `hist+ après`

---

## 5) Définition “jambe” (ajout + validé)
Dans un bloc, il y a **2 jambes** :

- **Jambe A (départ)** : `t0 → tfav`
- **Jambe B (fin)** : `tfav → t1`

### Jambe = trade (validé)
✅ **1 jambe = 1 trade (une opportunité)**  
Donc un bloc peut offrir **2 trades** possibles (A ou B).  
Et selon **tendance + contexte + signaux**, tu peux **ignorer** une jambe et **exploiter** l’autre.

---

## 6) Objectif trading vs apprentissage (validé, sans mélange)
### 6.1 Apprentissage (dataset)
- Tu peux utiliser des blocs “propres” (t0/tfav/t1 connus) pour :
  1) **Prédire** si la prochaine jambe sera ≥ **0.7%**
  2) **Décider** prendre / ignorer
  3) **Prédire** le % attendu de la jambe

Ici tu n’es pas obligé d’introduire des “faux t0/tfav/t1” **si** le modèle sert à **scorer/filtrer** des **candidats déjà détectés** par ta règle.

### 6.2 Trading (en continu, sans future)
- En réel, tu n’as pas t0/tfav/t1 “parfaits” : tu as des **candidats** à valider tôt.
- La difficulté = gérer les **faux candidats**.
- Mais ça peut être géré **par ta règle de détection** + le modèle qui **score** le candidat, sans que le modèle apprenne la détection “sur toutes les bougies”.

---

## 7) Stratifier : est-ce que tu en as besoin ? (réponse stricte)
### Cas 1 — Ton plan actuel : modèle **sur candidats** uniquement
➡️ **Stratification non obligatoire.**  
Parce que :
- ton code fournit déjà des candidats (filtre fort),
- le modèle apprend : “parmi ces candidats, lesquels donnent ≥0.7 / lesquels ignorer / quel %”.

✅ Dans ce cas, le point clé, c’est surtout :
- split temporel train/valid/test,
- éviter que ton dataset soit “tout 2021 bull” et presque rien ailleurs.

### Cas 2 — Tu veux que le modèle apprenne aussi “détecter”
➡️ Là il faut des **hors-candidats** (bougies/fenêtres où ta règle ne détecte rien).  
Et là, stratifier devient utile (sinon le modèle voit surtout du neutre d’un seul régime).

### À quoi sert la stratification (objectif)
- Éviter un modèle “bon” uniquement sur **un régime** (ex : vol haute / bull) parce que ton échantillon neutre ou tes candidats viennent majoritairement de ce régime.
- Donc stratifier = forcer une **variété de contextes** (vol, tendance, périodes), pour robustesse.

---

## 8) Indicateurs/features retenus (validé comme périmètre)
### Tendance / structure
- Vortex(300)
- DMI DI+/DI− (300)

### Extrêmes
- CCI(20)
- CCI(300)

### Volatilité
- ATR(14)

### Signaux
- MACD (line + hist) **normalisés par ATR**
- VWMA4 & VWMA12 (validation de signal)

### Forme / dérivés / quantiles / dynamiques
- Fenêtres forme : **3 / 6 / 12**
- Quantiles : **300**
- Pente / accélération / arrêt d’accélération
- Appliqué à : `close, vwma4, vwma12, cci20, cci300, macd_hist_norm_atr, macd_line_norm_atr`

---

## 9) Seuil amplitude MACD/MACD_hist (mise à jour)
Tu remplaces **10%** par **50% ATR** :

- Validation candidat :  
  `abs(macd_line_norm_atr) >= 0.50` **OU** `abs(macd_hist_norm_atr) >= 0.50`

---

## 10) Catalogue des features “forme / dérivés” à trier (validé)
Appliquées sur W ∈ {3,6,12} :

### Dérivés simples
1) Δ total, 2) Δ%, 3) range, 4) range%, 5) position max/min,  
6) monotone up/down, 7) nb flips de pente, 8) std(Δ1),  
9) somme abs(Δ1), 10) efficiency ratio.

### Pente / accélération / jerk
11) pente last, 12) pente moyenne, 13) accel last (Δ2),  
14) accel moyenne, 15) jerk last (Δ3, surtout W=6/12).

### “Arrêt d’accélération”
16) flip accel +→− (bool), 17) nb flips accel, 18) max abs(Δ2).

### Fit linéaire
19) slope_lin, 20) intercept, 21) R²_lin, 22) std résiduelle, 23) résidu last signé.

### Fit quadratique (courbure)
24) a2 (courbure), 25) a1, 26) R²_quad, 27) std résiduelle,  
28) position sommet/creux t*, 29) concave/convexe (signe a2).

### Patterns simples
30) V, 31) ∧, 32) U/∩ via a2 + t*, 33) plateau, 34) break (Δ1_last vs médiane).
