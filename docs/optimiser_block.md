# Récap validé + Calibration MACD (Mode A) — Blocs / Tranches / Jambes / Features (v1)

Date: 2026-01-05 (UTC)

---

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

**Précision validée**
- Pour `t0 / tfav / t1`, on se base sur des **extrêmes de prix de clôture (close)** dans chaque tranche (plateau possible au sommet/creux).
- Ce n’est **pas** une logique “uniquement à la fin” côté trading : la notion “parfaite” sert surtout à **labelliser**.

---

## 3) Bloc = 3 tranches consécutives (validé)
Fenêtre glissante de **3 tranches** :

- **Bloc LONG** : `hist− (avant)` → `hist+ (milieu)` → `hist− (après)`
- **Bloc SHORT** : `hist+ (avant)` → `hist− (milieu)` → `hist+ (après)`

**Matérialisation dataset**
- Un bloc est “entier” quand les **3 tranches sont connues** (utile pour labelliser proprement).

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

✅ **1 jambe = 1 trade (une opportunité)**  
Donc un bloc peut offrir **2 trades** possibles (A ou B).  
Selon **tendance + contexte + signaux**, tu peux **ignorer** une jambe et **exploiter** l’autre.

---

## 6) Objectif trading vs apprentissage (validé, sans mélange)
### 6.1 Apprentissage (dataset)
Utiliser des blocs “propres” (t0/tfav/t1 connus) pour :
1) Prédire si la prochaine jambe sera ≥ **0.7%**
2) Décider prendre / ignorer
3) Prédire le % attendu de la jambe

Point clé : le modèle peut **scorer/filtrer des candidats** déjà détectés par la règle,
sans apprendre la détection sur toutes les bougies.

### 6.2 Trading (en continu, sans future)
En réel, t0/tfav/t1 ne sont pas “parfaits” : on gère des **candidats**.
La difficulté = gérer les faux candidats.
Approche : **règle de détection** + modèle qui **score**.

---

## 7) Stratifier : est-ce que tu en as besoin ? (validé)
### Cas 1 — Modèle sur candidats uniquement
➡️ **Stratification non obligatoire.**
Clés : split temporel train/valid/test + éviter dataset mono-régime.

### Cas 2 — Modèle apprend aussi “détecter”
➡️ Besoin d’exemples hors-candidats (neutres) → stratification utile.

---

## 8) Indicateurs / features retenus (validé comme périmètre)
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

## 9) Seuil amplitude MACD/MACD_hist (validé)
Remplacement de **10%** par **50% ATR** :

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

---

# 11) Nouveau : Calibration MACD “segmenteur” via Optuna (Mode A = upper bound)

## 11.1 Constat (validé)
- Le **MACD_hist** est l’élément qui détermine les **tranches → blocs → jambes**.
- Les **jambes parfaites** (t0/tfav/t1 en close) servent de **références** (upper bound) pour mesurer la “qualité” du découpage.

## 11.2 Objectif Mode A (validé)
Optimiser les paramètres du **MACD** (adaptés au TF) pour :
- **Maximiser** le **total % capté** par les jambes parfaites (upper bound)
- **Minimiser** le **max drawdown** sur une equity construite à partir des jambes

⚠️ Mode A = calibration segmentation, **pas** preuve de robustesse trading.

## 11.3 Risque principal identifié (validé)
Optuna peut “tricher” en trouvant des paramètres qui :
- génèrent **peu de blocs/jambes** (rare) mais très beaux → score artificiellement haut.

Donc : besoin de **contraintes / pénalités** anti-dégénérées.

---

# 12) Cadre proposé : Contraintes + Multi-objectif + Score (validé)

## 12.1 Contraintes “hard” anti-triche (validé)
Au moins une (ou plusieurs) des contraintes suivantes :
- `nb_jambes >= N_min` (ou `nb_blocs >= B_min`) sur la période
- `couverture_mois >= C_min`  
  (mois avec ≥1 bloc) / (total mois)
- (optionnel) contrainte sur l’activité **par mois** mais **pas sur tous les mois**  
  (pour accepter des mauvais mois)

## 12.2 Objectifs multi-objectif (Pareto) (validé)
Optimisation Pareto sur solutions **faisables** :
- **Max** `R_total` : somme des % captés par jambe parfaite
- **Min** `DD_max` : max drawdown sur l’equity (enchaînement temporel des jambes)
- **Max** `nb_jambes` : activité
- **Max** `couverture_mois` : diversité temporelle

## 12.3 “Mauvais mois” : gérer la queue sans interdire (validé)
Au lieu d’exiger “0 mauvais mois”, intégrer une mesure de stabilité :
- **P25_mensuel** (25% pires mois) à maximiser, ou
- **worst_month** (pire mois) à minimiser, ou
- `% mois négatifs` à minimiser (poids faible)

## 12.4 Score unique pour choisir dans le Pareto (validé)
Après Pareto, classer les solutions faisables avec un **score** :
- Normaliser chaque métrique (0→1)
- Exemple de score (concept) :
  - `Score = + wR*R_total − wDD*DD_max + wT*nb_jambes + wC*couverture + wS*stabilité`
- Le score sert à **sélectionner automatiquement** un point du front Pareto,
tout en gardant la **vue Pareto** pour audit.

---

# 13) Sorties attendues (prod)
- Paramètres MACD retenus + métriques : `R_total`, `DD_max`, `nb_jambes`, `couverture_mois`, stabilité mensuelle.
- Tableau/trace du **front Pareto** (top solutions).
- La solution “choisie” par score + justification via métriques normalisées.

---

# 14) Points restant à trancher (non figés ici)
- Comment définir précisément la **stabilité mensuelle** (P25 vs worst_month).
- Définition exacte des contraintes mensuelles si utilisées (fraction de mois concernés, etc.).
- Règles exactes d’enchaînement jambe→equity (si besoin d’une convention stricte unique).

