# 🔍 MFI TradingView - Recherche d'Implémentation Précise
 
 ## 📋 Table des Matières
 1. [Spécification d’implémentation (reproductible, sans ambiguïté)](#-spécification-dimplémentation-reproductible-sans-ambiguïté)
 2. [Formule Officielle TradingView](#formule-officielle-tradingview)
 3. [Calculs Détaillés](#calculs-détaillés)
 4. [Astuces et Optimisations](#astuces-et-optimisations)
 5. [Cas d'Usage Avancés](#cas-dusage-avancés)
 6. [Sources et Références](#sources-et-références)

---

## 🧩 Spécification d’implémentation (reproductible, sans ambiguïté)

Cette section est normative pour ce repo (elle décrit exactement la logique implémentée dans `libs/indicators/volume/mfi_tv.py`).

Entrées:

- Séries de même longueur `n`: `high[i]`, `low[i]`, `close[i]`, `volume[i]`.
- Paramètre `period` (entier).

Règles de validité / pré-conditions:

- Si les longueurs diffèrent: l’implémentation lève une exception.
- Si `period <= 0` ou `n == 0`: la sortie est une liste de longueur `n` remplie de valeurs non valides.
- L’implémentation ne filtre pas explicitement `NaN/Inf` dans les entrées. Par conséquent:
  - une valeur non valide peut se propager aux calculs via les multiplications/sommes,
  - et rendre `mfi[i]` non valide par propagation arithmétique.

Étape 1 — Typical Price et Raw Money Flow:

- `tp[i] = (high[i] + low[i] + close[i]) / 3`
- `raw_mf[i] = tp[i] * volume[i]`

Étape 2 — Positive/Negative flows:

- `pos[0] = 0.0`, `neg[0] = 0.0`
- Pour `i >= 1`:
  - si `tp[i] > tp[i-1]`:
    - `pos[i] = raw_mf[i]`, `neg[i] = 0.0`
  - sinon si `tp[i] < tp[i-1]`:
    - `pos[i] = 0.0`, `neg[i] = raw_mf[i]`
  - sinon:
    - `pos[i] = 0.0`, `neg[i] = 0.0`

Étape 3 — Sommes glissantes et MFI:

- Pour les index strictement avant `period`, `mfi[i]` reste non valide.
- L’implémentation commence à produire des valeurs à partir de `i = period`:
  - fenêtre: `j ∈ [i - period + 1, i]` (soit exactement `period` valeurs)
  - `sum_pos = Σ pos[j]`
  - `sum_neg = Σ neg[j]`
  - calcul:
    - si `sum_pos > 0` et `sum_neg == 0` => `mfi[i] = 100.0`
    - sinon si `sum_pos == 0` et `sum_neg > 0` => `mfi[i] = 0.0`
    - sinon si `sum_pos == 0` et `sum_neg == 0` => `mfi[i] = 50.0`
    - sinon:
      - `ratio = sum_pos / sum_neg`
      - `mfi[i] = 100.0 - (100.0 / (1.0 + ratio))`

## 🎯 Formule Officielle TradingView

### Définition
Le **Money Flow Index (MFI)** est un oscillateur de momentum qui mesure la pression d'achat et de vente en analysant à la fois le prix et le volume. Il est similaire au RSI mais avec l'ajout du volume.

### Formule Mathématique Complète
MFI = 100 - (100 / (1 + Money Flow Ratio))

### Étapes de Calcul (4 étapes obligatoires)

#### Étape 1 - Typical Price (TP)
TP = (High + Low + Close) / 3

#### Étape 2 - Raw Money Flow (RMF)
RMF = TP × Volume

#### Étape 3 - Money Flow Ratio
Money Flow Ratio = (Positive Money Flow) / (Negative Money Flow)

- **Positive Money Flow** : Somme des RMF des périodes où TP > TP précédent
- **Negative Money Flow** : Somme des RMF des périodes où TP < TP précédent

#### Étape 4 - Money Flow Index
MFI = 100 - (100 / (1 + Money Flow Ratio))

---

## 📝 Calculs Détaillés

### Processus Complet pour Période 14

1. **Calculer TP pour chaque bougie**
    - TP[i] = (High[i] + Low[i] + Close[i]) / 3

2. **Calculer RMF pour chaque bougie**
    - RMF[i] = TP[i] × Volume[i]

3. **Classifier le flux d'argent**
    - Si TP[i] > TP[i-1]:
      - Positive Flow = RMF[i], Negative Flow = 0
    - Si TP[i] < TP[i-1]:
      - Positive Flow = 0, Negative Flow = RMF[i]
    - Si TP[i] = TP[i-1]:
      - Positive Flow = 0, Negative Flow = 0

4. **Calculer les sommes sur `period` périodes**
    - À chaque index `i`, utiliser une fenêtre de `period` éléments.
    - SumPositive = Σ PositiveFlow[j] sur la fenêtre.
    - SumNegative = Σ NegativeFlow[j] sur la fenêtre.

5. **Calculer le ratio final**
    - MFRatio = SumPositive / SumNegative
    - MFI = 100 - (100 / (1 + MFRatio))

---

## ⚡ Astuces et Optimisations

### 1. Sources Alternatives pour Plus de Précision
- Selon les plateformes, la “source” peut varier (ex: HLC3, OHLC4, HL2, weighted close).
- Dans ce repo, la définition normative utilise TP = (High + Low + Close) / 3.

### 2. Périodes Optimisées par Style
- Le paramètre `period` contrôle le compromis “réactivité vs stabilité”.
- Exemples usuels (indicatifs): 7 (court), 14 (standard), 20-30 (plus stable).

### 3. Niveaux Dynamiques
- Variante: adapter les seuils de surachat/survente (ex: 80/20) en fonction de la volatilité.

**Note importante** : Le MFI standard TradingView n'inclut aucun filtre de volume. La formule officielle utilise uniquement les sommes glissantes de Positive/Negative Money Flow sans lissage additionnel.

---

## 📊 Cas d'Usage Avancés

### 1. MFI Multi-Timeframe
- Variante classique: calculer le MFI sur un timeframe supérieur, puis aligner et “reporter” la série sur un timeframe inférieur.

### 2. Système MFI + Price Action
- Variante: utiliser le MFI comme filtre (ex: MFI < 20 / MFI > 80) puis confirmer avec des règles de price action.

### 3. MFI avec Zones de Accumulation/Distribution
- Variante: rechercher des zones “accumulation/distribution” via un MFI extrême et un prix relativement stable.

---

## 🎯 Points Clés à Retenir

### ✅ Avantages du MFI TradingView
- **Volume intégré** : Plus complet que le RSI
- **Oscillateur borné** : 0-100 pour niveaux clairs
- **Divergences puissantes** : Très fiables avec volume
- **Universel** : Fonctionne sur tous les marchés

### ⚠️ Points d'Attention
- **Dépendance au volume** : Moins fiable sur marchés peu liquides
- **Lag similaire au RSI** : Signal retardé
- **Niveaux subjectifs** : 80/20 sont des standards
- **False signals** : En trend fort peut rester extrême

### 🚀 Meilleures Pratiques
- Utiliser HLC3 comme source par défaut
- Combiner avec analyse de volume
- Adapter les niveaux selon l'instrument
- Confirmer avec price action ou autres indicateurs

---

## 📚 Sources et Références

### 📖 Documentation Officielle
1. **TradingView Support - MFI Documentation**
   - URL : https://www.tradingview.com/support/solutions/43000502348-money-flow-mfi/
   - Contenu : Formules officielles, étapes de calcul détaillées
   - Dernière consultation : 03/11/2025

2. **TradingView Pine Script Reference Manual**
   - URL : https://www.tradingview.com/pine-script-reference/v6/
   - Section : Built-in functions → ta.mfi()
   - Dernière consultation : 03/11/2025

### 📚 Guides et Tutoriels
3. **Money Wave Script - Visual Adaptive MFI**
   - URL : https://www.tradingview.com/script/SrwWcJpZ-Money-Wave-Script-Visual-Adaptive-MFI/
   - Contenu : Implémentation visuelle avancée avec HLC3
   - Dernière consultation : 03/11/2025

4. **TradingView Scripts - Money Flow Index**
   - URL : https://www.tradingview.com/scripts/moneyflow/
   - Contenu : Scripts communautaires et variantes
   - Dernière consultation : 03/11/2025

### 🔍 Références Historiques
5. **Gene Quong and Avrum Soudack**
   - Créateurs originaux du MFI
   - Référence fondamentale pour la théorie

---

## 📋 Implémentation Go Référence

Cette documentation ne contient volontairement aucun extrait de code. La section normative du repo (au début du document) définit complètement le calcul de manière reproductible.

---

 *Document créé le 03/11/2025 - Basé sur recherche TradingView et documentation officielle*
