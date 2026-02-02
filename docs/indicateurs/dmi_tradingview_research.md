# 🔍 DMI TradingView - Recherche d'Implémentation Précise

## 📋 Table des Matières
1. [Formule Officielle TradingView](#formule-officielle-tradingview)
2. [Calculs Détaillés](#calculs-détaillés)
3. [Implémentations Pine Script](#implémentations-pine-script)
4. [Astuces et Optimisations](#astuces-et-optimisations)
5. [Cas d'Usage Avancés](#cas-dusage-avancés)
6. [Sources et Références](#sources-et-références)

---

## 🧩 Spécification d’implémentation (reproductible, sans ambiguïté)

Cette section décrit la logique utilisée par l’implémentation de référence du repo:

- `libs/indicators/momentum/dmi_tv.py` (DMI/ADX)
- `libs/indicators/moving_averages/rma_tv.py` (RMA TradingView / Wilder smoothing)

Entrées:

- Séries de même longueur `n`: `high[i]`, `low[i]`, `close[i]`.
- Paramètres:
  - `period` (entier `> 0`) = longueur DI.
  - `adx_smoothing` (entier `> 0` ou `None`). Si `None`, l’implémentation utilise `adx_period = period`.

Règles de validité:

- Une valeur est dite “non valide” si elle est `NaN` ou `Inf`.
- Si `period <= 0` ou `period > n`, l’implémentation retourne 3 listes de longueur `n` remplies de valeurs non valides.
- Si `adx_period <= 0` ou `adx_period > n`, l’implémentation retourne 3 listes de longueur `n` remplies de valeurs non valides.

Étape 1 — True Range (`tr`):

- `tr[0] = high[0] - low[0]`.
- Pour `i >= 1`:
  - `hl = high[i] - low[i]`
  - `hc = abs(high[i] - close[i-1])`
  - `lc = abs(low[i] - close[i-1])`
  - `tr[i] = max(hl, hc, lc)`

Étape 2 — Directional Movement:

- Initialisation:
  - `plus_dm[0] = 0.0`
  - `minus_dm[0] = 0.0`
- Pour `i >= 1`:
  - `up_move = high[i] - high[i-1]`
  - `down_move = low[i-1] - low[i]`
  - `plus_dm[i] = up_move` si `up_move > down_move` et `up_move > 0`, sinon `0.0`
  - `minus_dm[i] = down_move` si `down_move > up_move` et `down_move > 0`, sinon `0.0`

Étape 3 — Lissage (Wilder/RMA):

- `tr_smooth = RMA_TV(tr, period)`
- `plus_dm_smooth = RMA_TV(plus_dm, period)`
- `minus_dm_smooth = RMA_TV(minus_dm, period)`

Étape 4 — +DI / -DI:

- Pour chaque index `i`:
  - Si `tr_smooth[i]` est valide et `tr_smooth[i] != 0`:
    - `plus_di[i] = (plus_dm_smooth[i] / tr_smooth[i]) * 100`
    - `minus_di[i] = (minus_dm_smooth[i] / tr_smooth[i]) * 100`
  - Sinon:
    - `plus_di[i]` et `minus_di[i]` sont non valides.

Étape 5 — DX:

- Pour chaque index `i`:
  - Si `plus_di[i]` et `minus_di[i]` sont valides:
    - `di_sum = plus_di[i] + minus_di[i]`
    - Si `di_sum != 0`:
      - `dx[i] = abs(plus_di[i] - minus_di[i]) / di_sum * 100`
    - Sinon:
      - `dx[i] = 0.0`
  - Sinon:
    - `dx[i]` est non valide.

Étape 6 — ADX:

- `adx = RMA_TV(dx, adx_period)`

Sortie:

- La fonction retourne `(adx, plus_di, minus_di)`.

## 🎯 Formule Officielle TradingView

### Composants du DMI
Le DMI (Directional Movement Index) se compose de **trois indicateurs** :
1. **ADX** (Average Directional Index) - Force de la tendance
2. **+DI** (Plus Directional Indicator) - Direction haussière
3. **-DI** (Minus Directional Indicator) - Direction baissière

### Formules Mathématiques Complètes

#### 1. Directional Movement (+DM / -DM)
- `UpMove[i] = High[i] - High[i-1]`
- `DownMove[i] = Low[i-1] - Low[i]`
- `+DM[i] = UpMove[i]` si `UpMove[i] > DownMove[i]` et `UpMove[i] > 0`, sinon `0.0`
- `-DM[i] = DownMove[i]` si `DownMove[i] > UpMove[i]` et `DownMove[i] > 0`, sinon `0.0`

#### 2. True Range (TR)
- `TR[0] = High[0] - Low[0]`
- Pour `i >= 1`:
  - `TR[i] = max(High[i] - Low[i], abs(High[i] - Close[i-1]), abs(Low[i] - Close[i-1]))`

#### 3. Directional Indicators (+DI / -DI)
TradingView utilise le lissage de Wilder, ce repo l’implémente via `RMA_TV` (voir `docs/indicateurs/rma_tradingview_research.md`).

- `TR_smooth = RMA_TV(TR, period)`
- `+DM_smooth = RMA_TV(+DM, period)`
- `-DM_smooth = RMA_TV(-DM, period)`
- Pour chaque index `i`, si `TR_smooth[i]` est valide et `TR_smooth[i] != 0`:
  - `+DI[i] = 100 × (+DM_smooth[i] / TR_smooth[i])`
  - `-DI[i] = 100 × (-DM_smooth[i] / TR_smooth[i])`

#### 4. Directional Index (DX)
- Si `+DI[i]` et `-DI[i]` sont valides:
  - `DX[i] = 100 × abs(+DI[i] - -DI[i]) / (+DI[i] + -DI[i])`
  - si `(+DI[i] + -DI[i]) == 0`, alors `DX[i] = 0.0`

#### 5. Average Directional Index (ADX)
- `ADX = RMA_TV(DX, adx_period)` avec `adx_period = adx_smoothing` si fourni, sinon `period`.

---

## 📝 Calculs Détaillés

### Étape 1 - Calcul du Directional Movement
Pour chaque période :
- Calculer `UpMove` et `DownMove`
- Déterminer `+DM` et `-DM` selon les règles
- Le plus grand des deux mouvements est retenu

### Étape 2 - Calcul du True Range
Le TR prend toujours le maximum des trois valeurs :
- High - Low (range de la période)
- |High - Previous Close| (gap up)
- |Low - Previous Close| (gap down)

### Étape 3 - Lissage avec Wilder's Smoothing
TradingView utilise **Wilder's Smoothing** (variante de l'EMA) :
`RMA(value, period)` (Wilder) suit la spécification normative de `docs/indicateurs/rma_tradingview_research.md`.

### Étape 4 - Calcul Final
- Normaliser +DM et -DM par le TR
- Appliquer le lissage sur +DI et -DI
- Calculer DX puis lisser pour obtenir ADX

---

## ⚡ Astuces et Optimisations

### 1. Paramètres Optimisés par Style de Trading
 - Le choix des périodes contrôle le compromis “réactivité vs stabilité”.
 - Exemples usuels (indicatifs):
   - Day trading: périodes plus courtes (ex: 7)
   - Swing trading: périodes standard (ex: 14)
   - Position trading: périodes plus longues (ex: 21)

### 2. Filtres de Trend Strength
 - Un usage courant consiste à filtrer les signaux DI par la force de tendance ADX.
 - Exemples usuels (indicatifs):
   - Trend fort: `ADX > 25`
   - Trend faible: `ADX < 20`
   - Absence de trend: `ADX < 15`
 - Ce type de filtre peut être combiné à d’autres critères (ex: oscillateurs) selon la stratégie.

### 3. Amélioration de la Précision
 - Selon les implémentations/plateformes, l’utilisation d’une source “typical” (ex: `HLC3`) peut stabiliser certains calculs.
 - Un lissage additionnel (ex: moyenne simple sur quelques périodes) peut réduire le bruit des séries `ADX`, `+DI`, `-DI`.

### 4. Multi-Timeframe DMI
 - Variante classique: calculer `ADX/+DI/-DI` sur un timeframe supérieur, puis les “reporter” sur un timeframe inférieur.
 - En pratique, cela revient à recalculer l’indicateur sur la série agrégée du timeframe supérieur et à aligner temporellement les résultats.

---

## 📊 Cas d'Usage Avancés

### 1. DMI avec Zones Dynamiques
 - Variante: adapter le seuil ADX (ex: 25) en fonction de la volatilité du marché (ex: via un indicateur de volatilité).
 - L’idée est de relever le seuil quand la volatilité est élevée et de l’abaisser quand elle est faible.

### 2. Système de Trading Complet
 - Un schéma fréquent combine:
   - un signal directionnel via croisement `+DI/-DI`,
   - un filtre de force via `ADX`,
   - et une confirmation externe (ex: volume supérieur à une moyenne de volume).

### 3. Divergences DMI
 - Des divergences peuvent être recherchées entre:
   - le prix (ex: plus haut / plus bas),
   - et la force de tendance (`ADX`).
 - La définition exacte d’un pivot et d’une divergence dépend de la méthode choisie (fenêtres, validation, etc.).

---

## 🎯 Points Clés à Retenir

### ✅ Avantages du DMI TradingView
- **Mesure de tendance** : ADX indique la force sans la direction
- **Direction claire** : +DI vs -DI pour sens de la tendance
- **Non-borné** : ADX peut monter indéfiniment en trend fort
- **Universel** : Fonctionne sur tous les timeframes et instruments

### ⚠️ Points d'Attention
- **Lag important** : DMI a un décalage significatif
- **Seuils subjectifs** : ADX 25/20 sont des recommandations
- **False signals** : croisements DI en trend faible sont peu fiables
- **Complexité** : Nécessite de l'expérience pour l'interprétation

### 🚀 Meilleures Pratiques
- Utiliser ADX > 25 comme filtre de trend minimum
- Combiner avec d'autres indicateurs pour confirmation
- Adapter les seuils selon l'instrument et la volatilité
- Privilégier les croisements en trend fort (ADX élevé)

---

## 📚 Sources et Références

### 📖 Documentation Officielle
1. **TradingView Support - DMI Documentation**
   - URL : https://www.tradingview.com/support/solutions/43000502250-directional-movement-dmi/
   - Contenu : Formules officielles, calculs détaillés, interprétation
   - Dernière consultation : 03/11/2025

2. **TradingView Pine Script Reference Manual**
   - URL : https://www.tradingview.com/pine-script-reference/v6/
   - Section : Built-in functions → ta.dmi()
   - Dernière consultation : 03/11/2025

### 📚 Guides et Tutoriels
3. **Tartigradia DMI Implementation**
   - URL : https://www.tradingview.com/script/5jVJuobZ-Directional-Movement-Indicator-DMI-and-ADX-Tartigradia/
   - Contenu : Implémentation manuelle complète avec Wilder's smoothing
   - Dernière consultation : 03/11/2025

4. **DinoTradez ADX-DMI Indicator**
   - URL : https://www.tradingview.com/script/eqAAiLTU-ADX-DMI/
   - Contenu : Calculs manuels avec techniques de lissage Wilder
   - Dernière consultation : 03/11/2025

5. **Medium - Mastering Market Direction**
   - URL : https://medium.com/@blackcat1402.tradingview/mastering-market-direction-complete-analysis-of-dmi-indicator-3aa349744976
   - Contenu : Analyse complète et applications pratiques
   - Dernière consultation : 03/11/2025

### 🔍 Références Historiques
6. **J. Welles Wilder - New Concepts in Technical Trading Systems (1978)**
   - Créateur original du DMI, RSI, ATR et Parabolic SAR
   - Référence fondamentale pour tous les calculs

---

 *Document créé le 03/11/2025 - Basé sur recherche TradingView et documentation officielle*
