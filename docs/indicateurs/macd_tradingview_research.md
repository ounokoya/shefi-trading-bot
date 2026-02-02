# 🔍 MACD TradingView - Recherche d'Implémentation Précise

## 📋 Table des Matières
1. [Formule Officielle TradingView](#formule-officielle-tradingview)
2. [Spécification d’implémentation (reproductible, sans ambiguïté)](#-spécification-dimplémentation-reproductible-sans-ambiguïté)
3. [Sources et Références](#-sources-et-références)

---

## 🎯 Formule Officielle TradingView

### Définition
Le **MACD (Moving Average Convergence/Divergence)** est un indicateur de tendance et momentum qui combine deux moyennes mobiles de périodes différentes avec leur écart.

### Formules Mathématiques Complètes

#### 1. MACD Line

MACD Line = EMA(Close, 12) - EMA(Close, 26)

#### 2. Signal Line

Signal Line = EMA(MACD Line, 9)

#### 3. MACD Histogram

MACD Histogram = MACD Line - Signal Line

### Paramètres Standards TradingView
- **Fast EMA** : 12 périodes
- **Slow EMA** : 26 périodes
- **Signal EMA** : 9 périodes

---

## 🧩 Spécification d’implémentation (reproductible, sans ambiguïté)

Cette section décrit la logique utilisée par l’implémentation de référence du repo:

- `libs/indicators/momentum/macd_tv.py` (MACD)
- `libs/indicators/moving_averages/ema_tv.py` (EMA TradingView)

Entrées:

- Série `prices[i]` de longueur `n`.
- Paramètres: `fast_period`, `slow_period`, `signal_period` (entiers `> 0`).

Règles de validité:

- Une valeur est dite “non valide” si elle est `NaN` ou `Inf`.
- Si `n == 0`, la sortie est une liste vide.

EMA TradingView (normatif):

- `EMA(src, p)` utilise:
  - `alpha = 2 / (p + 1)`
  - seed SMA à l’index `p-1` via `sma_tv`.
- Tant que l’EMA n’est pas seedée, la sortie est non valide.
- Si à un index `i` la valeur source `src[i]` est non valide ou si `EMA[i-1]` est non valide:
  - `EMA[i]` devient non valide et l’algorithme repasse en mode “non seedé” (il attend un seed SMA valide plus tard).

Définitions MACD:

- `fast_ema = EMA(prices, fast_period)`
- `slow_ema = EMA(prices, slow_period)`
- `macd_line[i]`:
  - si `fast_ema[i]` et `slow_ema[i]` sont valides:
    - `macd_line[i] = fast_ema[i] - slow_ema[i]`
  - sinon:
    - `macd_line[i]` est non valide
- `signal_line = EMA(macd_line, signal_period)`
- `hist[i]`:
  - si `macd_line[i]` et `signal_line[i]` sont valides:
    - `hist[i] = macd_line[i] - signal_line[i]`
  - sinon:
    - `hist[i]` est non valide

---

## 📚 Sources et Références

### 📖 Documentation Officielle
1. **TradingView Support - MACD Documentation**
   - URL : https://www.tradingview.com/support/solutions/43000502344-macd-moving-average-convergence-divergence/
   - Contenu : Formules officielles, composants détaillés
   - Dernière consultation : 03/11/2025

2. **TradingView Pine Script Reference Manual**
   - URL : https://www.tradingview.com/pine-script-reference/v6/
   - Section : Built-in functions → ta.macd()
   - Dernière consultation : 03/11/2025

### 📚 Guides et Tutoriels
3. **TradingView Education - MACD**
   - URL : https://www.tradingview.com/education/macd/
   - Contenu : Stratégies et interprétations pratiques
   - Dernière consultation : 03/11/2025

4. **TradingView Scripts - MACD**
   - URL : https://www.tradingview.com/scripts/macd/
   - Contenu : Implémentations avancées et variantes
   - Dernière consultation : 03/11/2025

5. **CoinMonks - Creating MACD Oscillator**
   - URL : https://medium.com/coinmonks/creating-the-macd-oscillator-in-tradingview-the-full-guide-6ffe71e4a7f9
   - Contenu : Guide complet de création en Pine Script
   - Dernière consultation : 03/11/2025

### 🔍 Références Historiques
6. **Gerald Appel (1970s)** - Créateur original de la MACD Line
7. **Thomas Aspray (1986)** - Ajout de l'histogramme MACD

---

*Document créé le 03/11/2025 - Basé sur recherche TradingView et documentation officielle*
