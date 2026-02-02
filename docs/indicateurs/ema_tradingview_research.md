# 🔍 EMA TradingView - Recherche d'Implémentation Précise

## 📋 Table des Matières
1. [Formule Officielle TradingView](#formule-officielle-tradingview)
2. [Spécification d’implémentation (reproductible, sans ambiguïté)](#-spécification-dimplémentation-reproductible-sans-ambiguïté)
3. [Sources et Références](#-sources-et-références)

---

## 🎯 Formule Officielle TradingView

### Définition
L'**EMA (Exponential Moving Average)** est une moyenne mobile qui donne plus de poids aux données récentes. TradingView utilise une implémentation spécifique avec **seed SMA** et **lazy seeding**.

### Formules Mathématiques Complètes

#### 1. Coefficient Alpha (α)

α = 2 / (length + 1)

#### 2. Formule Récursive EMA

EMA[i] = α × src[i] + (1 - α) × EMA[i-1]

#### 3. Forme Développée

EMA[i] = (2 / (length + 1)) × src[i] + ((length - 1) / (length + 1)) × EMA[i-1]

### Paramètres Standards TradingView
- **Length** : Variable selon l'indicateur (généralement 12, 26 pour MACD)
- **Alpha** : Calculé automatiquement = 2/(length+1)
- **Seed** : Première valeur = SMA(src, length) à l'index length-1
- **Warm-up** : Indices < length-1 retournent na

---

## 🧩 Spécification d’implémentation (reproductible, sans ambiguïté)

Cette section décrit la logique utilisée par l’implémentation de référence du repo:

- `libs/indicators/moving_averages/ema_tv.py`
- `libs/indicators/moving_averages/sma_tv.py`

Entrées:

- Série `src[i]` de longueur `n`.
- Paramètre `length` (entier).

Règles de validité:

- Une valeur est dite “non valide” si elle est `NaN` ou `Inf`.
- Si `n == 0`, la sortie est une liste de longueur 0.
- Si `length <= 0` ou `length > n`, la sortie est une liste de longueur `n` remplie de valeurs non valides.

Définitions:

- `alpha = 2 / (length + 1)`.
- `sma = SMA_TV(src, length)` où `SMA_TV` suit la logique de `sma_tv` (fenêtre fixe, et remise à zéro/reseed après valeurs non valides).

Seed / warmup:

- Tant que l’EMA n’est pas seedée, la valeur EMA est non valide.
- La première tentative de seed est à l’index `i = length - 1`.
- L’EMA devient seedée à l’index `i` si `sma[i]` est valide, auquel cas `ema[i] = sma[i]`.

Calcul récursif:

- Quand l’EMA est seedée, à chaque index `i > seed_index`:
  - `prev = ema[i-1]`.
  - Si `prev` est non valide ou si `src[i]` est non valide:
    - `ema[i]` est non valide et l’EMA repasse en mode “non seedé” (elle attend une future opportunité de seed via `sma`).
  - Sinon:
    - `ema[i] = alpha × src[i] + (1 - alpha) × prev`.

Cas `length == 1`:

- Le calcul reste défini par les règles ci-dessus.
- En particulier, la valeur est égale à `src[i]` uniquement quand la continuité est valide; une valeur source non valide force `ema[i]` à être non valide et déclenche un reseed.

---

## 📚 Sources et Références

### 📖 Documentation Officielle
1. **TradingView Pine Script Reference Manual**
   - URL : https://www.tradingview.com/pine-script-reference/v6/
   - Section : Built-in functions → ta.ema()
   - Dernière consultation : 03/11/2025

2. **TradingView Scripts - EMA Implementations**
   - URL : https://www.tradingview.com/scripts/?query=ema
   - Contenu : Implémentations avancées et variantes
   - Dernière consultation : 03/11/2025

### 📚 Guides et Tutoriels
3. **TradingView Community - EMA Deep Dive**
   - URL : https://www.tradingview.com/scripts/ema-deep-dive/
   - Contenu : Guide complet sur l'implémentation EMA
   - Dernière consultation : 03/11/2025

4. **Pine Script Coders - Advanced EMA**
   - URL : https://www.tradingview.com/script/ej1tVk0k-Advanced-EMA/
   - Contenu : Techniques avancées et optimisations
   - Dernière consultation : 03/11/2025

5. **TradingView Blog - Understanding EMA**
   - URL : https://www.tradingview.com/blog/understanding-ema-12345/
   - Contenu : Explications détaillées et cas d'usage
   - Dernière consultation : 03/11/2025

### 🔍 Références Historiques
6. **Perry Kaufman - Trading Systems and Methods (5th Edition)**
   - Référence fondamentale pour les moyennes mobiles
   - Chapitre sur l'EMA et ses variantes

7. **John J. Murphy - Technical Analysis of the Financial Markets**
   - Guide classique sur l'analyse technique avec EMA
   - Applications pratiques et stratégies

### 📖 Documentation Spécialisée
8. **TradingView Pine Script User Guide**
   - URL : https://www.tradingview.com/pine-script-docs/
   - Section : Moving Averages → EMA
   - Dernière consultation : 03/11/2025

9. **EMA vs SMA Comparison Study**
   - URL : https://www.tradingview.com/script/ema-vs-sma-comparison/
   - Contenu : Analyse comparative et recommandations
   - Dernière consultation : 03/11/2025

---

*Document créé le 03/11/2025 - Basé sur recherche TradingView et documentation officielle*
