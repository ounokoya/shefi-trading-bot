# 🔍 SMA TradingView - Recherche d'Implémentation Précise

## 📋 Table des Matières
1. [Formule Officielle TradingView](#formule-officielle-tradingview)
2. [Spécification d’implémentation (reproductible, sans ambiguïté)](#-spécification-dimplémentation-reproductible-sans-ambiguïté)
3. [Sources et Références](#-sources-et-références)

---

## 🎯 Formule Officielle TradingView

### Formule Mathématique Complète

SMA = (Sum of values over length) / length

---

## 🧩 Spécification d’implémentation (reproductible, sans ambiguïté)

Cette section décrit la logique utilisée par l’implémentation de référence du repo (`libs/indicators/moving_averages/sma_tv.py`).

Entrées:

- Série `src[i]` de longueur `n`.
- Paramètre `length` (entier).

Règles de validité:

- Une valeur est dite “non valide” si elle est `NaN` ou `Inf`.
- Si `n == 0`, la sortie est une liste de longueur 0.
- Si `length <= 0` ou `length > n`, la sortie est une liste de longueur `n` remplie de valeurs non valides.

Définitions:

- La SMA est définie sur une fenêtre de taille fixe `length`.
- Le calcul ne produit une valeur valide à l’index `i` que si les `length` dernières valeurs de la fenêtre sont toutes valides.

Règle de calcul (équivalente à l’implémentation):

- On maintient une somme glissante `s` et un compteur `count`.
- À chaque index `i`:
  - si `src[i]` est non valide:
    - `s = 0`, `count = 0`, et `sma[i]` est non valide.
  - sinon:
    - `s += src[i]` et `count += 1`.
    - si `i >= length`:
      - on considère `old = src[i-length]`.
      - si `old` est valide:
        - `s -= old` et `count -= 1`.
    - si `i >= length-1` ET `count == length`:
      - `sma[i] = s / length`.
    - sinon:
      - `sma[i]` est non valide.

### Caractéristiques Clés
- **Fenêtre fixe** : Toujours exactement `length` valeurs
- **Pondération égale** : Chaque valeur a le même poids (1/length)
- **Non récursive** : Recalcule complètement à chaque barre
- **Gestion des NA** : Les premières `length-1` barres retournent `na`

---

## 📚 Sources et Références

### 📖 Documentation Officielle
1. **TradingView Pine Script Reference Manual**
   - URL : https://www.tradingview.com/pine-script-reference/v6/
   - Section : Built-in functions → ta.sma()
   - Dernière consultation : 03/11/2025

2. **TradingView Built-ins Documentation**
   - URL : https://www.tradingview.com/pine-script-docs/language/built-ins/
   - Section : Technical indicators in the ta namespace
   - Dernière consultation : 03/11/2025

3. **TradingView Functions FAQ**
   - URL : https://www.tradingview.com/pine-script-docs/faq/functions/
   - Section : How do I calculate averages?
   - Dernière consultation : 03/11/2025

### 📚 Guides et Tutoriels
4. **Pine Script SMA Complete Guide**
   - URL : https://offline-pixel.github.io/pinescript-strategies/pine-script-SMA.html
   - Auteur : Offline Pixel Trading Strategies
   - Contenu : Exemples pratiques et implémentations
   - Dernière consultation : 03/11/2025

5. **TradingCode.net - Simple Moving Average**
   - URL : https://www.tradingcode.net/tradingview/simple-moving-average/
   - Contenu : Tutoriels détaillés et astuces
   - Dernière consultation : 03/11/2025

### 🔍 Tests et Validation
6. **Tests Pratiques BingX (300 klines)**
   - Implémentation testée sur SOL-USDT 5m
   - Validation SMA vs RMA : SMA confirmé comme standard TradingView
   - Date des tests : 03/11/2025

*Document créé le 03/11/2025 - Basé sur recherche TradingView et tests pratiques*
