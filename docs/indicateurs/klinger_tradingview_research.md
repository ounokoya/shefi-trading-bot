# 📊 KLINGER OSCILLATOR / KLINGER VOLUME OSCILLATOR (KVO) - TRADINGVIEW RESEARCH

## 📋 DÉFINITION

Le **Klinger Oscillator** (souvent appelé **Klinger Volume Oscillator / KVO**) vise à identifier la tendance long-terme du *money flow* tout en restant sensible aux fluctuations court-terme.

TradingView le décrit comme un indicateur comparant le mouvement de prix au volume, puis transformant ce résultat en oscillateur basé sur la différence de deux moyennes mobiles appliquées à une série de **Volume Force (VF)**.

---

## 🔗 SOURCES TRADINGVIEW STANDARD

### 1. **TradingView Help Center — Klinger Oscillator**
- **URL** : https://www.tradingview.com/support/solutions/43000589157-klinger-oscillator/
- **Contenu** : définition + formules exactes (VF, Trend, dm/cm) + périodes standard (34/55) + signal line (13)
- **Dernière consultation** : 16/01/2026

### 2. **TradingView Pine Script Reference Manual**
- **URL** : https://www.tradingview.com/pine-script-reference/v6/
- **Contenu** : fonctions nécessaires à une implémentation manuelle TradingView (ex: `ta.ema()`)
- **Dernière consultation** : 16/01/2026

---

## 🧮 FORMULES MATHÉMATIQUES EXACTES (TRADINGVIEW)

TradingView (Help Center) donne les étapes et définitions suivantes.

### 1) Trend (T)
Pour chaque période *i* :

- **Trend = +1** si :
  - `(H[i] + L[i] + C[i]) > (H[i-1] + L[i-1] + C[i-1])`
- **Trend = -1** sinon (`<=`).

Où :
- `H` = High
- `L` = Low
- `C` = Close

### 2) dm
- `dm[i] = H[i] - L[i]`

### 3) cm
TradingView:
- `cm[i] = cm[i-1] + dm[i]` si `Trend[i] == Trend[i-1]`
- `cm[i] = dm[i-1] + dm[i]` si `Trend[i] != Trend[i-1]`

Notes:
- Pour la première valeur de `cm`, si `cm[i-1]` n’existe pas, utiliser `dm` (ou démarrer avec `cm = dm`).

### 4) Volume Force (VF)
TradingView:

- `VF = V × [2 × ((dm/cm) − 1)] × T × 100`

Où :
- `V` = volume
- `T` = trend (+1 / -1)

### 5) Klinger Oscillator (KO / KVO)
TradingView:

- `KO = EMA(VF, 34) − EMA(VF, 55)`

Les périodes les plus courantes sont **34** et **55**.

---

## 📈 SIGNAL LINE (TRADINGVIEW)

TradingView indique :
- Une **13-period moving average** est typiquement utilisée comme **signal line**.

⚠️ TradingView (Help Center) ne précise pas ici le type exact (SMA vs EMA) dans le texte.

### Recommandation “précision TradingView”
- Implémenter et valider par comparaison directe avec TradingView:
  - Variante A: `signal = EMA(KO, 13)`
  - Variante B: `signal = SMA(KO, 13)`
- Conserver la variante qui matche exactement la courbe TradingView sur un même OHLCV.

---

## ⚙️ PARAMÈTRES TRADINGVIEW STANDARD

- **Fast EMA (VF)** : 34
- **Slow EMA (VF)** : 55
- **Signal length** : 13

---

## ⚠️ CAS LIMITES / POINTS DE PRÉCISION

### 1) `i == 0`
- `Trend` nécessite `i-1`.
- `cm` nécessite `cm[i-1]` et parfois `dm[i-1]`.

### 2) Division par zéro
- La formule contient `dm/cm`. Si `cm == 0`, il faut retourner `na` (ou une convention stable) et **revalider vs TradingView**.

### 3) Volume “base vs quote”
Comme rappelé dans `docs/indicateurs/indicateur_precision_rules.md`:
- Les indicateurs volume-dépendants sont sensibles à la définition du volume.
- En crypto (Bybit/Binance), on peut avoir un volume base et un turnover quote.

Objectif “100% TradingView”:
- Comparer avec TradingView et déterminer si TradingView correspond au volume base ou quote sur le marché choisi.

### 4) EMA “TradingView compatible”
Dans ce repo, l’EMA de référence est documentée dans:
- `docs/indicateurs/ema_tradingview_research.md`

Et implémentée dans:
- `libs/indicators/moving_averages/ema_tv.py`

Car TradingView utilise:
- seed SMA
- lazy seeding / reseeding après invalid values

---

## 🧩 FONCTIONS PINE SCRIPT À UTILISER (IMPLÉMENTATION MANUELLE)

TradingView fournit le KO comme indicateur, mais pour une reproduction exacte dans Pine, les briques nécessaires sont :

- `ta.ema(src, length)`
- `ta.sma(src, length)` (si la signal line est SMA)

Données:
- `high`, `low`, `close`, `volume`

Variables d’état:
- `cm` doit être maintenu d’un bar à l’autre (via `var float cm = na`)
- `trend` est défini par la comparaison entre la somme `(H+L+C)` courante et précédente.

---

## 🔧 RECOMMANDATION D’IMPLÉMENTATION PYTHON (STYLE DU PROJET)

### Fichier cible (à créer quand nécessaire)
- `libs/indicators/volume/klinger_oscillator_tv.py`

### Dépendances internes à réutiliser
- `libs/indicators/moving_averages/ema_tv.py` (EMA TradingView)
- `libs/indicators/common/is_bad.py` (gestion NaN/Inf)

### API suggérée
- `klinger_oscillator_tv(high, low, close, volume, fast=34, slow=55, signal=13) -> (ko, signal_line)`

---

## ✅ VALIDATION TRADINGVIEW

Procédure recommandée:
- Choisir un actif/TF.
- Exporter/aligner les OHLCV.
- Calculer KO (et signal line) côté Python.
- Comparer aux valeurs TradingView:
  - même `volume` (base/quote)
  - mêmes timestamps (open time)
  - mêmes conventions EMA (seed SMA)

---

## 📚 SOURCES ET RÉFÉRENCES

### 📖 Documentation Officielle
1. **TradingView Support - Klinger Oscillator**
   - URL : https://www.tradingview.com/support/solutions/43000589157-klinger-oscillator/
   - Dernière consultation : 16/01/2026

### 📚 Guides et Tutoriels (complémentaires)
2. **Investopedia - Klinger Oscillator**
   - URL : https://www.investopedia.com/terms/k/klingeroscillator.asp
   - Contenu : définition, interprétation, rappels formules
   - Dernière consultation : 16/01/2026

### 🔍 Références Historiques
3. **Stephen J. Klinger**
   - Créateur original du Klinger Oscillator

---

*Dernière mise à jour : 16/01/2026*
