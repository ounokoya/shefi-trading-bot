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

Signal line (référence d’implémentation de ce repo):

- La signal line est calculée comme une EMA de `KO`:
  - `signal_line = EMA(KO, 13)`

Cette définition est normative pour reproduire exactement les valeurs produites par `libs/indicators/volume/klinger_oscillator_tv.py`.

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
- La formule contient `dm/cm`.
- Dans l’implémentation de ce repo:
  - si `cm` est non valide (NaN/Inf) ou `cm == 0`, alors un facteur temporaire interne vaut `-2.0`.
  - la `VF` est alors calculée normalement avec ce facteur.

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

Règles exactes utilisées par les EMA de ce repo (impact direct sur Klinger):

- Pour une EMA de période `p`, la première valeur possible est à l’index `p-1`.
- Seed: la valeur initiale est la SMA sur les `p` premières valeurs valides de la fenêtre.
- Si une valeur source ou une EMA précédente est non valide (NaN/Inf), l’EMA devient non valide et l’algorithme repasse en mode “non seedé” jusqu’à pouvoir reseeder.

---

## 🧩 Spécification d’implémentation (reproductible, sans ambiguïté)

Entrées:

- Séries de même longueur `n`: `high[i]`, `low[i]`, `close[i]`, `volume[i]`.

Pré-conditions (invalides):

- Une valeur est dite “non valide” si elle est `NaN` ou `Inf`.
- Si à un index `i` une des valeurs nécessaires à l’étape courante est non valide, alors les valeurs intermédiaires (`dm`, `cm`, `vf`) et les sorties (`KO`, `signal_line`) sont non valides à cet index.

Définitions:

- `dm[i] = high[i] - low[i]`.
- `trend[i]`:
  - calculer `s0 = high[i] + low[i] + close[i]` et `s1 = high[i-1] + low[i-1] + close[i-1]`.
  - `trend[i] = +1` si `s0 > s1`, sinon `trend[i] = -1`.
- `cm[i]` (cumulative measurement):
  - soit `prev_trend = trend[i-1]`.
  - soit `prev_dm = dm[i-1]`.
  - soit `prev_cm = cm[i-1]`.
  - si `prev_trend` n’est pas défini (cas initial) alors il est remplacé par `trend[i]`.
  - si `prev_dm` n’est pas défini alors il est remplacé par `dm[i]`.
  - si `prev_cm` n’est pas défini alors il est remplacé par `dm[i]`.
  - si `trend[i] == prev_trend` alors `cm[i] = prev_cm + dm[i]`, sinon `cm[i] = prev_dm + dm[i]`.
- Facteur VF:
  - si `cm[i]` est non valide ou `cm[i] == 0`:
    - `temp = -2.0`
  - sinon:
    - `raw = 2 * ((dm[i] / cm[i]) - 1)`
    - si `vf_use_abs_temp == true` alors `temp = abs(raw)`, sinon `temp = raw`.
  - `vf[i] = volume[i] * trend[i] * temp * 100`.

Sorties:

- `ema_fast = EMA(vf, fast)`
- `ema_slow = EMA(vf, slow)`
- `KO[i] = ema_fast[i] - ema_slow[i]` quand les 2 sont valides.
- `signal_line = EMA(KO, signal)`.

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
