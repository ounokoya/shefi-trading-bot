# 📊 ON BALANCE VOLUME (OBV) - TRADINGVIEW RESEARCH

## 📋 DÉFINITION

L’**On Balance Volume (OBV)** est un indicateur cumulatif qui additionne ou soustrait le volume selon le sens de variation du prix de clôture. L’objectif est de mesurer la pression d’achat/vente et d’identifier confirmations de tendance et divergences.

---

## 🔗 SOURCES TRADINGVIEW STANDARD

### 1. **TradingView Help Center (OBV)**
- **URL** : https://www.tradingview.com/support/solutions/43000502593-on-balance-volume-obv/
- **Dernière consultation** : 04/01/2026

---

## 🧮 FORMULES MATHÉMATIQUES EXACTES

### Règles de mise à jour
Soit `OBV[i]` la valeur à l’instant `i`, `close[i]` la clôture, `volume[i]` le volume.

1. Si `close[i] > close[i-1]` :
```text
OBV[i] = OBV[i-1] + volume[i]
```

2. Si `close[i] < close[i-1]` :
```text
OBV[i] = OBV[i-1] - volume[i]
```

3. Si `close[i] == close[i-1]` :
```text
OBV[i] = OBV[i-1]
```

### Initialisation
TradingView décrit la règle “previous OBV +/− volume”. Pour la première bougie (`i=0`), il n’existe pas de `close[-1]`. L’OBV est **défini à une constante initiale**.

- En pratique, **`OBV[0] = 0`** est un choix standard.
- Toute autre constante donnerait la même courbe à un décalage vertical près (les signaux basés sur variations/dérivées ne changent pas).

---

## 📊 PARAMÈTRES TRADINGVIEW STANDARD

OBV n’a pas de paramètre de période (c’est une somme cumulative). TradingView peut proposer une section *Smoothing* (lissage) mais l’OBV “brut” reste la série cumulée ci-dessus.

---

## 🎯 INTERPRÉTATION

- **OBV monte** : le volume s’accumule sur des bougies haussières → pression acheteuse.
- **OBV baisse** : le volume s’accumule sur des bougies baissières → pression vendeuse.
- **Divergence** :
  - **Bullish divergence** : prix baisse mais OBV monte.
  - **Bearish divergence** : prix monte mais OBV baisse.

---

## 🔧 IMPLÉMENTATION PYTHON CONFORME TV

### Fonction du projet
- **Fichier** : `libs/indicators/volume/obv_tv.py`
- **Signature** :
```python
from libs.indicators.volume.obv_tv import obv_tv

obv = obv_tv(close, volume)
```

### Cas limites gérés
- **NaN/Inf** : propagation en `NaN`.
- **Initialisation** : `OBV[0] = 0.0`.

---

## ✅ VALIDATION TRADINGVIEW

Pour valider “TV standard” :
- Vérifier que les règles de signe sont exactement celles du Help Center.
- Comparer une série de prix/volume exportée (mêmes OHLCV) entre TradingView et `obv_tv()`.
- Tolérance : un éventuel **décalage constant** (initialisation) ne change pas les deltas.

---

## 🎯 POINT CLÉ DE “PRÉCISION” : QUEL VOLUME ? (BASE vs QUOTE)

OBV est **directement proportionnel** au volume utilisé. Si deux plateformes n’utilisent pas le même champ volume (base vs quote), les valeurs seront différentes même si la formule est correcte.

### TradingView
- Sur crypto, TradingView affiche en général un volume “bar” lié au marché du broker/exchange.
- Selon la source de données, ce volume peut correspondre à :
  - **volume en base** (ex: BTC),
  - ou **volume en quote/turnover** (ex: USDT),
  - ou un volume “contracts” sur certains dérivés.

### Bybit (v5/market/kline)
- La réponse contient typiquement :
  - `volume` (champ `[5]`) : volume,
  - `turnover` (champ `[6]`) : turnover.
- Pour les comparaisons avec TradingView, il faut choisir **le champ qui correspond à ce que TradingView affiche pour ce marché**.

### Outil de validation du projet
- Script : `indicators_demo.py`
- Option : `--volume-field base|quote`
  - `base` utilise le champ Bybit `volume`
  - `quote` utilise le champ Bybit `turnover`

---

*Dernière mise à jour : 04/01/2026*
