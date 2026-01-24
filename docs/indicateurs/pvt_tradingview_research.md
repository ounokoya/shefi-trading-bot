# 📊 PRICE VOLUME TREND (PVT) - TRADINGVIEW RESEARCH

## 📋 DÉFINITION

Le **Price Volume Trend (PVT)** est un indicateur cumulatif volume-prix. Il ressemble à l’OBV, mais au lieu d’ajouter/soustraire tout le volume, il pondère le volume par la variation relative du prix.

---

## 🔗 SOURCE TRADINGVIEW STANDARD

- **TradingView Help Center (PVT)**
- **URL** : https://www.tradingview.com/support/solutions/43000502345-price-volume-trend-pvt/

---

## 🧮 FORMULE EXACTE

Soit `PVT[i]`, `close[i]`, `volume[i]`.

- Initialisation : `PVT[0] = 0`
- Pour `i >= 1` :

```text
PVT[i] = PVT[i-1] + volume[i] * (close[i] - close[i-1]) / close[i-1]
```

---

## 🔧 IMPLÉMENTATION PYTHON CONFORME TV

- **Fichier** : `libs/indicators/volume/pvt_tv.py`
- **Signature** :

```python
from libs.indicators.volume.pvt_tv import pvt_tv

pvt = pvt_tv(close, volume)
```

### Cas limites gérés

- Si `close[i-1] == 0` : retour `NaN` (division par zéro)
- Propagation `NaN/Inf` si une valeur requise est invalide

---

*Dernière mise à jour : 22/01/2026*
