# 📊 POSITIVE VOLUME INDEX (PVI) - TRADINGVIEW RESEARCH

## 📋 DÉFINITION

Le **Positive Volume Index (PVI)** est l’analogue du NVI : il met à jour sa valeur uniquement lorsque le volume augmente par rapport à la veille.

---

## 🔗 SOURCE TRADINGVIEW STANDARD

TradingView ne fournit pas un Help Center officiel unique, mais les scripts TradingView qui reproduisent le standard utilisent la règle suivante :

- Mise à jour uniquement si `volume[i] > volume[i-1]`

---

## 🧮 FORMULE EXACTE

Soit `PVI[i]`, `close[i]`, `volume[i]`.

- Initialisation : `PVI[0] = 1000`
- Pour `i >= 1` :

```text
if volume[i] > volume[i-1]:
    PVI[i] = PVI[i-1] + PVI[i-1] * (close[i] - close[i-1]) / close[i-1]
else:
    PVI[i] = PVI[i-1]
```

---

## 🔧 IMPLÉMENTATION PYTHON CONFORME TV

- **Fichier** : `libs/indicators/volume/pvi_tv.py`
- **Signature** :

```python
from libs.indicators.volume.pvi_tv import pvi_tv

pvi = pvi_tv(close, volume, start=1000.0)
```

### Cas limites gérés

- Si `close[i-1] == 0` : retour `NaN`
- Propagation `NaN/Inf` si une valeur requise est invalide

---

*Dernière mise à jour : 22/01/2026*
