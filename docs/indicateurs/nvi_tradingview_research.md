# 📊 NEGATIVE VOLUME INDEX (NVI) - TRADINGVIEW RESEARCH

## 📋 DÉFINITION

Le **Negative Volume Index (NVI)** est un indice cumulatif qui met à jour sa valeur uniquement lorsque le volume baisse par rapport à la veille. La théorie classique : les « smart money » agiraient davantage quand le volume est plus faible.

---

## 🔗 SOURCE TRADINGVIEW STANDARD

TradingView ne fournit pas un Help Center officiel unique comme pour OBV/PVT, mais les scripts TradingView qui reproduisent le standard utilisent la règle suivante :

- Mise à jour uniquement si `volume[i] < volume[i-1]`

---

## 🧮 FORMULE EXACTE

Soit `NVI[i]`, `close[i]`, `volume[i]`.

- Initialisation : `NVI[0] = 1000`
- Pour `i >= 1` :

```text
if volume[i] < volume[i-1]:
    NVI[i] = NVI[i-1] + NVI[i-1] * (close[i] - close[i-1]) / close[i-1]
else:
    NVI[i] = NVI[i-1]
```

---

## 🔧 IMPLÉMENTATION PYTHON CONFORME TV

- **Fichier** : `libs/indicators/volume/nvi_tv.py`
- **Signature** :

```python
from libs.indicators.volume.nvi_tv import nvi_tv

nvi = nvi_tv(close, volume, start=1000.0)
```

### Cas limites gérés

- Si `close[i-1] == 0` : retour `NaN`
- Propagation `NaN/Inf` si une valeur requise est invalide

---

*Dernière mise à jour : 22/01/2026*
