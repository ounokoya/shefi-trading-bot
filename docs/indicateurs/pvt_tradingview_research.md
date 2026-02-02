# 📊 PRICE VOLUME TREND (PVT) - TRADINGVIEW RESEARCH

## 📋 DÉFINITION

Le **Price Volume Trend (PVT)** est un indicateur cumulatif volume-prix. Il ressemble à l’OBV, mais au lieu d’ajouter/soustraire tout le volume, il pondère le volume par la variation relative du prix.

---

## 🔗 SOURCE TRADINGVIEW STANDARD

- **TradingView Help Center (PVT)**
- **URL** : https://www.tradingview.com/support/solutions/43000502345-price-volume-trend-pvt/

---

## FORMULE EXACTE

Soit `PVT[i]`, `close[i]`, `volume[i]`.

- Initialisation (implémentation de ce repo):
  - `PVT[0] = 0.0` si `close[0]` est valide.
  - sinon `PVT[0]` est non valide.
- Pour `i >= 1` :

  - Si `PVT[i-1]`, `close[i]`, `close[i-1]` ou `volume[i]` est non valide:
    - `PVT[i]` est non valide.
  - Sinon si `close[i-1] == 0`:
    - `PVT[i]` est non valide.
  - Sinon:
    - `PVT[i] = PVT[i-1] + volume[i] × (close[i] - close[i-1]) / close[i-1]`.

---

## IMPLÉMENTATION PYTHON CONFORME TV

- **Fichier** : `libs/indicators/volume/pvt_tv.py`
- **Sortie** : une série `pvt[i]` de longueur `n`.

### Cas limites gérés

- Si `close[i-1] == 0` : retour `NaN` (division par zéro)
- Propagation `NaN/Inf` si une valeur requise est invalide, incluant:
  - `close[i]`, `close[i-1]`, `volume[i]`,
  - et `PVT[i-1]` (si l’état précédent est non valide, `PVT[i]` devient non valide).

---

*Dernière mise à jour : 22/01/2026*
