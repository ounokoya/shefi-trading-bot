# 📊 ACCUMULATION / DISTRIBUTION LINE (ADL) - TRADINGVIEW RESEARCH

## 📋 DÉFINITION

L’**Accumulation/Distribution Line (ADL)** (aussi appelée **A/D**) est un indicateur cumulatif basé sur le volume, qui estime la pression d’accumulation/distribution selon la position du **close** dans la plage **[low, high]** de la bougie.

---

## 🔗 SOURCES TRADINGVIEW STANDARD

### 1. **TradingView Help Center (ADL)**
- **URL** : https://www.tradingview.com/support/solutions/43000501770-accumulation-distribution-adl/
- **Dernière consultation** : 04/01/2026

---

## 🧮 FORMULES MATHÉMATIQUES EXACTES

TradingView donne :
```text
AD = ((Close – Low) – (High – Close)) / (High – Low) * Volume
```

### Décomposition (standard Chaikin)

1. **Money Flow Multiplier (MFM)**
```text
MFM = ((Close - Low) - (High - Close)) / (High - Low)
    = (2*Close - High - Low) / (High - Low)
```

2. **Money Flow Volume (MFV)**
```text
MFV = MFM * Volume
```

3. **Accumulation/Distribution Line (ADL)**
```text
ADL[i] = ADL[i-1] + MFV[i]
```

---

## ⚠️ CAS LIMITES (IMPORTANT)

### Cas `high == low`
Le terme `(high - low)` est au dénominateur.

- En pratique (et pour rester stable), on fixe **`MFM = 0`** quand `high == low`.
- Cela donne **`MFV = 0`** sur cette bougie → l’ADL ne bouge pas.

### Gaps
TradingView rappelle que l’ADL peut se désynchroniser du prix, car la formule ne “voit” pas explicitement les gaps (c’est une limite connue de l’indicateur).

---

## 🎯 INTERPRÉTATION

- **ADL monte** : close plutôt dans la partie haute du range, avec volume → pression acheteuse.
- **ADL baisse** : close plutôt dans la partie basse du range, avec volume → pression vendeuse.
- **Divergences** : prix et ADL évoluent en sens contraire → signal potentiel de retournement.

---

## 🔧 IMPLÉMENTATION PYTHON CONFORME TV

### Fonction du projet
- **Fichier** : `libs/indicators/volume/adl_tv.py`
- **Signature** :
```python
from libs.indicators.volume.adl_tv import adl_tv

adl = adl_tv(high, low, close, volume)
```

### Cas limites gérés
- `high == low` → `MFM = 0`.
- NaN/Inf → propagation en `NaN` (comme une somme cumulative).

---

## ✅ VALIDATION TRADINGVIEW

Pour valider “TV standard” :
- Comparer la série ADL TradingView avec `adl_tv()` sur les mêmes OHLCV.
- Vérifier les bougies `high == low` (sur certains actifs/TF ça arrive) : ADL doit rester stable.

---

## 🎯 POINT CLÉ DE “PRÉCISION” : QUEL VOLUME ? (BASE vs QUOTE)

ADL/A-D est **directement proportionnel** au volume utilisé (`MFV = MFM * Volume`). Si le champ volume n’est pas le même entre TradingView et la source (Bybit), la série ADL divergera numériquement.

### Bybit (v5/market/kline)
- `volume` (champ `[5]`) et `turnover` (champ `[6]`) existent généralement.
- TradingView peut afficher un volume correspondant plutôt à l’un ou l’autre selon le marché.

### Outil de validation du projet
- Script : `indicators_demo.py`
- Option : `--volume-field base|quote`
  - `base` utilise `volume`
  - `quote` utilise `turnover`

---

*Dernière mise à jour : 04/01/2026*
