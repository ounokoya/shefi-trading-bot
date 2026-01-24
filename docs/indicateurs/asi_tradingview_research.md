# Accumulative Swing Index (ASI) - Documentation TradingView

## Overview

L'Accumulative Swing Index (ASI) est un indicateur technique développé par J. Welles Wilder Jr. et présenté dans son livre "New Concepts in Technical Trading Systems" (1978). L'ASI est conçu pour isoler les "véritables" mouvements de prix en comparant les relations entre les prix actuels (open, high, low, close) et ceux de la période précédente.

## Formule Mathématique

### Swing Index (SI)

La formule du Swing Index est :

```
SI = 50 × [(C - Cy + 0.5(C - O) + 0.25(Cy - Oy)) / R] × K/T
```

Où :
- **C** : Prix de clôture actuel
- **Cy** : Prix de clôture précédent  
- **O** : Prix d'ouverture actuel
- **Oy** : Prix d'ouverture précédent
- **H** : Plus haut actuel
- **L** : Plus bas actuel
- **Hy** : Plus haut précédent
- **Ly** : Plus bas précédent
- **T** : Variation maximale de prix pour la période (limit up/down)
- **K** : max(H - Cy, Cy - L)
- **R** : Variable calculée à partir de TR/ER/SH

### Calcul de R

Le calcul de R est :

```
R = TR - 0.5 * ER + 0.25 * SH
```

Où :

```
TR = max(H - Cy, Cy - L, H - L)
SH = Cy - Oy
ER =
  H - Cy   si Cy > H
  0       si L <= Cy <= H
  Cy - L  si Cy < L
```

### Accumulative Swing Index (ASI)

```
ASI_t = ASI_{t-1} + SI_t
```

L'ASI est la somme cumulée des valeurs du Swing Index.

## Paramètres

### Limit Move Value (T)
- **Définition** : T représente le « limit move » (variation maximale attendue/autorisé sur la session). Dans de nombreuses implémentations (incluant des scripts TradingView), T est une valeur saisie par l'utilisateur.
- **Important** : La valeur de T change l'échelle de l'ASI. Un T trop petit amplifie SI/ASI.

## Interprétation

### Signaux de Tendance
- **ASI croissant** : Tendance haussière confirmée
- **ASI décroissant** : Tendance baissière confirmée
- **ASI plat** : Marché sans tendance claire

### Niveaux Extremes
- **ASI > +100** : Surachat potentiel
- **ASI < -100** : Survente potentiel

### Signaux de Breakout
- **Fracture de ligne de tendance ASI** : Signal possible de changement de tendance
- **Divergence prix/ASI** : Signal de retournement potentiel

## Avantages
1. **Filtre de bruit** : Élimine les mouvements de prix non significatifs
2. **Trend-following** : Excellent pour suivre les tendances établies
3. **Universel** : Fonctionne sur tous les marchés et timeframes
4. **Précision** : Basé sur les 4 prix OHLC

## Inconvénients
1. **Lag** : Indicateur retardé comme la plupart des trend-followers
2. **Paramétrage** : Le choix de T est crucial et dépend du marché
3. **Complexité** : Calcul plus complexe que les indicateurs basiques

## TradingView Implementation

La précision TradingView nécessite :
1. **Gestion des arrondis** : Utiliser la précision float64
2. **Shift correct** : Utiliser .shift(1) pour les valeurs précédentes
3. **Gestion NaN** : Pour l'accumulation (ASI), appliquer un comportement type `nz()` (les valeurs non calculables de SI doivent contribuer 0 à l'accumulation)
4. **Division par zéro** : Protéger les divisions sur R et T (si R==0 ou T==0, SI=0)

## Usage Recommandé

### Timeframes
- **Daily/Weekly** : Pour l'analyse de tendance long terme
- **4H/1H** : Pour le swing trading
- **15M/5M** : Pour le scalping (avec T ajusté)

### Combinaisons
- **ASI + Trend lines** : Pour identifier les breakouts
- **ASI + Volume** : Pour confirmer les mouvements
- **ASI + Support/Resistance** : Pour renforcer les niveaux

## Exemple d'Utilisation

```python
# Paramètres pour crypto
limit_move_pct = 0.10  # 10% pour les cryptomonnaies

# Signaux
buy_signal = df['ASI'] > df['ASI'].shift(1) and df['ASI'] > 0
sell_signal = df['ASI'] < df['ASI'].shift(1) and df['ASI'] < 0
```

## Précision vs TradingView

Pour correspondre à TradingView :
1. Utiliser `float64` pour tous les calculs
2. Appliquer les mêmes règles de gestion des NaN
3. Utiliser `cumsum(skipna=True)` pour l'accumulation
4. Paramétrer T selon les spécifications de TradingView

## Références

- Wilder, J. Welles (1978). "New Concepts in Technical Trading Systems"
- TradingView Pine Script Reference
- QuantConnect LEAN Implementation
