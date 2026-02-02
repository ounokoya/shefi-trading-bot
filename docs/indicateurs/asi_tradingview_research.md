# Accumulative Swing Index (ASI) - Documentation TradingView

## Overview

L'Accumulative Swing Index (ASI) est un indicateur technique développé par J. Welles Wilder Jr. et présenté dans son livre "New Concepts in Technical Trading Systems" (1978). L'ASI est conçu pour isoler les "véritables" mouvements de prix en comparant les relations entre les prix actuels (open, high, low, close) et ceux de la période précédente.

## Formule Mathématique

### Swing Index (SI)

Cette documentation décrit la logique utilisée par l’implémentation de référence du repo (`libs/indicators/asi.py`).

Définition:

- Le Swing Index `SI[i]` est calculé à partir des valeurs de la bougie `i` et des valeurs de la bougie précédente `i-1`.
- L’Accumulative Swing Index `ASI[i]` est la somme cumulée des `SI`.

Formule utilisée (non ambiguë):

- `SI[i] = 50 × (numerator[i] / R[i]) × (K[i] / T[i])`, puis `SI[i]` est forcé à `0.0` si non calculable.

Où :
- `high[i]`, `low[i]`, `close[i]`, `open[i]` sont les OHLC de la bougie courante.
- `high_prev = high[i-1]`, `low_prev = low[i-1]`, `close_prev = close[i-1]`, `open_prev = open[i-1]`.
- `T[i]` est le “limit move value”.
- `K[i]` et `R[i]` sont définis ci-dessous.

### Spécification d’implémentation (reproductible, sans ambiguïté)

Entrées:

- Séries de même longueur `n`: `high[i]`, `low[i]`, `close[i]`, `open[i]`.

Règles de validité:

- Une valeur est dite “non valide” si elle est `NaN` ou `Inf`.
- Pour `i == 0`, les valeurs précédentes sont non définies et `SI[0]` est `0.0`.
- Pour `i > 0`, si un terme nécessaire au calcul de `SI[i]` est non valide, alors `SI[i]` est `0.0`.

Définition de `T[i]` (limit move value):

- Si `limit_move_value == "auto"`:
  - `T[i] = close_prev × limit_move_pct`.
- Sinon:
  - `T[i] = limit_move_value` (constante).

Définitions intermédiaires:

- `a[i] = abs(high[i] - close_prev)`.
- `b[i] = abs(low[i] - close_prev)`.
- `c[i] = abs(high[i] - low[i])`.
- `sh[i] = abs(close_prev - open_prev)`.
- `K[i] = max(a[i], b[i])`.

Définition de `R[i]` (non ambiguë):

- Si `a[i] >= b[i]` et `a[i] >= c[i]`:
  - `R[i] = a[i] - 0.5 × b[i] + 0.25 × sh[i]`.
- Sinon, si `b[i] >= a[i]` et `b[i] >= c[i]`:
  - `R[i] = b[i] - 0.5 × a[i] + 0.25 × sh[i]`.
- Sinon:
  - `R[i] = c[i] + 0.25 × sh[i]`.

Définition de `numerator[i]`:

- `numerator[i] = (close_prev - close[i]) + 0.5 × (close_prev - open_prev) + 0.25 × (close[i] - open[i])`.

Calcul de `SI[i]`:

- Si `R[i] == 0` ou `T[i] == 0`, alors `SI[i] = 0.0`.
- Sinon:
  - `SI[i] = 50 × (numerator[i] / R[i]) × (K[i] / T[i])`.

Calcul de `ASI[i]`:

- `ASI[i]` est la somme cumulée des `SI`.

### Accumulative Swing Index (ASI)

À chaque index `i`, `ASI[i]` est égal à `ASI[i-1] + SI[i]` (et `ASI[0]` est égal à `SI[0]`).

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
1. **Précision float64** sur tous les calculs.
2. **Décalage exact** des valeurs précédentes (`i-1`).
3. **Comportement type `nz()`** pour `SI`: si `SI[i]` n’est pas calculable, il vaut `0.0` et l’accumulation continue.
4. **Division par zéro** : si `R[i]==0` ou `T[i]==0`, alors `SI[i]=0.0`.

## Usage Recommandé

### Timeframes
- **Daily/Weekly** : Pour l'analyse de tendance long terme
- **4H/1H** : Pour le swing trading
- **15M/5M** : Pour le scalping (avec T ajusté)

### Combinaisons
- **ASI + Trend lines** : Pour identifier les breakouts
- **ASI + Volume** : Pour confirmer les mouvements
- **ASI + Support/Resistance** : Pour renforcer les niveaux

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
