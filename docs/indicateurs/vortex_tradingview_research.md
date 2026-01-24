# 🌀 VORTEX INDICATOR (VI) - TRADINGVIEW RESEARCH

## 📋 DÉFINITION

Le **Vortex Indicator (VI)** est un indicateur technique développé par Etienne Botes et Douglas Siepman en 2009, conçu pour identifier les débuts de nouvelles tendances et les inversions de tendance existantes. Il se compose de deux lignes oscillantes : **VI+** (mouvement positif) et **VI-** (mouvement négatif).

---

## 🔗 SOURCES TRADINGVIEW STANDARD

### 1. **TradingView Pine Script Built-in**
- **URL** : https://www.tradingview.com/pine-script-docs/#ta_vi
- **Fonction** : `ta.vi(length)`
- **Description** : Implémentation officielle TradingView
- **Dernière consultation** : 03/01/2026

### 2. **Investopedia - Vortex Indicator**
- **URL** : https://www.investopedia.com/terms/v/vortex-indicator-vi.asp
- **Contenu** : Formules originales et calculs détaillés
- **Dernière consultation** : 03/01/2026

### 3. **Pine Script Vortex Indicator Guide**
- **URL** : https://offline-pixel.github.io/pinescript-strategies/pine-script-VortexIndicator.html
- **Contenu** : Implémentation complète et exemples
- **Dernière consultation** : 03/01/2026

### 4. **TradingView Scripts - Vortex Implementations**
- **URL** : https://www.tradingview.com/scripts/vortex/
- **Contenu** : Scripts communautaires et variantes
- **Dernière consultation** : 03/01/2026

---

## 🧮 FORMULES MATHÉMATIQUES EXACTES

### ÉTAPE 1: CALCUL DU TRUE RANGE (TR)
Pour chaque période *i* :
```
TR_i = max(
    high_i - low_i,
    abs(high_i - close_{i-1}),
    abs(low_i - close_{i-1})
)
```

### ÉTAPE 2: CALCUL DES VORTEX MOVEMENTS
```
VM+_i = abs(high_i - low_{i-1})    # Mouvement positif
VM-_i = abs(low_i - high_{i-1})     # Mouvement négatif
```

### ÉTAPE 3: SOMMES SUR PÉRIODE *n* (généralement 14)
```
SUM_TR_n = Σ(TR_i) sur les n dernières périodes
SUM_VM+_n = Σ(VM+_i) sur les n dernières périodes  
SUM_VM-_n = Σ(VM-_i) sur les n dernières périodes
```

### ÉTAPE 4: CALCUL FINAL DES LIGNES VI
```
VI+_n = SUM_VM+_n / SUM_TR_n
VI-_n = SUM_VM-_n / SUM_TR_n
```

---

## 📊 PARAMÈTRES TRADINGVIEW STANDARD

| Paramètre | Valeur par défaut | Plage recommandée | Description |
|-----------|------------------|-------------------|-------------|
| Length | **14** | 14-30 | Période de calcul |
| Source | OHLC | - | Données OHLC standard |
| VI+ Color | Vert | - | Ligne tendance haussière |
| VI- Color | Rouge | - | Ligne tendance baissière |

---

## 🎯 SIGNAUX ET INTERPRÉTATION

### SIGNAUX D'ACHAT
- **Croisement haussier** : VI+ passe au-dessus de VI-
- **Confirmation** : VI+ reste au-dessus de VI-
- **Force tendance** : VI+ > 1.0

### SIGNAUX DE VENTE  
- **Croisement baissier** : VI- passe au-dessus de VI+
- **Confirmation** : VI- reste au-dessus de VI+
- **Force tendance** : VI- > 1.0

### ZONES NEUTRES
- **0.8 - 1.0** : Zone de transition
- **< 0.8** : Faible tendance
- **> 1.2** : Forte tendance

---

## 🔧 IMPLÉMENTATION PYTHON CONFORME TV

### STRUCTURE DE FONCTION
```python
def vortex_tv(high: Sequence[float], low: Sequence[float], 
              close: Sequence[float], period: int) -> Tuple[List[float], List[float]]:
    """
    Calcul Vortex Indicator conforme TradingView
    
    Args:
        high: Prix hauts
        low: Prix bas  
        close: Prix de clôture
        period: Période de calcul (défaut 14)
    
    Returns:
        Tuple (VI_plus, VI_minus): Deux listes de valeurs
    """
```

### GESTION DES CAS LIMITES
- **Premières périodes** : NaN comme TradingView
- **Valeurs nulles** : Propagation correcte des NaN
- **Validation inputs** : Longueurs égales requises

---

## 📈 EXEMPLES D'UTILISATION

### CONFIGURATION CLASSIQUE
```python
# Paramètres TradingView par défaut
vi_plus, vi_minus = vortex_tv(high, low, close, 14)

# Détection croisements
buy_signal = (vi_plus[-1] > vi_minus[-1]) and (vi_plus[-2] <= vi_minus[-2])
sell_signal = (vi_minus[-1] > vi_plus[-1]) and (vi_minus[-2] <= vi_plus[-2])
```

### COMBINAISON AVEC AUTRES INDICATEURS
- **MACD** : Confirmation momentum
- **RSI** : Zones surachat/survente
- **Volumes** : Validation force tendance

---

## ⚠️ POINTS D'ATTENTION

### FAUX SIGNAUX
- **Marchés latéraux** : Croisements fréquents sans tendance
- **Volatilité extrême** : Fausse force tendance
- **Périodes courtes** : Plus de bruit, moins de fiabilité

### OPTIMISATION RECOMMANDÉE
- **Augmenter période** : Réduire faux signaux (ex: 25 au lieu de 14)
- **Filtres additionnels** : Confirmations multi-timeframes
- **Volumes** : Validation des croisements

---

## 📚 RÉFÉRENCES COMPLÉMENTAIRES

### THÉORIE ORIGINALE
- **Botes & Siepman (2009)** : "The Vortex Indicator"
- **Technical Analysis of Stocks & Commodities** : Article fondateur

### APPLICATIONS PRATIQUES
- **Swing Trading** : Identification retournements tendance
- **Trend Following** : Confirmation force momentum
- **Risk Management** : Sorties de position optimisées

---

## ✅ VALIDATION TRADINGVIEW

Pour garantir une précision 100% TradingView :

1. **Utiliser formules exactes** ci-dessus
2. **Paramètres par défaut** : length=14
3. **Gestion NaN** : Premières `period-1` valeurs = NaN
4. **Tests comparatifs** : vs Pine Script `ta.vi()`

---

*Dernière mise à jour : 03/01/2026*  
*Précision visée : 100% TradingView Standard*
