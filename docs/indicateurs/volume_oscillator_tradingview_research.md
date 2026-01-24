# 📊 VOLUME OSCILLATOR - TRADINGVIEW RESEARCH

## 📋 DÉFINITION

Le **Volume Oscillator** est un indicateur technique qui mesure la différence entre deux moyennes mobiles du volume, aidant les traders à identifier les tendances de volume et à confirmer les mouvements de prix. Il existe deux variantes principales : **Volume Oscillator (différence)** et **Percentage Volume Oscillator (PVO)**.

---

## 🔗 SOURCES TRADINGVIEW STANDARD

### 1. **TradingView Pine Script Built-in**
- **URL** : https://www.tradingview.com/pine-script-docs/#ta_sma
- **Fonction** : `ta.sma(source, length)`
- **Description** : Base de calcul du Volume Oscillator
- **Dernière consultation** : 03/01/2026

### 2. **Pine Script Volume Oscillator Guide**
- **URL** : https://offline-pixel.github.io/pinescript-strategies/pine-script-volume-oscillator.html
- **Contenu** : Implémentation complète et exemples
- **Dernière consultation** : 03/01/2026

### 3. **Stack Overflow - Volume Oscillator Source**
- **URL** : https://stackoverflow.com/questions/73269509/editing-built-in-volume-oscillator
- **Contenu** : Code source exact built-in TradingView
- **Dernière consultation** : 03/01/2026

### 4. **TradingView Scripts - Volume Oscillator**
- **URL** : https://www.tradingview.com/scripts/volume/
- **Contenu** : Scripts communautaires et variantes
- **Dernière consultation** : 03/01/2026

### 5. **Percentage Volume Oscillator (PVO)**
- **URL** : https://www.tradingview.com/support/solutions/43000591350-percentage-volume-oscillator-pvo/
- **Contenu** : Formule PVO exacte TradingView
- **Dernière consultation** : 03/01/2026

---

## 🧮 FORMULES MATHÉMATIQUES EXACTES

### VOLUME OSCILLATOR (DIFFÉRENCE)
```python
# Moyennes mobiles du volume
fastMA = SMA(volume, fast_length)   # défaut: 10
slowMA = SMA(volume, slow_length)   # défaut: 30

# Oscillateur (différence)
oscillator = fastMA - slowMA
```

### PERCENTAGE VOLUME OSCILLATOR (PVO)
```python
# Moyennes mobiles du volume
fastMA = SMA(volume, fast_length)   # défaut: 10
slowMA = SMA(volume, slow_length)   # défaut: 30

# Oscillateur pourcentage
PVO = ((fastMA - slowMA) / slowMA) * 100
```

### LIGNE DE SIGNAL (OPTIONNEL)
```python
# Lissage de l'oscillateur
signal_line = SMA(PVO, signal_length)  # défaut: 9
histogram = PVO - signal_line
```

---

## 📊 PARAMÈTRES TRADINGVIEW STANDARD

| Paramètre | Valeur par défaut | Plage recommandée | Description |
|-----------|------------------|-------------------|-------------|
| Fast Length | **10** | 5-20 | Période MA rapide |
| Slow Length | **30** | 20-50 | Période MA lente |
| Signal Length | **9** | 5-15 | Période signal (PVO) |
| MA Type | **SMA** | SMA/EMA | Type moyenne mobile |
| Output | Différence | Diff/Pct | Format sortie |

---

## 🎯 SIGNAUX ET INTERPRÉTATION

### VOLUME OSCILLATOR (DIFFÉRENCE)
- **Positif** : Volume récent > volume moyen (pression haussière)
- **Négatif** : Volume récent < volume moyen (pression baissière)
- **Zéro** : Volume en équilibre

### PERCENTAGE VOLUME OSCILLATOR (PVO)
- **> 0** : Volume rapide > volume lent (momentum volume positif)
- **< 0** : Volume rapide < volume lent (momentum volume négatif)
- **Croisements** : Changements de tendance volume

### SIGNAUX DE CONFIRMATION
- **Prix + Volume** : Tendance validée
- **Divergence Prix/Volume** : Possible inversion
- **Volume extrême** : Force tendance maximale

---

## 🔧 IMPLÉMENTATION PYTHON CONFORME TV

### STRUCTURE DE FONCTION
```python
def volume_oscillator_tv(volume: Sequence[float], fast: int, slow: int) -> List[float]:
    """
    Calcul Volume Oscillator conforme TradingView
    
    Args:
        volume: Série des volumes
        fast: Période moyenne rapide (défaut 10)
        slow: Période moyenne lente (défaut 30)
    
    Returns:
        Liste des valeurs de l'oscillateur
    """

def percentage_volume_oscillator_tv(volume: Sequence[float], fast: int, slow: int) -> List[float]:
    """
    Calcul Percentage Volume Oscillator (PVO) conforme TradingView
    
    Args:
        volume: Série des volumes
        fast: Période moyenne rapide (défaut 10)  
        slow: Période moyenne lente (défaut 30)
    
    Returns:
        Liste des valeurs PVO en pourcentage
    """
```

### GESTION DES CAS LIMITES
- **Premières périodes** : NaN comme TradingView
- **Volume nul** : Gestion des zéros correcte
- **Validation inputs** : fast < slow requis

---

## 📈 EXEMPLES D'UTILISATION

### CONFIGURATION CLASSIQUE
```python
# Volume Oscillator (différence)
vol_osc = volume_oscillator_tv(volume, 10, 30)

# Percentage Volume Oscillator
pvo = percentage_volume_oscillator_tv(volume, 10, 30)

# Ligne de signal PVO
signal = sma_tv(pvo, 9)
histogram = [p - s for p, s in zip(pvo, signal)]

# Signaux de base
volume_bullish = vol_osc[-1] > 0
volume_bearish = vol_osc[-1] < 0
pvo_bullish = pvo[-1] > 0
```

### STRATÉGIES COMBINÉES
```python
# Confirmation tendance prix + volume
price_up = close[-1] > close[-2]
volume_confirmed = price_up and (vol_osc[-1] > 0)

# Divergence haussière
price_lower = close[-1] < close[-5]
volume_higher = vol_osc[-1] > vol_osc[-5]
bullish_divergence = price_lower and volume_higher
```

---

## ⚠️ POINTS D'ATTENTION

### FAUX SIGNAUX
- **Volumes sporadiques** : Pics sans signification
- **Marchés illiquides** : Volumes faibles non pertinents
- **Gap horaires** : Variations volume artificielles

### OPTIMISATION RECOMMANDÉE
- **Ajuster périodes** : Selon timeframe et volatilité
- **Filtrer volumes extrêmes** : Éviter pics artificiels
- **Combiner avec prix** : Validation croisée obligatoire

---

## 🔄 VARIANTES ET EXTENSIONS

### VOLUME WEIGHTED MOVING AVERAGE
```python
# Alternative : VWMA au lieu de SMA
fast_vwma = vwma_tv(volume, volume, fast)
slow_vwma = vwma_tv(volume, volume, slow)
vwma_osc = fast_vwma - slow_vwma
```

### EXPONENTIAL VOLUME OSCILLATOR
```python
# Version EMA pour plus de réactivité
fast_ema = ema_tv(volume, fast)
slow_ema = ema_tv(volume, slow)
ema_osc = fast_ema - slow_ema
```

### MULTI-TIMEFRAME VOLUME
```python
# Analyse volume sur timeframe supérieur
higher_tf_volume = resample_volume(volume, '1D')
higher_tf_osc = volume_oscillator_tv(higher_tf_volume, 10, 30)
```

---

## 📚 RÉFÉRENCES COMPLÉMENTAIRES

### ANALYSE VOLUME-PRIX
- **Volume Spread Analysis** : Wyckoff method
- **On-Balance Volume** : Indicateur complémentaire
- **Accumulation/Distribution** : Flow volume-prix

### STRATÉGIES AVANCÉES
- **Volume Breakout** : Détection explosions volume
- **Volume Divergence** : Signaux inversion précoce
- **Volume Profile** : Distribution volume par prix

---

## ✅ VALIDATION TRADINGVIEW

Pour garantir une précision 100% TradingView :

1. **Utiliser SMA exact** : `ta.sma(volume, length)`
2. **Paramètres par défaut** : fast=10, slow=30
3. **Gestion NaN** : Premières `slow-1` valeurs = NaN
4. **Tests comparatifs** : vs scripts Pine Script référencés

---

*Dernière mise à jour : 03/01/2026*  
*Précision visée : 100% TradingView Standard*
