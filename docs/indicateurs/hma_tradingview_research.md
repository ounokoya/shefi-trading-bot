# 🔍 HMA TradingView - Recherche d'Implémentation Précise

## 📋 Table des Matières
1. [Formule Officielle TradingView](#formule-officielle-tradingview)
2. [Calculs Détaillés](#calculs-détaillés)
3. [Implémentations Pine Script](#implémentations-pine-script)
4. [Astuces et Optimisations](#astuces-et-optimisations)
5. [Cas d'Usage Avancés](#cas-dusage-avancés)
6. [Sources et Références](#sources-et-références)

---

## 🎯 Formule Officielle TradingView

### Définition
Le **Hull Moving Average (HMA)** est un indicateur de moyenne mobile créé par Alan Hull. Il combine des moyennes mobiles pondérées (WMA) pour réduire le lag tout en maintenant la courbure. L'HMA est extrêmement réactif aux changements de prix.

### Formule Mathématique Complète
**Source : Alan Hull - https://alanhull.com/hull-moving-average**
```
Integer(SquareRoot(Period)) WMA [2 x Integer(Period/2) WMA(Price) - Period WMA(Price)]
```

**Formule simplifiée TradingView :**
```
HMA = WMA(2 × WMA(n/2) - WMA(n), sqrt(n))
```

### Étapes de Calcul (4 étapes obligatoires)

#### Étape 1 - WMA sur n/2 périodes
```
WMA_half = WMA(Source, n/2)
```

#### Étape 2 - WMA sur n périodes
```
WMA_full = WMA(Source, n)
```

#### Étape 3 - Calcul de la série intermédiaire
```
HMA_intermediate = (2 × WMA_half) - WMA_full
```

#### Étape 4 - HMA final
```
HMA = WMA(HMA_intermediate, sqrt(n))
```

---

## 📝 Calculs Détaillés

### Processus Complet pour Période 16 (Exemple d'Alan Hull)

1. **Calculer WMA sur 8 périodes** (n/2)
   ```
   WMA_half[i] = Σ(Source[j] × Weight[j]) / Σ(Weight[j])
   ```
   où Weight[j] = j+1 pour j = 0 à 7

2. **Calculer WMA sur 16 périodes**
   ```
   WMA_full[i] = Σ(Source[j] × Weight[j]) / Σ(Weight[j])
   ```
   où Weight[j] = j+1 pour j = 0 à 15

3. **Calculer la série intermédiaire**
   ```
   HMA_intermediate[i] = (2 × WMA_half[i]) - WMA_full[i]
   ```

4. **Calculer HMA final sur sqrt(16) = 4**
   ```
   HMA[i] = WMA(HMA_intermediate, 4)
   ```

### Calcul WMA (Weighted Moving Average)
**Source : HullMovingAverage.com - https://hullmovingaverage.com/hull-moving-average-formula/**
```
WMA = (Price1 × 1 + Price2 × 2 + ... + Pricen × n) / (1 + 2 + ... + n)
```

**Notes importantes sur les arrondis :**
- `n/2` : Utiliser la partie entière (Integer)
- `sqrt(n)` : Utiliser la partie entière (Integer)
- Ces arrondis sont CRUCIAUX pour la conformité TradingView

---

## 📝 Implémentations Pine Script

### 1. Version Standard TradingView
**Source : TradingView Pine Script Reference**
```pine
//@version=5
indicator("Hull Moving Average", shorttitle="HMA", overlay=true)

length = input.int(9, title="Length", minval=1)
src = input(close, title="Source")

hmaValue = ta.hma(src, length)

plot(hmaValue, title="HMA", color=color.blue, linewidth=2)
```

### 2. Implémentation Manuelle Complète
**Source : Pine Script HMA Guide - https://offline-pixel.github.io/pinescript-strategies/pine-script-HMA.html**
```pine
//@version=5
indicator("Manual HMA", shorttitle="MHMA", overlay=true)

length = input.int(9, title="Length")
src = input(close, title="Source")

// WMA sur n/2 (arrondi à l'entier inférieur)
halfLength = math.round(length / 2)
wmaHalf = ta.wma(src, halfLength)

// WMA sur n
wmaFull = ta.wma(src, length)

// Série intermédiaire
hmaIntermediate = (2 * wmaHalf) - wmaFull

// HMA final sur sqrt(n) (arrondi à l'entier)
sqrtLength = math.round(math.sqrt(length))
hma = ta.wma(hmaIntermediate, sqrtLength)

plot(hma, title="HMA", color=color.blue, linewidth=2)
```

### 3. HMA avec Signaux de Trading
**Source : Pine Script HMA Guide**
```pine
//@version=5
indicator("HMA Trading Signals", shorttitle="HMA Signals", overlay=true)

length = input.int(9, title="Length")
src = input(close, title="Source")

hmaValue = ta.hma(src, length)

// Signaux de croisement
crossAbove = ta.crossover(src, hmaValue)
crossBelow = ta.crossunder(src, hmaValue)

// Détection de tendance
hmaTrend = hmaValue > hmaValue[1]
trendColor = hmaTrend ? color.green : color.red

// Affichage
plot(hmaValue, title="HMA", color=trendColor, linewidth=3)

plotshape(crossAbove, title="Buy Signal", location=location.belowbar,
          style=shape.labelup, color=color.green, text="BUY")
plotshape(crossBelow, title="Sell Signal", location=location.abovebar,
          style=shape.labeldown, color=color.red, text="SELL")
```

### 4. HMA Multi-Timeframe
```pine
//@version=5
indicator("HMA MTF", shorttitle="HMA MTF", overlay=true)

length = input.int(9, title="Length")
tf1 = input.timeframe("1H", "Timeframe 1")
tf2 = input.timeframe("4H", "Timeframe 2")

src = close

// HMA sur différents timeframes
hmaCurrent = ta.hma(src, length)
hmaTF1 = request.security(syminfo.tickerid, tf1, ta.hma(src, length))
hmaTF2 = request.security(syminfo.tickerid, tf2, ta.hma(src, length))

plot(hmaCurrent, title="HMA Current", color=color.blue, linewidth=2)
plot(hmaTF1, title="HMA " + tf1, color=color.orange, linewidth=2)
plot(hmaTF2, title="HMA " + tf2, color=color.purple, linewidth=2)
```

---

## ⚡ Astuces et Optimisations

### 1. Périodes Optimisées par Style
**Source : Expérience communautaire TradingView**
```pine
// Scalping (très réactif)
scalpingHMA = ta.hma(close, 5)

// Day Trading
dayHMA = ta.hma(close, 9)

// Swing Trading
swingHMA = ta.hma(close, 15)

// Position Trading
positionHMA = ta.hma(close, 20)

// Long terme
longTermHMA = ta.hma(close, 50)
```

### 2. Sources Alternatives
```pine
// Close (standard)
src1 = close

// Typical Price
src2 = hlc3

// Weighted Close
src3 = (high + low + 2 * close) / 4

// Median Price
src4 = hl2

hma1 = ta.hma(src1, 9)
hma2 = ta.hma(src2, 9)
hma3 = ta.hma(src3, 9)
hma4 = ta.hma(src4, 9)
```

### 3. HMA avec Filtre de Volatilité
```pine
length = input.int(9, title="Length")
src = input(close, title="Source")
volatilityFilter = input.bool(true, title="Volatility Filter")

hmaValue = ta.hma(src, length)

// Filtre ATR
atr = ta.atr(14)
atrPercent = atr / close * 100

// Couleur basée sur la volatilité
hmaColor = volatilityFilter and atrPercent > 2 ? color.yellow : color.blue

plot(hmaValue, title="HMA", color=hmaColor, linewidth=2)
```

### 4. Détection de Changement de Pente
**Source : Pine Script HMA Guide**
```pine
hmaValue = ta.hma(close, 9)

// Calcul de la pente
slope = hmaValue - hmaValue[1]
slopeAngle = math.atan(slope) * 180 / math.pi

// Signaux de pente
steepUp = slope > 0 and slopeAngle > 30
steepDown = slope < 0 and slopeAngle < -30
flat = math.abs(slopeAngle) < 5

plot(hmaValue, title="HMA", color=color.blue, linewidth=2)
bgcolor(steepUp ? color.new(color.green, 90) : steepDown ? color.new(color.red, 90) : na)
```

**Note importante** : L'HMA est extrêmement réactif mais peut générer des faux signaux en marché sans tendance. Il est recommandé de le combiner avec d'autres indicateurs pour confirmation.

---

## 📊 Cas d'Usage Avancés

### 1. HMA + RSI Système Complet
```pine
// HMA pour la tendance, RSI pour les entrées
hmaValue = ta.hma(close, 9)
rsiValue = ta.rsi(close, 14)

// Tendance HMA
trendUp = close > hmaValue
trendDown = close < hmaValue

// Signaux RSI dans la tendance
buySignal = trendUp and rsiValue < 30 and ta.crossover(rsiValue, 30)
sellSignal = trendDown and rsiValue > 70 and ta.crossunder(rsiValue, 70)

plot(hmaValue, title="HMA", color=color.blue, linewidth=2)
plotshape(buySignal, title="Buy", location=location.belowbar,
          style=shape.labelup, color=color.green, text="BUY")
plotshape(sellSignal, title="Sell", location=location.abovebar,
          style=shape.labeldown, color=color.red, text="SELL")
```

### 2. HMA Bandes Enveloppes
```pine
hmaValue = ta.hma(close, 9)
atr = ta.atr(14)

// Bandes autour de HMA
upperBand = hmaValue + (atr * 2)
lowerBand = hmaValue - (atr * 2)

// Signaux de sortie de bandes
priceAboveUpper = close > upperBand
priceBelowLower = close < lowerBand

plot(hmaValue, title="HMA", color=color.blue, linewidth=2)
plot(upperBand, title="Upper Band", color=color.red, linestyle=hline.style_dashed)
plot(lowerBand, title="Lower Band", color=color.green, linestyle=hline.style_dashed)
```

### 3. HMA Divergence Detector
```pine
hmaValue = ta.hma(close, 9)

// Divergences haussières
bullishDiv = low[5] < low[10] and hmaValue[5] > hmaValue[10] and 
             low < low[5] and hmaValue > hmaValue[5]

// Divergences baissières  
bearishDiv = high[5] > high[10] and hmaValue[5] < hmaValue[10] and
             high > high[5] and hmaValue < hmaValue[5]

plot(hmaValue, title="HMA", color=color.blue, linewidth=2)
plotshape(bullishDiv, title="Bullish Divergence", location=location.bottom,
          style=shape.labelup, color=color.green, text="BULL DIV")
plotshape(bearishDiv, title="Bearish Divergence", location=location.top,
          style=shape.labeldown, color=color.red, text="BEAR DIV")
```

---

## 🎯 Points Clés à Retenir

### ✅ Avantages du HMA TradingView
- **Réactivité extrême** : Lag minimal par rapport aux autres moyennes mobiles
- **Courbure naturelle** : Suit bien les mouvements de prix
- **Polyvalent** : Fonctionne sur tous les timeframes
- **Simple à interpréter** : Position prix vs HMA

### ⚠️ Points d'Attention
- **Sensibilité extrême** : Peut générer des faux signaux
- **Bruitage** : En marché plat, peut osciller rapidement
- **Pas de bornes** : Contrairement aux oscillateurs
- **Dépendant de la période** : Le choix de n est crucial

### 🚀 Meilleures Pratiques
- Utiliser n=9 pour day trading par défaut
- Combiner avec filtre de tendance ou volatilité
- Confirmer les croisements avec volume ou price action
- Adapter la période selon le style de trading

---

## 📚 Sources et Références

### 📖 Documentation Officielle
1. **Alan Hull - Site Officiel**
   - URL : https://alanhull.com/hull-moving-average
   - Contenu : Formule originale et explication du créateur
   - Formule exacte : `Integer(SquareRoot(Period)) WMA [2 x Integer(Period/2) WMA(Price) - Period WMA(Price)]`
   - Dernière consultation : 09/11/2025

2. **TradingView Pine Script Reference Manual**
   - URL : https://www.tradingview.com/pine-script-reference/v5/
   - Section : Built-in functions → ta.hma()
   - Contenu : Implémentation officielle TradingView
   - Dernière consultation : 09/11/2025

3. **Hull Moving Average Formula - Guide Complet**
   - URL : https://hullmovingaverage.com/hull-moving-average-formula/
   - Contenu : Calcul détaillé étape par étape avec exemples
   - Dernière consultation : 09/11/2025

### 📚 Guides et Tutoriels
4. **Pine Script Hull Moving Average - Complete TradingView Guide**
   - URL : https://offline-pixel.github.io/pinescript-strategies/pine-script-HMA.html
   - Contenu : Implémentation Pine Script complète avec stratégies
   - Dernière consultation : 09/11/2025

5. **TradingView Scripts - Hull Moving Average**
   - URL : https://www.tradingview.com/scripts/hullma/
   - Contenu : Scripts communautaires et variantes
   - Dernière consultation : 09/11/2025

### 🔍 Références Historiques
6. **Alan Hull**
   - Créateur original du HMA
   - Analyste technique australien, spécialiste des moyennes mobiles
   - Site officiel : https://alanhull.com/

---

## 📋 Implémentation Go Référence

```go
// Implémentation HMA compatible TradingView
// Basée sur la formule : Integer(SquareRoot(Period)) WMA [2 x Integer(Period/2) WMA(Price) - Period WMA(Price)]
type HMA struct {
    period int
}

func NewHMA(period int) *HMA {
    return &HMA{period: period}
}

func (hma *HMA) Calculate(prices []float64) []float64 {
    n := len(prices)
    result := make([]float64, n)
    
    if n < hma.period {
        return result
    }
    
    // Calculer n/2 arrondi à l'entier inférieur (selon formule Alan Hull)
    halfPeriod := hma.period / 2
    
    // Calculer sqrt(n) arrondi à l'entier (selon formule Alan Hull)
    sqrtPeriod := int(math.Sqrt(float64(hma.period)))
    
    // Calculer WMA sur n/2
    wmaHalf := hma.calculateWMA(prices, halfPeriod)
    
    // Calculer WMA sur n
    wmaFull := hma.calculateWMA(prices, hma.period)
    
    // Calculer la série intermédiaire
    intermediate := make([]float64, n)
    for i := 0; i < n; i++ {
        if !math.IsNaN(wmaHalf[i]) && !math.IsNaN(wmaFull[i]) {
            intermediate[i] = (2 * wmaHalf[i]) - wmaFull[i]
        }
    }
    
    // Calculer HMA final sur sqrt(n)
    result = hma.calculateWMA(intermediate, sqrtPeriod)
    
    return result
}

func (hma *HMA) calculateWMA(prices []float64, period int) []float64 {
    n := len(prices)
    result := make([]float64, n)
    
    for i := period - 1; i < n; i++ {
        var sum, weightSum float64
        
        for j := 0; j < period; j++ {
            weight := float64(j + 1)
            sum += prices[i-period+1+j] * weight
            weightSum += weight
        }
        
        if weightSum != 0 {
            result[i] = sum / weightSum
        }
    }
    
    return result
}
```

---

*Document créé le 09/11/2025 - Basé sur recherche TradingView, Alan Hull et documentation officielle*
