# 🎯 RÈGLES PRÉCISION 100% POUR INDICATEURS

## 📋 CHECKLIST OBLIGATOIRE APRÈS IMPLEMENTATION

### 1️⃣ SOURCE DE DONNÉES
```go
✅ Vérifier: Futures vs Spot vs Autre
✅ Confirmer: Endpoint exact utilisé
✅ Valider: Symbol format (SOL_USDT vs SOL-USDT)
```

### 2️⃣ FORMAT DE DONNÉES
```go
✅ Identifier: Array vs Struct vs JSON
✅ Mapper: Champs exacts (T, V, C, H, L, O, Sum)
✅ Parser: Types corrects (string vs float64 vs int64)
```

### 3️⃣ VOLUME ET VALEURS
```go
✅ Confirmer: Volume base (SOL) vs quote (USDT)
✅ Vérifier: Champs volume disponibles
✅ Choisir: Volume le plus pertinent pour indicateur
```

### 4️⃣ TIMESTAMPS ET DATES
```go
✅ Identifier: OpenTime vs CloseTime vs Timestamp
✅ Vérifier: Format Unix vs ISO vs autre
✅ Calculer: Intervalles corrects (5m = +300s)
```

---

## 🔧 PROCÉDURE VALIDATION INDICATEUR

### ÉTAPE 1: DEBUG SOURCE
```go
// Toujours créer un debug script
fmt.Printf("Source: %s\n", endpoint)
fmt.Printf("Format: %T\n", rawData)
fmt.Printf("Champs: %+v\n", rawData[0])
```

### ÉTAPE 2: VÉRIFICATION DONNÉES
```go
// Valider chaque champ
timestamp := int64(candle.T)  // Unix timestamp?
volume := float64(candle.V)   // SOL ou USDT?
price := parseFloat(candle.C) // String ou float?
```

### ÉTAPE 3: TEST FORMULES
```go
// Comparer avec documentation TradingView
TP := (high + low + close) / 3  // ✓
MF := TP * volume               // ✓
MFI := 100 - (100 / (1 + ratio)) // ✓
```

### ÉTAPE 4: DÉMO COMPLÈTE
```go
// Script validation avec:
- Récupération données
- Calcul indicateur  
- Affichage dates/heures
- Vérification formules
- Test cas limites
```

---

## 🎯 RÈGLES SPÉCIFIQUES PAR EXCHANGE

### 📊 GATE.IO
```go
✅ Futures: FuturesApi.ListFuturesCandlesticks
✅ Struct: {T, V, C, H, L, O, Sum}
✅ Volume: champ V (SOL) ou Sum (USDT)
✅ Params: pas de limit avec from/to
✅ Timestamp: T = OpenTime
✅ Interval: 5m = 300 secondes
```

### 📊 BINGX
```go
✅ Futures: /openApi/swap/v2/quote/klines
✅ Array: [timestamp, open, high, low, close, volume, end_timestamp, volume_quote]
✅ Volume: champ [7] (USDT)
✅ Params: limit + interval
✅ Timestamp: [0] = OpenTime, [6] = CloseTime
```

### 📊 BINANCE
```go
✅ Futures: /fapi/v1/klines
✅ Array: [open_time, open, high, low, close, volume, close_time, quote_asset_volume, ...]
✅ Volume: champ [5] (base) ou [7] (quote)
✅ Params: symbol + interval + limit
✅ Timestamp: [0] = OpenTime, [6] = CloseTime
```

---

## 💯 VALIDATION FINALE

### ✅ CHECKLIST PRÉCISION 100%
1. **Source** → Bon endpoint (futures/spot)
2. **Format** → Bon parsing (array/struct)
3. **Volume** → Bon champ (base/quote)
4. **Dates** → Bon timestamp (open/close)
5. **Formules** → Conformes documentation
6. **Tests** → Démo fonctionnelle
7. **Cas limites** → Gérés (NaN, zéro, etc.)

---

## 🚀 PROCÉDURE AUTOMATISÉE

### Template validation indicateur :
```go
func ValidateIndicateur(name string) {
    // 1. Test source données
    // 2. Vérifie format parsing
    // 3. Confirme volume utilisé
    // 4. Valide timestamps
    // 5. Test formules mathématiques
    // 6. Démo complète
    fmt.Printf("✅ %s: PRÉCISION 100%\n", name)
}
```

---

## 📝 ERREURS COURANTES À ÉVITER

### ❌ Erreurs de source
- Utiliser spot au lieu de futures
- Mauvais endpoint API
- Symbol format incorrect

### ❌ Erreurs de parsing
- Index array incorrect
- Type mismatch (string vs float)
- Struct fields inexistants

### ❌ Erreurs de volume
- Volume base au lieu de quote
- Champ volume incorrect
- Conversion manuelle fausse

### ❌ Erreurs de dates
- CloseTime au lieu de OpenTime
- Mauvais format timestamp
- Calcul interval incorrect

---

## 🎯 CONCLUSION

En suivant ces règles systématiquement, chaque indicateur atteint la précision 100% avec n'importe quel exchange et n'importe quel type de données.

**La clé : validation rigoureuse à chaque étape !**
