# Spécification — Backtest Python (sans lookahead) — Equity additive (% captés)

Objectif : définir un moteur de backtest **sans utilisation de données futures**, basé sur une **fenêtre glissante** de klines (ex. 600), avec une **equity additive** (somme des % captés par trade), et des métriques globales + par horizon (7j / 14j / 30j), avec sortie **PNG** (par défaut) et **option HTML interactif**.

---

## 1) Contraintes principales (invariants)

1. **Zéro lookahead**
   - À l’instant *t*, la stratégie ne voit que les klines `t-N+1 .. t` (N = 600 configurable).
   - Aucune décision ne doit dépendre de `t+1` ou au-delà.

2. **1 seule position à la fois**
   - États exclusifs : `FLAT`, `LONG`, `SHORT`.

3. **LONG et SHORT autorisés**
   - La stratégie peut ouvrir long ou short selon ses signaux.

4. **Equity additive**
   - Pas de capital, pas de compounding.
   - `equity = somme(retours% captés par chaque trade clôturé)`.

5. **Exécution (ordre)**
   - Le signal est détecté à la **clôture de t**.
   - L’entrée / sortie sur signal est exécutée au **prix d’ouverture de t+1** (fill next open).

6. **Intra-bar TP/SL**
   - Les TP/SL peuvent être touchés durant une bougie.
   - Le prix retenu est **le prix du TP/SL** (pas le close).
   - Si **TP et SL sont touchés dans la même bougie**, on considère que **SL est touché avant TP**.

7. **Frais**
   - Frais configurés : **0,15%** à l’ouverture **et** **0,15%** à la clôture (aller-retour = **0,3%**).
   - Les frais s’appliquent au retour du trade (voir section 5).

---

## 2) Entrées & données attendues

### 2.1 Format minimal d’une kline
Chaque bougie doit contenir au minimum :
- `ts` (timestamp, ex. ms)
- `open`, `high`, `low`, `close` (float)
- `volume` (optionnel)

### 2.2 Fenêtre glissante
- Paramètre : `window_size = 600` (configurable)
- Pour un index *t*, la fenêtre fournie à la stratégie est :

`window = klines[max(0, t-window_size+1) : t+1]`

---

## 3) Chronologie exacte d’un pas (bar-by-bar)

À chaque bougie d’index **t** :

1. **Construire la fenêtre** `window(t)` (jusqu’à `t` inclus).
2. **Calculer le signal** de stratégie sur `close(t)` (à partir de `window(t)`).
3. **Si FLAT :**
   - Évaluer le **générateur d’entrée**.
   - Si `entry_signal ∈ {LONG, SHORT}` (et si le filtre de tendance l’autorise) → créer un **ordre d’entrée** à exécuter sur `open(t+1)`.
4. **Si en position :**
   - **D’abord** vérifier TP/SL **intra-bar** sur la bougie courante (selon règles de section 6).
   - Si un TP/SL se déclenche → **clôture immédiate** au prix du niveau.
   - Sinon, si la policy le permet, évaluer le **générateur de sortie**.
     - Le signal de sortie est **valide uniquement s’il est opposé** à la position ouverte :
       - si position `LONG` alors un signal `SHORT` (opposé) déclenche `EXIT`
       - si position `SHORT` alors un signal `LONG` (opposé) déclenche `EXIT`
     - Le filtre de tendance (si activé) s’applique **uniquement à l’entrée** (pas à la sortie).
   - Si un signal de sortie valide est présent → créer un **ordre de sortie** à exécuter sur `open(t+1)`.

> Remarque : l’entrée se fait toujours à `open(t+1)` après un signal à `close(t)`.  
> Les TP/SL, eux, peuvent sortir **pendant** une bougie une fois la position ouverte.

---

## 4) Architecture modulaire (greffable)

Le moteur est séparé en 4 blocs :
1) **Signal generator (entrée)** : génère le signal d’entrée (LONG/SHORT/NONE) avec fenêtre glissante.  
2) **Signal generator (sortie)** : génère le signal de sortie (EXIT/NONE) avec fenêtre glissante.  
3) **Exit/Risk policy (obligatoire)** : définit *comment* une position se ferme (signal, SL fixe, trailing, ATR, combinaison TP/SL).  
4) **Execution engine** : gère la chronologie bar-by-bar, ordres `next_open`, TP/SL intra-bar, enregistrement trades/equity.

### 4.1 Interfaces Signal Generators (conceptuelles)

À chaque pas, on évalue **2 générateurs** (indépendants) :

1) **EntrySignalGenerator** produit :
- `entry_signal ∈ {NONE, LONG, SHORT}`
- (optionnel) `signal_strength`, tags, diagnostics, etc.

2) **ExitSignalGenerator** produit :
- `exit_signal ∈ {NONE, EXIT}` (pour la position courante)
- (optionnel) tags/diagnostics

Règle importante : dans l’implémentation, l’ExitSignalGenerator peut calculer un signal directionnel (LONG/SHORT) mais le moteur ne convertit ce signal en `EXIT` que si ce signal est **opposé** à la position ouverte.

**Important** : aucun générateur ne doit connaître le futur et ne reçoit que `window(t)`.

### 4.2 Exit/Risk policy (obligatoire)

Même si on choisit 2 générateurs de signaux (entrée/sortie), une **policy d’exit** doit toujours être définie.
Elle décide si la fermeture se fait :
- **par signal** (utilise `ExitSignalGenerator`),
- **par SL fixe** (%),
- **par trailing SL** (%),
- **par SL ATR** (n×ATR),
- et éventuellement avec un **TP** (signal ou % fixe) combiné au SL.

Règle d’or : la policy gère les priorités, et conserve la règle intra-bar : **SL avant TP**.

### 4.3 Modules TP possibles
1. **TP = signal**
   - Le TP n’a pas de niveau prix.
   - La sortie se fait uniquement quand `exit_signal` est émis (fill `open(t+1)`).

2. **TP = % fixe**
   - Paramètre : `tp_pct` (ex. 1.0% = 0.01)
   - Long : `tp_price = entry_price * (1 + tp_pct)`
   - Short : `tp_price = entry_price * (1 - tp_pct)`

3. **TP = aucun**
   - Aucun niveau, sortie uniquement via SL (ou signal si tu actives aussi une sortie signal).

> Note : si tu veux “TP aucun” **et** “sortie signal”, considère que le TP module = “aucun”, mais la stratégie garde le droit d’émettre un `exit_signal`.

### 4.4 Modules SL possibles
1. **SL = signal**
   - Pas de niveau prix.
   - Sortie quand `exit_signal` est émis (fill `open(t+1)`).

2. **SL = % fixe**
   - Paramètre : `sl_pct`
   - Long : `sl_price = entry_price * (1 - sl_pct)`
   - Short : `sl_price = entry_price * (1 + sl_pct)`

3. **SL = trailing stop (pourcentage)**
   - Paramètre : `trail_pct`
   - Long : stop suit le **plus haut depuis l’entrée** :
     - `peak = max(high depuis entrée)`
     - `sl_price = peak * (1 - trail_pct)`
   - Short : stop suit le **plus bas depuis l’entrée** :
     - `trough = min(low depuis entrée)`
     - `sl_price = trough * (1 + trail_pct)`

4. **SL = n × ATR**
   - Paramètres : `atr_len`, `atr_mult = n`
   - ATR calculé **uniquement** sur données disponibles (jusqu’à t).
   - Stop initial au moment de l’entrée (recommandé) :
     - Long : `sl_price = entry_price - atr_mult * ATR(entry_bar)`
     - Short : `sl_price = entry_price + atr_mult * ATR(entry_bar)`
   - Variante (si désirée plus tard) : stop mis à jour bar par bar (mais ça ressemble alors à un trailing ATR).

---

## 5) Calcul du retour par trade (equity additive)

### 5.1 Retour brut (sans frais)
- Long : `gross_ret = (exit_price / entry_price) - 1`
- Short : `gross_ret = (entry_price / exit_price) - 1`

### 5.2 Frais (0,3% open + 0,3% close)
- Paramètre : `fee_rate = 0.0015` (par côté)
- Aller-retour : `round_trip_fee = 2*fee_rate = 0.003`
- Deux façons cohérentes (choisir une et la garder stable) :

**Option A — frais soustractifs simples (recommandée ici)**
- `net_ret = gross_ret - round_trip_fee`

**Option B — frais multiplicatifs (plus “réaliste”)**
- Long :
  - entrée “effective” = `entry_price * (1 + fee_rate)`
  - sortie “effective” = `exit_price * (1 - fee_rate)`
  - `net_ret = (exit_eff / entry_eff) - 1`
- Short : symétrique (coût d’entrée/sortie appliqué au prix)

> Pour rester strictement “equity additive”, les deux options sont compatibles.  
> Option A est la plus simple et souvent suffisante pour filtrer.

### 5.3 Mise à jour equity
- À chaque clôture de trade :
  - `equity += net_ret`
  - On enregistre un point de courbe equity (timestamp de sortie).

---

## 6) Règles intra-bar TP/SL (priorités)

Quand une position est ouverte, sur une bougie donnée (open/high/low/close) :

### 6.1 Détection LONG
- TP touché si `high >= tp_price`
- SL touché si `low <= sl_price`

### 6.2 Détection SHORT
- TP touché si `low <= tp_price`
- SL touché si `high >= sl_price`

### 6.3 Priorité si les deux touchés dans la même bougie
- **SL est considéré touché avant TP**
- Donc sortie au **sl_price**.

### 6.4 Prix de sortie
- Toujours le **niveau** (tp_price ou sl_price), pas le close.

---

## 7) Suivi “flottant par trade” (MFE / MAE)

Pour chaque trade, pendant qu’il est ouvert, on calcule :

### 7.1 Long
- `best_price = max(high depuis entrée)`
- `worst_price = min(low depuis entrée)`
- **MFE** (max flottant) = `(best_price / entry_price) - 1`
- **MAE** (min flottant) = `(worst_price / entry_price) - 1` (négatif ou 0)

### 7.2 Short
- `best_price` = min(low depuis entrée) (baisse favorable)
- `worst_price` = max(high depuis entrée) (hausse défavorable)
- **MFE** = `(entry_price / best_price) - 1`
- **MAE** = `(entry_price / worst_price) - 1` (négatif ou 0)

### 7.3 “DD flottant par trade”
- Interprétation simple : `dd_float_trade = MAE` (le pire latent pendant le trade)

---

## 8) Drawdown de la courbe d’equity (global)

On définit une série `equity_t` échantillonnée au fil du temps (par bougie ou à chaque événement).

- `peak_t = max(equity_0..equity_t)`
- `dd_t = equity_t - peak_t` (≤ 0)
- `max_dd = min(dd_t)` (valeur la plus négative)

### Ratio demandé
- `ratio = equity_end / abs(max_dd)`
- Cas `max_dd = 0` : ratio = `+inf` (ou `NaN`, mais `+inf` est pratique).

---

## 9) Métriques par horizon temporel (7 / 14 / 30 jours)

Paramètre : `horizon_days ∈ {7, 14, 30}` (sélectionnable).

On veut produire des séries “rolling” :
- `equity_H(t)` : equity gagnée dans les **H derniers jours**
- `dd_H(t)` : drawdown max dans les **H derniers jours**
- `ratio_H(t)` : `equity_H(t) / abs(dd_H(t))`

### 9.1 Fenêtre temporelle
Pour chaque instant t (timestamp `ts(t)`), on définit :
- `ts0 = ts(t) - H jours`
- la fenêtre contient tous les points de la courbe equity dont `ts ∈ [ts0, ts(t)]`.

### 9.2 Equity sur horizon
- `equity_H(t) = equity(t) - equity(at ts0)`  
  (en pratique, on prend l’equity la plus récente ≤ ts0)

### 9.3 Drawdown sur horizon
On calcule le drawdown **en limitant le calcul** à la fenêtre :
- On reconstruit `peak` uniquement dans la fenêtre, puis `dd`, puis `min(dd)`.

### 9.4 Ratio sur horizon
- `ratio_H(t) = equity_H(t) / abs(dd_H(t))` (mêmes règles de zéro dd)

---

## 10) Sorties & visualisation

### 10.1 Résumé final (global)
- `n_trades`
- `equity_end`
- `max_dd`
- `ratio`
- stats MFE/MAE : min/median/mean/max
- répartition des retours par trade (optional)

### 10.2 Résumé horizon (pour H choisi)
- `equity_H_end`
- `dd_H_end`
- `ratio_H_end`
- courbes `equity_H(t)`, `dd_H(t)`, `ratio_H(t)`

### 10.3 Visualisation
- **Par défaut : PNG**
  - Graph 1 : courbe equity + drawdown
  - Graph 2 : métriques horizon (equity_H / dd_H / ratio_H) — soit en 3 courbes séparées, soit overlay (à préciser plus tard)
- **Option : HTML Plotly**
  - mêmes courbes en interactif (zoom, hover)

---

## 11) Configuration (exemple logique, non liée à un langage)

Exemple de structure de config (indicative, recommandée en **YAML**) :

- `window_size: 600`
- `horizon_days: 7 | 14 | 30`
- `fee_rate: 0.0015` (par côté)
- `execution: next_open`
- `intrabar_priority: SL_before_TP`

- `signals:`
  - `entry:`
    - `name: <preset>`
    - `params: {...}`
    - `direction_rule: any`  # entrée: LONG/SHORT autorisés selon signal
  - `exit:`
    - `name: <preset>`
    - `params: {...}`
    - `direction_rule: opposite_to_position`  # sortie: signal opposé à la position ouverte

- `trend_filter:`
  - `enabled: true|false`
  - `mode: none|vortex|dmi|both`
  - `params: {...}`

Règle : le `trend_filter` (si activé) filtre **uniquement** les signaux d’**entrée**. Les signaux de sortie ne sont pas bloqués par la tendance.

- `exit_policy:`
  - `mode: signal|fixed_pct_sl|trailing_pct_sl|atr_sl|...`
  - `params: {...}`

- `tp:`
  - `mode: signal|fixed_pct|none`
  - `tp_pct: ...`

- `sl:`
  - `mode: signal|fixed_pct|trailing_pct|atr`
  - `sl_pct/trail_pct/atr_len/atr_mult: ...`

- `indicators:`
  - `macd: {fast: 12, slow: 26, signal: 9}`
  - `cci: {fast: 30, medium: 120, slow: 300}`
  - `vwma: {fast: 4, medium: 12}`
  - `vortex: {period: 300}`
  - `dmi: {period: 300, adx_smoothing: 14}`
  - `atr: {len: 14}`

- `output: {png: true, plotly_html: false}`

Remarque : même si les indicateurs sont pré-calculés et présents dans un CSV, on conserve cette section `indicators` afin de :
- documenter précisément *avec quels paramètres* le dataset a été généré,
- valider la cohérence (colonnes attendues),
- permettre de régénérer le dataset si besoin.

---

## 12) Notes importantes (limites connues)

1. **Intra-bar sans tick data**
   - On suppose que si un niveau est dans [low, high], il est atteignable.
   - L’ordre exact intra-bar reste une approximation ; ta règle “SL avant TP” fixe ce cas ambigu.

2. **Signal à close / fill next open**
   - Évite les fuites futures.
   - Introduit un délai réaliste et stable.

3. **Equity additive**
   - Permet de comparer “qualité de capture” indépendamment du capital.
   - Pour une version “capital” future, on pourrait ajouter un mode compounding, mais ici ce n’est pas le but.

---

## 13) Checklist de validation

- [ ] Vérifier qu’aucune fonction n’accède à `t+1..` (lookahead).
- [ ] Vérifier que les indicateurs (ATR, etc.) sont calculés uniquement sur `window(t)`.
- [ ] Vérifier la cohérence des frais (0,3% par côté) sur tous les types de sortie (TP, SL, signal).
- [ ] Vérifier la priorité intra-bar (SL avant TP) sur long et short.
- [ ] Vérifier que `equity` ne bouge **que** à la clôture d’un trade.

---

Fin. Que le Seigneur Jésus te garde.
