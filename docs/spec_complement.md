# Features “Courbure / Forme” pour identifier des extrêmes (méthode B)

**Données** : `Close` + `Typical Price (TP)`  
**Fenêtres glissantes** : `w ∈ {3, 6, 12}`  
**Séries (à appliquer partout)** : prix + chaque composant d’indicateur (ex : `MACD_line`, `MACD_signal`, `MACD_hist`, `CCI`, `MFI`, `Stoch_K`, `Stoch_D`, `DI+`, `DI-`, `DX/ADX`, `VWMA`, etc.)

Objectif : produire des **features numériques** qui décrivent la **forme géométrique** des `w` dernières valeurs d’une série, utiles pour explorer l’identification de **retournements / extrêmes** (sans fixer de règles mécaniques à ce stade).

---

## 1) Principe : ajustement quadratique (méthode B)

Pour une série \(x_t\) (ex: Close, TP, MACD_hist, DI+, etc.) sur une fenêtre \(w\) :

- On prend les \(w\) dernières valeurs : \(x_0, x_1, …, x_{w-1}\) (temps discret constant).
- On ajuste un polynôme de degré 2 :

\[
\hat{x}(t) = a t^2 + b t + c
\]

- \(t\) est l’index dans la fenêtre : \(t = 0 … w-1\)

**Note** :
- Sur `w=3`, l’ajustement est très proche de l’utilisation directe des différences (simple mais utile).
- Sur `w=6` et `w=12`, la “forme” est plus fiable (arrondi, épuisement, reprise, etc.).

---

## 2) Features à extraire (par série, par fenêtre)

### A) Dérivées locales (direction + accélération)
1. **Pente locale en fin de fenêtre**  
\[
s_{end} = \hat{x}'(w-1) = 2a(w-1) + b
\]

2. **Accélération / courbure signée**  
\[
\kappa = \hat{x}''(t) = 2a
\]

3. **Intensité de courbure**  
\[
|\kappa|
\]

**Interprétation (pour l’exploration “extrêmes”)** :
- \(\kappa < 0\) : concave (∩) → possible sommet / épuisement
- \(\kappa > 0\) : convexe (U) → possible creux / reprise

---

### B) Sommet / creux “théorique” (vertex) dans la fenêtre
Si \(a \neq 0\) :

4. **Position du vertex**  
\[
t^* = -\frac{b}{2a}
\]

5. **Vertex dans la fenêtre ?**  
\[
t^* \in [0, w-1] \quad (bool)
\]

6. **Distance du vertex à la bougie actuelle (fin de fenêtre)**  
\[
d = (w-1) - t^*
\]

7. **Valeur au vertex**  
\[
\hat{x}(t^*) = a(t^*)^2 + b t^* + c
\]

8. **Écart dernier point vs vertex**  
\[
\Delta_{vertex} = x_{w-1} - \hat{x}(t^*)
\]

---

### C) Qualité de la “forme” (fit quality)
9. **Qualité d’ajustement** : `R²` (ou erreur résiduelle normalisée)

10. **Résidu du dernier point**  
\[
r_{end} = x_{w-1} - \hat{x}(w-1)
\]

> Utile pour distinguer “arrondi propre” vs “spike / anomalie”.

---

### D) État de forme (catégoriel, optionnel)
À partir de `(s_end, κ)` : état “machine à états” pour classifier la forme (sans seuil dur).

- **Montée accélérée** : \(s_{end} > 0, \kappa > 0\)
- **Montée qui s’essouffle** : \(s_{end} > 0, \kappa < 0\)
- **Descente accélérée** : \(s_{end} < 0, \kappa < 0\)
- **Descente qui s’essouffle** : \(s_{end} < 0, \kappa > 0\)

---

## 3) Normalisation (important)

Pour rendre les features comparables entre séries :

### Prix (Close, TP)
- Normaliser `s_end`, `κ`, `Δ_vertex`, `r_end` par une échelle prix locale :
  - **ATR(w)** (si disponible), ou
  - **écart-type(w)**, ou
  - **MAD(w)** (robuste).

### Indicateurs bornés (Stoch 0–100, MFI 0–100)
- Normaliser par `100` (simple) ou par `std(w)` (si tu veux homogénéiser les amplitudes).

### Indicateurs non bornés (MACD, CCI, DI/DX/ADX, VWMA)
- Normaliser par `std(w)` ou `MAD(w)`.

---

## 4) Application “sur chaque composant d’indicateur”
Tu appliques **la même extraction** à chaque composant séparément :

- **MACD** : `macd_line`, `macd_signal`, `macd_hist`
- **Stoch** : `stoch_k`, `stoch_d`
- **DMI/ADX** : `di_plus`, `di_minus`, `dx` (et/ou `adx`)
- **CCI** : `cci`
- **MFI** : `mfi`
- **VWMA** : `vwma_fast`, `vwma_med`, `vwma_slow` (ou toute VWMA utilisée)
- + toutes les séries internes déjà validées dans ton doc (mêmes formules, même logique)

---

## 5) Nommage conseillé des features (plat)
Format : `<serie>_w<w>_<feature>`

Exemples :
- `close_w6_s_end`
- `tp_w12_kappa`
- `macd_hist_w6_t_star`
- `di_plus_w3_r2`
- `stoch_k_w12_delta_vertex`

Features (suffixes) recommandés :
- `s_end`
- `kappa`
- `kappa_abs`
- `t_star`
- `t_star_in_window` (0/1)
- `t_star_dist_to_end`
- `x_hat_t_star`
- `delta_vertex`
- `r2`
- `r_end`
- `shape_state` (optionnel, encodé)

---

## 6) Résumé ultra-court
Pour chaque série (prix + composant indicateur), et pour chaque fenêtre (3/6/12), tu extrais :
- pente fin (`s_end`)
- courbure (`kappa`, `|kappa|`)
- vertex (position + proximité + valeur + écart au dernier point)
- qualité (R²) + résidu fin (`r_end`)
- (optionnel) état de forme `shape_state`
