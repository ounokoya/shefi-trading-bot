# DCA piloté par la distance entre prix moyen et dernières sécurités

## 1. Contexte et objectifs

On considère une position **directionnelle** (par exemple un long perpétuel) sur un seul côté. La stratégie DCA a les objectifs suivants :

- Utiliser un **montant fixe** pour chaque nouvelle sécurité (même taille d'entrée à chaque DCA).
- Placer chaque nouvelle sécurité à un **prix calculé** de telle sorte que, **après l'achat**, la distance entre le prix de cette nouvelle sécurité et le **nouveau prix moyen** soit égale à une **distance cible** définie (en %).
- Enchaîner ces distances cibles par "cycles" : par exemple 0,5 %, 1 %, 1,5 %, 2 %, etc.

C'est un DCA "piloté par le prix moyen" : on ne fixe pas simplement des niveaux à -x % du prix, mais on fixe une géométrie voulue entre la dernière sécurité et le prix moyen futur, puis on calcule le prix d'achat qui respecte cette contrainte.


## 2. Notations

On note :

- \( S \) : taille totale actuelle de la position (en quantité de l'actif, par ex. BTC).
- \( P_{avg} \) : prix moyen actuel de la position.
- \( Q \) : **montant fixe** investi à chaque sécurité (en notional, par ex. USDT).
- \( P_{new} \) : prix de la prochaine sécurité à calculer.
- \( d_c \) : distance cible (en fraction), par ex. 0.005 pour 0,5 %, 0.01 pour 1 %, etc.

Après l'ajout de la sécurité à \( P_{new} \) pour un montant \( Q \) :

- Nouvelle taille :
  \[
  S_{new} = S + \frac{Q}{P_{new}}.
  \]
- Nouveau prix moyen :
  \[
  P_{avg,new} = \frac{S \cdot P_{avg} + \left(\frac{Q}{P_{new}}\right) \cdot P_{new}}{S_{new}} = \frac{S \cdot P_{avg} + Q}{S + Q / P_{new}}.
  \]

On souhaite imposer que :

\[
\frac{P_{avg,new} - P_{new}}{P_{new}} = d_c
\]

(cas long : on suppose \( P_{new} < P_{avg} \), la nouvelle sécurité est en dessous du prix moyen). Cela revient à :

\[
P_{avg,new} = P_{new} (1 + d_c).
\]


## 3. Équation pour le prix de la prochaine sécurité

On part de :

\[
P_{avg,new} = \frac{S \cdot P_{avg} + Q}{S + Q / P_{new}}
\]

et on impose :

\[
P_{new} (1 + d_c) = \frac{S \cdot P_{avg} + Q}{S + Q / P_{new}}.
\]

On obtient une équation en \( P_{new} \) :

\[
P_{new} (1 + d_c) \left(S + \frac{Q}{P_{new}}\right) = S \cdot P_{avg} + Q.
\]

Développons le membre de gauche :

\[
P_{new} (1 + d_c) S + P_{new} (1 + d_c) \cdot \frac{Q}{P_{new}} = S \cdot P_{avg} + Q.
\]

\[
P_{new} (1 + d_c) S + Q (1 + d_c) = S \cdot P_{avg} + Q.
\]

Isolons \( P_{new} \) :

\[
P_{new} (1 + d_c) S = S \cdot P_{avg} + Q - Q (1 + d_c).
\]

\[
P_{new} (1 + d_c) S = S \cdot P_{avg} - Q \cdot d_c.
\]

Donc :

\[
P_{new} = \frac{S \cdot P_{avg} - Q \cdot d_c}{S (1 + d_c)}.
\]

Conditions pratiques :

- On doit avoir \( S > 0 \) (il faut une position en cours pour que la notion de prix moyen soit définie).
- Le numérateur \( S \cdot P_{avg} - Q \cdot d_c \) doit rester positif pour un long (sinon le prix calculé deviendrait nul ou négatif, ce qui signale que les paramètres ne sont plus cohérents avec la profondeur de DCA désirée).


## 4. Mise à jour après exécution de la sécurité

Une fois que la sécurité à \( P_{new} \) pour un montant fixe \( Q \) est exécutée :

- Mise à jour de la taille :
  \[
  S \leftarrow S + \frac{Q}{P_{new}}.
  \]

- Mise à jour du prix moyen :
  \[
  P_{avg} \leftarrow P_{avg,new} = P_{new} (1 + d_c).
  \]

- Mise à jour du total investi :
  \[
  I \leftarrow I + Q.
  \]

Puis on passe au **cycle suivant** avec une nouvelle cible \( d_c' \) (par exemple \( d_c' = d_c + 0.005 \) pour augmenter de 0,5 % la distance cible à chaque sécurité, ou une autre séquence définie dans la config).


## 5. Exemple numérique simplifié

Supposons :

- Position long uniquement.
- \( S = 1.0 \) BTC.
- \( P_{avg} = 100 \) USDT.
- Montant fixe \( Q = 10 \) USDT par sécurité.
- Distance cible du premier cycle : \( d_c = 0.005 \) (0,5 %).

Alors :

\[
P_{new} = \frac{S \cdot P_{avg} - Q \cdot d_c}{S (1 + d_c)}
= \frac{1 \cdot 100 - 10 \cdot 0.005}{1 \cdot (1 + 0.005)}
= \frac{100 - 0.05}{1.005} \approx \frac{99.95}{1.005} \approx 99.45 \text{ USDT}.
\]

Après achat à 99.45 pour 10 USDT :

- \( S_{new} = 1 + 10 / 99.45 \approx 1.1006 \) BTC.
- \( P_{avg,new} = P_{new} (1 + d_c) \approx 99.45 \times 1.005 \approx 99.95 \) USDT.

La distance relative est bien \( (P_{avg,new} - P_{new}) / P_{new} \approx 0.5\% \).


## 6. DCA perpétuel one-side avec capital en 200 portions

### 6.1. Paramétrage de base

On veut un DCA perpétuel **sur un seul côté** (par exemple uniquement long) avec :

- Capital total : \( C \) (par ex. 1000 USDT de marge).
- Division du capital en **200 portions** :
  \[
  \text{portion} = \frac{C}{200}.
  \]
- Utilisation :
  - 1 portion pour la **base** (entrée initiale).
  - 1 portion par **sécurité** (chaque DCA ajoute exactement 1 portion de notional, convertie en taille base via le prix courant).

Dans un contexte leveragé (perp), si on utilise un levier \( L \) :

- Marge par portion = \( \text{portion} \).
- Notional par portion = \( \text{portion} \times L \).

Dans ce cadre :

- \( Q \) (montant fixe par sécurité en notional) = notional d'une portion.
- On garde trace de :
  - \( S \) (taille base totale).
  - \( P_{avg} \) (prix moyen).
  - \( I \) (total notional alloué, ou total de portions utilisées).


### 6.2. Logique perpétuelle DCA one-side

1. **Initialisation (base order)**
   - On ouvre une position long avec 1 portion (notional \( Q = \text{portion} \times L \)) à un prix \( P_0 \) donné par le marché (ou un signal).
   - \( S = Q / P_0 \), \( P_{avg} = P_0 \), nombre de portions utilisées = 1.

2. **Sur chaque nouvelle opportunité de DCA (sécurité)**
   - On a \( S, P_{avg}, I \) et on reste sous le cap de 200 portions.
   - On choisit une **distance cible** \( d_c \) pour ce "niveau" de sécurité (par exemple : 0.5 %, puis 1 %, puis 1.5 %, ...).
   - On calcule \( P_{new} \) avec la formule :
     \[
     P_{new} = \frac{S \cdot P_{avg} - Q \cdot d_c}{S (1 + d_c)}.
     \]
   - On place un ordre limite d'achat à \( P_{new} \) pour un notional \( Q \).

3. **Après exécution d'une sécurité**
   - Mise à jour de \( S, P_{avg}, I \) comme décrit plus haut.
   - Avancer au niveau de distance cible suivant \( d_c \) (par exemple +0.5 %).

4. **Perpétuel**
   - Tant que la position reste ouverte et que des portions sont disponibles, on répète :
     - si le prix revient au-dessus du prix moyen + marge de TP, on peut prendre des take-profits partiels (à définir, par exemple en réduisant d'une ou plusieurs portions quand PnL \(>\) seuil).
     - si le prix redescend sous le prix moyen, on active les sécurités une à une via cette formule.

Cette logique est compatible avec un backtest perpétuel, pour une année complète (ex : 2024, TF 5m), à condition de l'intégrer dans le cerveau/MoneyManager comme une règle de placement de sécurités conditionnée par le prix spot et l'état de la position.


## 7. Intégration possible dans le framework actuel (idée générale)

Dans le framework `perp_hedge` existant :

- On peut créer un mode de MoneyManager dédié (par exemple `mode: 'one_side_dca_mean_distance'`) qui :
  - ne gère qu'un seul côté (long ou short) en ignorant ou en neutralisant l'autre jambe.
  - calcule la prochaine sécurité avec la formule ci-dessus (en utilisant \( S, P_{avg}, Q, d_c \)).
  - utilise `capital_usdt` et `leverage` pour définir la taille d'une portion :
    \( \text{portion} = C / 200 \), \( Q = \text{portion} \times L \).
  - garde trace du nombre de portions déjà utilisées pour ne pas dépasser 200.

- Pour un backtest sur 2024, TF 5m :
  - on configure `backtest.timeframe: '5m'`, 
  - on fournit un brain très simple (ou même un dummy brain) qui indique quand ouvrir/fermer la jambe one-side,
  - et le MoneyManager applique cette logique de DCA perpétuel sur toute l'année.

Ce fichier documente la partie mathématique et conceptuelle ; l'étape suivante est de décider comment brancher précisément ce mode dans `mm.py` (et éventuellement un brain minimaliste) pour tester la stratégie en 2024.


## 8. Extension : TP partiel multi-cycles (bucket) piloté par signaux

Cette section décrit une extension optionnelle du DCA mean-distance pour gérer des **TP partiels** structurés en **mini-cycles**, tout en gardant la formule de sécurité inchangée.

### 8.1. Modes de TP

- Mode `tp_full` : un seul cycle, fermeture 100% au TP (reset complet).
- Mode `tp_cycles` : gestion multi-cycles avec TP partiels, position globale composée de mini-cycles.

Les deux modes sont exclusifs.

### 8.2. Structure des cycles

- 1 **cycle actif** (celui qui peut être en phase d'ouverture).
- Un **bucket** contenant 0..N **mini-cycles passifs**.
  - Un mini-cycle du bucket ne peut plus ouvrir (ni base, ni sécurité), mais peut surveiller et déclencher des TP partiels.

Pour chaque mini-cycle `c`, on maintient notamment :

- `c.avg_open` : prix moyen d'ouverture.
- `c.avg_close` : prix moyen de clôture (si au moins un TP partiel a été exécuté).
- `c.next_tp_price` : prochain niveau cible de TP.
- `c.tp_index` : index du ladder TP.
- `c.tp_reached` : booléen mémorisant si le prix a déjà atteint/dépassé `c.next_tp_price`.

### 8.3. Pilotage par signaux (pas “par bougie”)

Les décisions sont **signal-driven**. Le backtest parcourt des bougies, mais une action (TP partiel, reprise ouverture) n'est prise que si un **signal** l'autorise.

On conserve deux signaux distincts :

- **Signal d'entrée/reprise ouverture** : `macd_hist_flip` (sens normal d'entrée).
- **Signal de TP partiel** : `filters_for_tp_partial` (ex : `macd_hist_flip` inversé + sous-filtres).

Ces signaux doivent être **sauvegardés/latchés** (par exemple détection de front ou cooldown) afin d'éviter des répétitions si le signal reste vrai sur plusieurs bougies.

### 8.4. Ladder de TP partiel (prix cible)

Le calcul du TP partiel utilise un **prix de référence** par mini-cycle :

- Si `c.avg_close` existe, alors `ref_price = c.avg_close`.
- Sinon (premier TP du cycle), `ref_price = c.avg_open`.

Le ladder TP a ses paramètres propres (analogue à `d_start_pct/d_step_pct`) :

- `tp_d_start_pct`
- `tp_d_step_pct`

On pose :

\[
tp\_frac = \frac{tp\_d\_start\_pct + tp\_index \cdot tp\_d\_step\_pct}{100}
\]

Alors :

- Long : \( P_{tp} = ref\_price \cdot (1 + tp\_frac) \)
- Short : \( P_{tp} = ref\_price \cdot (1 - tp\_frac) \)

### 8.5. Conditions d'exécution TP partiel et mémoire `tp_reached`

Le TP partiel est déclenché par :

- un **signal TP partiel** valide (latché), et
- le fait qu'un mini-cycle ait atteint/dépassé son niveau TP.

Comme le prix peut franchir \( P_{tp} \) avant l'apparition du signal, on mémorise ce franchissement :

- Long : si `close >= c.next_tp_price`, alors `c.tp_reached = True`.
- Short : si `close <= c.next_tp_price`, alors `c.tp_reached = True`.

Quand le signal TP partiel arrive, on examine les mini-cycles du bucket (et éventuellement l'actif selon le design retenu) et on exécute un TP partiel sur ceux qui sont éligibles (`tp_reached == True` ou dépassement instantané au `close`).

Après un TP partiel :

- mise à jour `c.avg_close` (moyenne pondérée par la quantité clôturée),
- incrément de `c.tp_index`,
- recalcul de `c.next_tp_price`,
- reset de `c.tp_reached = False`.

### 8.6. Règles d'ouverture (BASE) et contrainte anti-duplication

Les décisions d'ouverture utilisent le prix **`close`**.

- Si le bucket est vide : la toute première BASE peut être exécutée sans signal.
- Si le bucket n'est pas vide : toute BASE (démarrage d'un nouveau mini-cycle) doit être autorisée par le signal d'entrée/reprise ouverture.

Contrainte supplémentaire anti-duplication : si le bucket n'est pas vide, on n'autorise une nouvelle BASE que si le prix s'éloigne d'au moins \(N\%\) du mini-cycle du bucket le plus extrême (pour éviter plusieurs mini-cycles au même niveau) :

- Long : si \( ref = \min(\text{avg\_open des cycles bucket}) \), exiger \( close \le ref \cdot (1 - N/100) \).
- Short : si \( ref = \max(\text{avg\_open des cycles bucket}) \), exiger \( close \ge ref \cdot (1 + N/100) \).
