# Spec stratégie — Perp Hedge (LONG + SHORT) + MM (Money Management)
 
 Ce document décrit la **logique pure** (sans détails techniques) d’une stratégie en **perp hedge mode** (LONG et SHORT possibles en même temps sur le même actif) où :
 
 - **Le générateur de signaux ne donne pas de quantité** : il dit seulement s’il faut **compléter** (augmenter) ou **retirer** (réduire) la jambe LONG et/ou SHORT.
 - Les exécutions d’ajout/retrait se font **au prix du marché** (pas d’ordres en attente).
 - Le module **MM (Money Management)** peut, si besoin, **calculer la quantité à ajouter/retirer** via une fonction utilitaire optionnelle, en respectant des limites de risque.
 - La quantité à ajouter/retirer dépend de l’**écart entre les deux jambes**, et n’est activée que si cet écart dépasse un **seuil en % configurable**.

---

 ## 1) Architecture (ce que fait chaque bloc)
 
### 1.1 Générateur de signaux (le “cerveau”)
 À chaque instant, il produit des **intentions** (sans quantité), du type :
 - **LONG peut augmenter** (compléter)
 - **LONG peut réduire** (retirer)
 - **SHORT peut augmenter** (compléter)
 - **SHORT peut réduire** (retirer)
 
 Important :
 - le signal décide **quoi** faire (quelle jambe + quel sens)
 - le signal peut aussi décider **de ne rien faire**
 
### 1.2 MM (Money Management)
 Le MM **ne décide pas** la direction.
 Son rôle :
 - convertir une intention “compléter/retirer” en une **quantité** (si on active l’utilitaire de sizing)
 - appliquer des **caps de risque** (limites d’exposition, tailles max, etc.)
 - exécuter l’ajout/retrait **au prix du marché**

---

 ## 2) Définitions simples (vocabulaire)

 - **Jambe LONG** : position qui gagne si le prix monte.
 - **Jambe SHORT** : position qui gagne si le prix baisse.
 - **Intention** : demande du signal au MM, ex: “LONG peut augmenter”, “SHORT peut réduire”.
 - **Sizing** : règle (optionnelle) qui calcule la quantité à ajouter/retirer.
 - **Notional LONG/SHORT** : valeur (en USDT) de chaque jambe.
 - **Écart entre jambes** : différence relative entre les 2 notionnels.
 - **Levier (L)** : multiplicateur d’exposition. Exemple : L=10.
 - **Cross margin** : la marge est **mutualisée** au niveau du compte.
 - **Wallet balance** : Solde total du compte hors PnL non réalisé.
  - Se calcule à l'instant T comme : `wallet_balance = marge_non_investie + marge_investie`.
  - C'est une variable d'état mise à jour à chaque opération (frais, funding, PnL réalisé), sans recalculer tout l'historique.
 - **PnL non réalisé** : profit/perte latente des positions ouvertes.
 - **PnL non réalisé (hedge mode)** : `pnl_non_realise = pnl_long + pnl_short`.
 - **Equity du compte** : `equity = wallet_balance + pnl_non_realise`.
 - **Marge totale (simplifiée)** : total disponible en cross margin en incluant uniquement le PnL non réalisé **positif** :
   - `margin_total = wallet_balance + max(0, pnl_non_realise)`
 - **Used margin (approx.)** : marge “utilisée” (marge investie) par les positions.
   - en backtest, on la suit comme une **comptabilité** des actions exécutées (levier constant `L`) :
     - si on augmente une jambe au notional `step_usdt` : `used_margin += step_usdt / L`
     - si on réduit une jambe au notional `step_usdt` : `used_margin -= step_usdt / L`
 - **Free margin (approx.)** : marge “libre”, approximativement `free_margin = equity - used_margin` (métrique informative, pas le critère de liquidation).
 - **Notional (valeur notionnelle)** : exposition de la jambe, approximativement `notional_usdt = qty * prix`.
 - **Lien levier/marge/notional** (approx.) : pour une action d’augmentation, le “capital” mobilisé est la marge du compte, et `notional_usdt ≈ margin_usdt * L` reste une approximation utile.

---

 ## 3) Ce que fait le MM (et ce qu’il ne fait pas)
 
### 3.1 Le MM fait
 - reçoit des intentions (“compléter/retirer LONG/SHORT”)
 - calcule une quantité via une règle de sizing (si activée)
 - exécute l’ajout/retrait au marché
 - applique des règles de protection (cap d’exposition, taille max par action, cadence)
 
 ### 3.2 Le MM ne fait pas
 - ne choisit pas LONG vs SHORT (c’est le signal)
 - ne transforme une intention en action que si les conditions MM le permettent (caps/cadence)

---

 ## 4) Entrées du MM (ce qu’il reçoit)
 
 ### 4.1 Les intentions du signal
 À chaque cycle, le signal peut produire 0 à N intentions, par exemple :
 - `LONG_CAN_INCREASE`
 - `LONG_CAN_DECREASE`
 - `SHORT_CAN_INCREASE`
 - `SHORT_CAN_DECREASE`
 
 Invariants :
 - à un instant t, on ne peut pas avoir **2 intentions opposées pour la même jambe**
   - pas de `LONG_CAN_INCREASE` et `LONG_CAN_DECREASE` en même temps
   - pas de `SHORT_CAN_INCREASE` et `SHORT_CAN_DECREASE` en même temps
 - on peut avoir **jusqu’à 2 intentions simultanées** : 1 pour LONG et 1 pour SHORT.
 - cela permet au MM d’exécuter **deux actions au même instant** (si souhaité) :
   - par exemple **réduire (TP) une jambe** et **compléter l’autre** en même temps, selon le signal.
 
 ### 4.2 Le contexte de marché
 Le MM a besoin d’une vue simple du marché :
 - prix courant
 - positions actuelles LONG/SHORT
 - taille minimale et pas de quantité (contraintes exchange)

---

 ## 5) Sorties du MM (ce qu’il produit)
 
 Le MM ne produit pas un “signal”. Il produit des **actions d’exécution** :
 - exécuter un **achat/vendre au marché** pour augmenter/réduire une jambe
 - ne rien faire si la quantité calculée est trop petite ou si une limite est atteinte

---

 ## 6) Logique MM : comment décider la quantité (sizing) puis exécuter
 
 Le MM peut fonctionner en 2 modes :
 
 - **Mode 1 : quantité imposée ailleurs**
   - le MM exécute au marché la quantité déjà décidée (si le projet le fait plus tard)
 - **Mode 2 : quantité calculée par le MM (utilitaire optionnel)**
   - le MM calcule la quantité à ajouter/retirer à partir de règles simples
 
 ### 6.1 Règle de sizing basée sur l’écart à combler
 
 Idée : l’**écart** entre les deux jambes est précisément la quantité (en USDT notionnel) à **combler**.
 Cet écart représente ce qui manque pour ré-équilibrer (le “dip” à compléter), ou ce qu’on peut clôturer/réduire pour revenir à l’équilibre.
 
 Paramètres configurables (logique) :
 - `capital_usdt` : capital de référence (wallet balance initial)
 - `initial_long_usdt` : investissement initial jambe LONG
 - `initial_short_usdt` : investissement initial jambe SHORT
 - `max_initial_invest_usdt` : maximum du capital initial à investir (cap global)
 - `gap_threshold_pct` : seuil % minimal d’écart pour déclencher un ajustement
 - `min_step_usdt` : taille minimale d’un ajout/retrait
 - `max_step_usdt` : taille maximale d’un ajout/retrait
 - `cooldown_bars` : cooldown (en bougies) entre deux actions sur la même jambe
 - `fee_rate_per_op` : frais par opération (par entrée/sortie), ex: `0.0015` (0.15%)
 - `liquidation_threshold_pct` : seuil simplifié de liquidation (ex: `0.05` = 5%) basé sur la **marge totale** `margin_total`
 
 Définition de l’écart à combler :
 - `gap_usdt = abs(notional_long - notional_short)`
 - `gap_pct = gap_usdt / (notional_long + notional_short)` si la somme > 0
 
 Déclenchement :
 - si `gap_pct <= gap_threshold_pct` : **ne rien faire**
 - si `gap_pct > gap_threshold_pct` : on essaye de combler une partie de l’écart
 
 Quantité d’action (notionnel USDT) :
 - `step_usdt = clamp(gap_usdt, min_step_usdt, max_step_usdt)`

 Interprétation avec levier :
  - si on raisonne en **notional**, l’action `step_usdt` correspond à une marge approximative de :
    - `step_margin_usdt ≈ step_usdt / L`
 - suivi de la marge investie (backtest) :
   - `used_margin_next ≈ used_margin_prev ± step_usdt / L` selon augmentation/réduction
 - “capital dispo” en cross margin (simplifié) signifie :
   - après l’action, l’**equity totale** reste au-dessus du seuil de liquidation.
 
 Choix de l’action (principe) :
 - si une jambe est plus petite, le MM essaye **d’abord** de la compléter (si le capital disponible le permet)
 - sinon, le MM réduit (clôture partiellement) la jambe la plus grosse
 
 Le MM doit toujours respecter :
 - le cap `max_initial_invest_usdt`
 - les intentions du signal (une jambe ne fait que ce que le signal autorise)

 ### 6.3 Frais et liquidation (principes)
 
 Frais :
  - les actions se faisant **au marché**, on considère des frais par opération.
  - valeur : `fee_rate_per_op = 0.0015` (0.15%) par opération.
  - rappel : un aller-retour (entrée + sortie) vaut environ `2 * fee_rate_per_op`.
  - coût approximatif d’une action au marché : `fee_cost_usdt ≈ fee_rate_per_op * step_usdt`.
  - règle simple : ne pas agir si l’amélioration attendue est trop faible vs frais + slippage.
 
 Liquidation :
  - en **cross margin**, **toute** la valeur du compte sert de “marge” :
    - `equity = wallet_balance + pnl_non_realise`
    - il est possible desr `'itntscoêrantlimargéninvesi+mrgevese.
  - dans cette spec, on utilise une règle **simplifiée** (pour coller au comportement observé) :
    - `wallet_balance` inclut le **PnL réalisé** jusqu’ici (et peut donc être > capital initial)
    - `margin_total = wallet_balance + max(0, pnl_non_realise)`
    - `loss_unrealized = max(0, -pnl_non_realise)`
    - liquidation si `loss_unrealized >= (1 - liquidation_threshold_pct) * margin_total`
    - valeur cible : `liquidation_threshold_pct = 0.05` (5%)
  - interprétation : liquidation quand les **pertes non réalisées** ont “consommé” ~95% de :
    - (non investi + investi) + (pnl non réalisé positif)
  - comportement attendu du MM :
    - si `loss_unrealized` se rapproche de ce seuil, privilégier les réductions (clôturer partiellement) plutôt qu’augmenter.

 ### 6.2 Exécution
 Si la quantité finale est > 0 :
 - exécution **au marché** pour l’action demandée (augmenter/réduire LONG/SHORT)

---

 ## 7) Règles MM simples (cadence + priorités)
 
 ### 7.1 Cadence (anti-spam)
 Le MM doit éviter d’ajouter/retirer trop souvent :
 - impose un **cooldown** configurable entre deux actions sur la même jambe (valeur par défaut à définir)
 - impose une **taille minimale** (sinon on ne fait rien)
 
 ### 7.2 Priorité entre intentions
 Si plusieurs intentions arrivent en même temps, le MM doit avoir une règle de priorité. Exemple de règle simple :
 - priorité aux **réductions** avant les augmentations
 - et si besoin, priorité à la jambe qui est la plus risquée (ex: la plus grosse)

 La “priorité” ne veut pas dire qu’on ne fait qu’une seule action.
 Elle définit seulement l’**ordre d’exécution** quand deux actions sont possibles (ex: réduire SHORT et augmenter LONG).

---

 ## 8) Cas importants (logique)
 
 ### 8.1 Si le signal autorise “réduire”
 - Le MM doit traiter la réduction en priorité.
 - Si une jambe devient trop grosse (cap dépassé), le MM peut forcer une réduction (règle de sécurité).
 
 ### 8.2 Si le signal change souvent
 - Le MM doit éviter de “courir après le marché”.
 - Comportement simple :
   - cooldown
   - pas de sizing trop agressif

---

 ## 9) Résumé en une phrase (pour validation)
 
 Le signal dit si LONG/SHORT peut être complété ou retiré, et le MM calcule (si activé) la quantité à ajouter/retirer puis exécute au marché en respectant des limites de risque et une cadence.

---

 ## 10) Points à valider (MM uniquement)
 
 1) **Seuil d’écart** (défaut proposé)
 - `gap_threshold_pct = 0.03` (3%)
 
 2) **Caps / capital** (défauts proposés)
 - `capital_usdt = 1000`
 - `initial_long_usdt = 100`
 - `initial_short_usdt = 100`
 - `max_initial_invest_usdt = 400`
 
 3) **Taille des actions** (défauts proposés)
 - `min_step_usdt = 25`
 - `max_step_usdt = 150`
 
 4) **Cooldown** (défaut proposé)
 - `cooldown_bars = 3`
 
 5) **Liquidation (simplifiée)** (défaut proposé)
 - `liquidation_threshold_pct = 0.05` (5%)

 6) **Règle d’exécution quand l’écart dépasse le seuil** (à valider)
 - priorité : compléter la jambe la plus petite si capital dispo, sinon réduire la plus grosse

 Fin.
