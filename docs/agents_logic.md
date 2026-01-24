# Logique des agents (libs/agents)

Ce document décrit la logique implémentée (pas une spec idéale) des agents présents dans le dossier libs/agents.

## Convention commune

- **Données en entrée**: une série de bougies (temps, prix d’ouverture, plus haut, plus bas, clôture) et des indicateurs (VWMA, MACD, CCI, ATR...) selon l’agent.
- **Résultat**: chaque agent détecte des segments (cycles, tranches, événements) puis les classe pour faire ressortir les cas les plus intéressants.
- **Temps**: les repères temporels sont exprimés en UTC.

---

# 1) MacdHistTrancheAgent

- **Fichier**: libs/agents/macd_hist_tranche_agent.py
- **But**: découper la série en tranches haussières/baissières basées sur le signe de l’histogramme MACD, puis évaluer la qualité de chaque tranche.

## Données attendues

- Bougies (temps, clôture).
- Histogramme MACD.

## Logique

1. **Découpage en tranches**
   - On regroupe les bougies consécutives où l’histogramme MACD garde le même signe.

2. **Force de l’histogramme MACD**
   - On exprime une force normalisée en divisant la valeur de l’histogramme MACD par le prix de clôture.
   - Plus cette force est élevée (en valeur absolue), plus le signal MACD est considéré comme “fort”.

## Critères de sélection (idée générale)

- Force (normalisée) suffisamment élevée.

---

# 2) TripleCciTrancheAgent

- **Fichier**: libs/agents/triple_cci_tranche_agent.py
- **But**: dans chaque tranche basée sur l’histogramme MACD, construire un CCI global pondéré (calculé à chaque bougie) afin d’évaluer la profondeur et la dynamique du mouvement.

## Données attendues

- Bougies (temps, plus haut, plus bas, clôture).
- Histogramme MACD.
- Trois CCIs (rapide, moyen, lent).

## Logique

1. **Découpage en tranches (histogramme MACD)**
   - On sépare les phases haussières et baissières selon le signe de l’histogramme MACD.

2. **CCI global pondéré (calculé à chaque bougie)**
   - On construit une valeur “globale” de CCI comme une moyenne pondérée des trois CCIs.
   - Poids du CCI rapide: 1.
   - Poids du CCI moyen: période du CCI moyen divisée par la période du CCI rapide.
   - Poids du CCI lent: période du CCI lent divisée par la période du CCI rapide.
   - Le CCI global pondéré est ensuite:
     - (CCI rapide × poids rapide + CCI moyen × poids moyen + CCI lent × poids lent) ÷ (somme des poids).
   - Cette valeur est traitée comme un indicateur à part entière, très important pour caractériser la “profondeur” à chaque bougie.

## Critères de sélection (idée générale)

- Tranche suffisamment longue.
- CCI global pondéré atteignant des niveaux suffisamment marqués.

---

# 2ter) DoubleCciTrancheAgent

- **Fichier**: libs/agents/double_cci_tranche_agent.py
- **But**: dans chaque tranche basée sur l’histogramme MACD, construire un CCI global pondéré (calculé à chaque bougie) à partir de deux CCIs (rapide, lent), afin d’évaluer la profondeur et la dynamique du mouvement.

## Données attendues

- Bougies (temps, plus haut, plus bas, clôture).
- Histogramme MACD.
- Deux CCIs (rapide, lent).

## Logique

1. **Découpage en tranches (histogramme MACD)**
   - Tranches haussières / baissières selon le signe de l’histogramme MACD.

2. **CCI global pondéré (calculé à chaque bougie)**
   - On construit une valeur “globale” de CCI comme une moyenne pondérée des deux CCIs.
   - Poids du CCI rapide: 1.
   - Poids du CCI lent: période du CCI lent divisée par la période du CCI rapide.
   - Le CCI global pondéré est ensuite:
     - (CCI rapide × poids rapide + CCI lent × poids lent) ÷ (somme des poids).

3. **Extrêmes et dernier extrême local**
   - On extrait l’extrême global de la tranche (max ou min selon le sens de la tranche).
   - On tente aussi d’extraire le dernier extrême local (pivot) dans la tranche.

## Critères de sélection (idée générale)

- Tranche suffisamment longue.
- CCI global pondéré atteignant des niveaux suffisamment marqués.

---

# 2bis) TripleStochGlobalAgent

- **Fichier**: libs/agents/triple_stoch_global_agent.py
- **But**: construire un unique Stochastique global pondéré (K et D) à partir de trois Stochastiques (rapide, moyen, lent), puis calculer les extrêmes, croisements, pentes et positions relatives uniquement sur ce Stoch global.

## Données attendues

- Bougies (temps).
- Trois Stochastiques: (K_fast, D_fast), (K_medium, D_medium), (K_slow, D_slow).

## Logique

1. **Poids (dépendants de la longueur de calcul, comme CCI)**
   - Pour chaque niveau (fast/medium/slow), on définit la longueur effective `L = k_period`.
   - Poids fast: 1.
   - Poids medium: L_medium / L_fast.
   - Poids slow: L_slow / L_fast.
   - On normalise avec la somme des poids.

2. **Stoch global pondéré (calculé à chaque bougie)**
   - K_global = (K_fast × w_fast + K_medium × w_medium + K_slow × w_slow) ÷ (w_fast + w_medium + w_slow).
   - D_global = (D_fast × w_fast + D_medium × w_medium + D_slow × w_slow) ÷ (w_fast + w_medium + w_slow).
   - Toutes les métriques suivantes sont calculées uniquement à partir de (K_global, D_global).

3. **Position relative (global)**
   - Spread global: spread = K_global − D_global.
   - Position par rapport au milieu: K_global − 50 et D_global − 50.

4. **Extrêmes (global)**
   - Extrêmes sur K_global et/ou D_global avec des seuils (ex: high=80, low=20).
   - Mode d’extrême configurable (optionnel):
     - `k_only` (défaut)
     - `d_only`
     - `k_and_d` (plus strict)

5. **Croisements (global)**
   - Bull cross quand (K_global − D_global) passe de ≤0 à >0.
   - Bear cross quand (K_global − D_global) passe de ≥0 à <0.
   - Filtre anti-bruit optionnel: exiger abs(spread) ≥ min_cross_gap_global au moment du cross.

6. **Pentes (global)**
   - Deux modes possibles (choix utilisateur):
     - `delta` (défaut): pente = valeur[i] − valeur[i−N]
     - `linreg`: pente par régression linéaire sur une fenêtre N
   - Les pentes peuvent être calculées sur K_global, D_global, et spread.

## Critères de sélection (idée générale)

- Stoch global atteignant un extrême (K_global et/ou D_global) puis retournement confirmé par un croisement global.
- Spread global suffisamment marqué (croisements propres) et pente cohérente (accélération/décélération).

---

# 2quater) DoubleStochGlobalAgent

- **Fichier**: libs/agents/double_stoch_global_agent.py
- **But**: construire un unique Stochastique global pondéré (K et D) à partir de deux Stochastiques (rapide, lent), puis calculer les extrêmes, croisements, pentes et positions relatives uniquement sur ce Stoch global.

## Données attendues

- Bougies (temps).
- Deux Stochastiques: (K_fast, D_fast), (K_slow, D_slow).

## Logique

1. **Poids (dépendants de la longueur de calcul, comme CCI)**
   - Pour chaque niveau (fast/slow), on définit la longueur effective `L = k_period`.
   - Poids fast: 1.
   - Poids slow: L_slow / L_fast.
   - On normalise avec la somme des poids.

2. **Stoch global pondéré (calculé à chaque bougie)**
   - K_global = (K_fast × w_fast + K_slow × w_slow) ÷ (w_fast + w_slow).
   - D_global = (D_fast × w_fast + D_slow × w_slow) ÷ (w_fast + w_slow).
   - Toutes les métriques suivantes sont calculées uniquement à partir de (K_global, D_global).

3. **Position relative (global)**
   - Spread global: spread = K_global − D_global.
   - Position par rapport au milieu: K_global − 50 et D_global − 50.

4. **Extrêmes (global)**
   - Extrêmes sur K_global et/ou D_global avec des seuils (ex: high=80, low=20).
   - Mode d’extrême configurable (optionnel):
     - `k_only` (défaut)
     - `d_only`
     - `k_and_d` (plus strict)

5. **Croisements (global)**
   - Bull cross quand (K_global − D_global) passe de ≤0 à >0.
   - Bear cross quand (K_global − D_global) passe de ≥0 à <0.
   - Filtre anti-bruit optionnel: exiger abs(spread) ≥ min_cross_gap_global au moment du cross.

6. **Pentes (global)**
   - Deux modes possibles (choix utilisateur):
     - `delta` (défaut): pente = valeur[i] − valeur[i−N]
     - `linreg`: pente par régression linéaire sur une fenêtre N
   - Les pentes peuvent être calculées sur K_global, D_global, et spread.

## Critères de sélection (idée générale)

- Stoch global atteignant un extrême (K_global et/ou D_global) puis retournement confirmé par un croisement global.
- Spread global suffisamment marqué (croisements propres) et pente cohérente (accélération/décélération).

---

# 2quinquies) TripleMacdGlobalAgent

- **Fichier**: libs/agents/triple_macd_global_agent.py
- **But**: construire un histogramme MACD global pondéré à partir de trois histogrammes MACD (rapide, moyen, lent), puis calculer des métriques uniquement sur cet histogramme global (zéro-cross et pente).

## Données attendues

- Bougies (temps).
- Trois histogrammes MACD: hist_fast, hist_medium, hist_slow.

## Logique

1. **Poids (dépendants de la longueur de calcul)**
   - Pour chaque niveau (fast/medium/slow), on définit la longueur effective `L = slow_period`.
   - Poids fast: 1.
   - Poids medium: L_medium / L_fast.
   - Poids slow: L_slow / L_fast.
   - On normalise avec la somme des poids.

2. **Histogramme MACD global pondéré (calculé à chaque bougie)**
   - hist_global = (hist_fast × w_fast + hist_medium × w_medium + hist_slow × w_slow) ÷ (w_fast + w_medium + w_slow).
   - Toutes les métriques suivantes sont calculées uniquement à partir de hist_global.

3. **Zéro-cross (flip de momentum)**
   - Bull cross quand hist_global passe de ≤0 à >0.
   - Bear cross quand hist_global passe de ≥0 à <0.
   - Filtre anti-bruit optionnel: exiger abs(hist_global) ≥ min_cross_abs au moment du cross.

4. **Pente (global)**
   - Deux modes possibles (choix utilisateur):
     - `delta` (défaut): pente = valeur[i] − valeur[i−N]
     - `linreg`: pente par régression linéaire sur une fenêtre N

## Critères de sélection (idée générale)

- Zéro-cross propre (momentum net) et/ou pente significative sur hist_global.

---

# 2sexies) DoubleMacdGlobalAgent

- **Fichier**: libs/agents/double_macd_global_agent.py
- **But**: construire un histogramme MACD global pondéré à partir de deux histogrammes MACD (rapide, lent), puis calculer des métriques uniquement sur cet histogramme global (zéro-cross et pente).

## Données attendues

- Bougies (temps).
- Deux histogrammes MACD: hist_fast, hist_slow.

## Logique

1. **Poids (dépendants de la longueur de calcul)**
   - Pour chaque niveau (fast/slow), on définit la longueur effective `L = slow_period`.
   - Poids fast: 1.
   - Poids slow: L_slow / L_fast.
   - On normalise avec la somme des poids.

2. **Histogramme MACD global pondéré (calculé à chaque bougie)**
   - hist_global = (hist_fast × w_fast + hist_slow × w_slow) ÷ (w_fast + w_slow).
   - Toutes les métriques suivantes sont calculées uniquement à partir de hist_global.

3. **Zéro-cross (flip de momentum)**
   - Bull cross quand hist_global passe de ≤0 à >0.
   - Bear cross quand hist_global passe de ≥0 à <0.
   - Filtre anti-bruit optionnel: exiger abs(hist_global) ≥ min_cross_abs au moment du cross.

4. **Pente (global)**
   - Deux modes possibles (choix utilisateur):
     - `delta` (défaut): pente = valeur[i] − valeur[i−N]
     - `linreg`: pente par régression linéaire sur une fenêtre N

## Critères de sélection (idée générale)

- Zéro-cross propre (momentum net) et/ou pente significative sur hist_global.

---

# 3) VwmaBreakTouchTrancheAgent

- **Fichier**: libs/agents/vwma_break_touch_tranche_agent.py
- **But**: dans chaque tranche basée sur l’histogramme MACD, détecter comment le prix interagit avec une VWMA (touches) et qualifier si la VWMA agit comme support/résistance (rejet), avec un bonus si la VWMA présente un pivot (changement de pente).

## Données attendues

- Bougies (temps, plus haut, plus bas, clôture).
- Histogramme MACD.
- Une VWMA (souvent rapide).

## Logique

1. **Découpage en tranches (histogramme MACD)**
   - Tranches haussières / baissières selon le signe de l’histogramme MACD.

2. **Analyse du mouvement de la VWMA**
   - On mesure si la VWMA progresse dans le sens de la tranche et si sa pente est régulière.

3. **Détection d’un pivot VWMA**
   - On cherche un changement clair de pente (passage d’une phase de hausse à une phase de baisse, ou inversement).
   - On estime si ce pivot est net ou faible.

4. **Touches prix↔VWMA**
   - On considère qu’il y a une touche quand une bougie vient recouvrir la zone autour de la VWMA.
   - On compte les touches, on garde la dernière touche et sa récence.

5. **Validation du rejet après la dernière touche**
   - Après la dernière touche, on vérifie si le prix repart suffisamment dans le sens de la tranche et s’éloigne de la zone VWMA.

## Critères de sélection (idée générale)

- Tranche suffisamment longue.
- Dernière touche récente.
- Rejet valide après la touche, ou pivot VWMA très net.

---

# 4) DoubleVwmaCycleAgent

- **Fichier**: libs/agents/double_vwma_cycle_agent.py
- **But**: découper la série en cycles de tendance selon la position relative de deux VWMAs (rapide vs lente), puis analyser chaque cycle (tendance, collisions, tests de zones, pullbacks).

## Données attendues

- Bougies (temps, prix d’ouverture, plus haut, plus bas, clôture).
- Deux VWMAs: une rapide et une lente.

## Logique

1. **Découpage en cycles**
   - On définit une zone autour de chaque VWMA.
   - On distingue trois états: rapide au-dessus, rapide au-dessous, ou collision des zones.
   - Un nouveau cycle commence quand, après une collision, la relation au-dessus/au-dessous s’inverse.

2. **Sens de tendance du cycle**
   - Le sens est déterminé par la position relative des VWMAs au début du cycle.

3. **Collisions et tests de zones**
   - On identifie les périodes de collision.
   - On détecte les tests de zones par le prix (approche depuis au-dessus ou au-dessous) et on qualifie l’issue (rejet, traversée, ou test incomplet).

4. **Pullbacks**
   - On repère les excursions contre la tendance où le prix vient toucher/casser des zones VWMA.
   - Les pullbacks sont classés par profondeur (touch simple, cassure de zone rapide, cassure de zone lente).
   - Pendant le pullback, on suit le point le plus extrême atteint contre la tendance (le “sommet” ou le “creux” du pullback selon le sens du cycle).
   - Un événement important est le moment où cet extrême semble atteint, puis où le prix commence à repartir dans le sens du cycle.
   - Ce retournement est considéré plus solide lorsqu’il est confirmé par plusieurs bougies consécutives allant dans le sens du retour (nombre de bougies ajustable via la configuration).
   - Optionnellement, la séquence de confirmation peut être filtrée pour exiger une bougie d’impact (sur la bougie de confirmation ou sur un agrégat de bougies).
   - Optionnellement, la confirmation peut aussi être filtrée par une micro-direction cohérente avec le sens du retour (basée sur le sens croissant/décroissant de la VWMA micro, période 4).
   - Une cassure de la zone lente peut être considérée “confirmée” si plusieurs bougies vont dans le même sens.

## Événements détectés (ce que l’agent rapporte)

- Début d’un nouveau cycle de tendance (changement de régime après une phase de collision).
- Phases de collision entre les zones des deux VWMAs (zones qui se chevauchent ou se mélangent).
- Tests de zone par le prix (approche d’une zone depuis au-dessus ou depuis au-dessous) et qualification de l’issue (rejet, traversée, ou test inachevé).
- Début et fin d’un pullback (contre-mouvement temporaire), avec sa profondeur.
- Point extrême d’un pullback et retournement confirmé (lorsque, après l’extrême, plusieurs bougies consécutives indiquent une reprise dans le sens du cycle).
- Pullback profond potentiellement “confirmé” lorsqu’il se prolonge de façon cohérente (ce qui signale un risque plus élevé pour la tendance).

## Critères de sélection (idée générale)

- Cycle suffisamment long.
- Tendance suffisamment claire (écart entre VWMAs).
- Pullback récent et/ou profond.
- Pentes des VWMAs cohérentes avec le sens du cycle.

---

# 5) TripleVwmaCycleAgent

- **Fichier**: libs/agents/triple_vwma_cycle_agent.py
- **But**: découper la série en cycles de tendance en s’appuyant sur trois VWMAs (rapide, intermédiaire, lente) et qualifier ces cycles (tendance claire, collisions, tests de zones, pullbacks).

## Données attendues

- Bougies (temps, prix d’ouverture, plus haut, plus bas, clôture).
- Trois VWMAs: rapide, intermédiaire, lente.

## Logique

1. **Découpage en cycles (tendance principale)**
   - On observe la relation entre la VWMA intermédiaire et la VWMA lente.
   - Les cycles sont délimités quand, après une phase de “mélange” (zones proches), la relation entre les deux change de sens.

2. **Tendance principale et tendance locale**
   - La tendance principale est évaluée via la relation entre VWMA intermédiaire et VWMA lente.
   - La tendance locale est évaluée via la relation entre VWMA rapide et VWMA intermédiaire.

3. **Clarté du trend**
   - On mesure si les VWMAs sont suffisamment séparées (trend “lisible”) et si elles progressent dans le sens du cycle.
   - On favorise les cycles où les pentes des trois VWMAs vont globalement dans la même direction que la tendance principale.

4. **Collisions et tests de zones**
   - On repère les phases où les zones autour des VWMAs se chevauchent (collision).
   - On repère les moments où le prix vient “tester” une zone VWMA depuis au-dessus ou depuis au-dessous, puis on qualifie le résultat (rejet, traversée, ou test inachevé).

5. **Pullbacks (contre-tendance temporaire)**
   - On détecte les excursions contre la tendance principale où le prix revient vers les VWMAs.
   - Le pullback est classé selon la profondeur (jusqu’à la VWMA rapide, jusqu’à l’intermédiaire, ou jusqu’à la lente).
   - Pendant le pullback, on suit le point le plus extrême atteint contre la tendance principale.
   - Un événement important est le moment où l’extrême semble atteint, puis où le prix repart dans le sens de la tendance principale.
   - Ce retournement est considéré plus solide lorsqu’il est confirmé par plusieurs bougies consécutives allant dans le sens du retour (nombre de bougies ajustable via la configuration).
   - Optionnellement, la séquence de confirmation peut être filtrée pour exiger une bougie d’impact (sur la bougie de confirmation ou sur un agrégat de bougies).
   - Optionnellement, la confirmation peut aussi être filtrée par une micro-direction cohérente avec le sens du retour (basée sur le sens croissant/décroissant de la VWMA micro, période 4).
   - Une cassure profonde (proche de la VWMA lente) peut être considérée plus grave si elle est confirmée par plusieurs bougies cohérentes.

## Événements détectés (ce que l’agent rapporte)

- Début et fin d’un cycle de tendance principale (bascule de tendance après une phase de mélange).
- Tendance principale et tendance locale (lorsqu’elles sont alignées ou lorsqu’elles divergent).
- Phases de collision entre les zones autour des VWMAs (notamment entre l’intermédiaire et la lente, et entre la rapide et l’intermédiaire).
- Tests de zones VWMA par le prix, avec qualification de l’issue (rejet, traversée, ou test inachevé).
- Début et fin d’un pullback, avec classification par profondeur (retour sur la VWMA rapide, sur l’intermédiaire, ou sur la lente).
- Point extrême d’un pullback et retournement confirmé (lorsque, après l’extrême, plusieurs bougies consécutives indiquent une reprise dans le sens de la tendance principale).
- Signalement d’un pullback profond “confirmé”, interprété comme une alerte de fragilisation du cycle.

## Critères de sélection (idée générale)

- Cycle suffisamment long.
- Tendance principale claire (VWMAs bien ordonnées et suffisamment séparées).
- Pullback récent et/ou profond.
- Pentes des VWMAs cohérentes avec le sens du cycle.

---

# 6) ImpactBarAgent

- **Fichier**: libs/agents/impact_bar_agent.py
- **But**: repérer des bougies (ou un agrégat de plusieurs bougies consécutives) considérées comme “d’impact”, c’est-à-dire une poussée directionnelle forte et lisible.

## Données attendues

- Bougies (temps, prix d’ouverture, plus haut, plus bas, clôture).

## Logique

1. **Agrégation optionnelle**
   - L’agent peut analyser une bougie seule, ou regrouper plusieurs bougies consécutives pour former une “bougie agrégée”.
   - L’ouverture de l’agrégat est l’ouverture de la première bougie, la clôture est la clôture de la dernière bougie.

2. **Couleur (direction)**
   - Une bougie d’impact est associée à une direction: haussière (clôture au-dessus de l’ouverture) ou baissière (clôture en dessous).
   - Optionnellement, on peut exiger que toutes les bougies composant un agrégat aient la même direction.

3. **Pourcentage de corps**
   - On mesure à quel point le “corps” (distance entre ouverture et clôture) représente une grande part de l’amplitude totale (distance entre plus haut et plus bas) de la bougie ou de l’agrégat.
   - Une bougie (ou agrégat) est “d’impact” si ce pourcentage se situe dans une plage cible.

## Événements détectés (ce que l’agent rapporte)

- Bougie d’impact détectée (ou agrégat d’impact) avec:
  - position de début/fin de l’agrégat,
  - direction,
  - niveau de “corps relatif”.

---

# 7) StochCrossAgent

- **Fichier**: libs/agents/stoch_cross_agent.py
- **But**: détecter un croisement du Stochastique (ligne principale vs ligne de signal), et qualifier ce croisement avec des filtres optionnels.

## Données attendues

- Bougies (temps, prix d’ouverture, plus haut, plus bas, clôture).
- Stochastique (ligne principale et ligne de signal).

## Logique

1. **Détection du croisement**
   - Croisement haussier: la ligne principale passe au-dessus de la ligne de signal.
   - Croisement baissier: la ligne principale passe au-dessous de la ligne de signal.

2. **Filtre optionnel “bougie d’impact”**
   - Si le filtre est activé, la bougie (ou l’agrégat de bougies autour du croisement) doit être une bougie d’impact.
   - La direction de la bougie d’impact doit être cohérente avec le sens du croisement.

3. **Filtre optionnel “micro-direction”**
   - Si le filtre est activé, l’agent vérifie que la VWMA micro (période 4) est déjà orientée dans le même sens (croissante pour un signal haussier, décroissante pour un signal baissier).

## Événements détectés (ce que l’agent rapporte)

- Croisement Stochastique détecté, avec:
  - sens (haussier / baissier),
  - éventuelle validation par une bougie d’impact,
  - éventuelle validation par la micro-direction.

---

# 8) MicroDirectionAgent

- **Fichier**: libs/agents/micro_direction_agent.py
- **But**: détecter une micro-direction en se basant uniquement sur l’orientation (croissante ou décroissante) de la VWMA micro (période fixe à 4).

## Données attendues

- Bougies (temps).
- Une VWMA micro (période fixe à 4).

## Logique

1. **Sens de la VWMA micro**
   - Micro-direction haussière si la VWMA micro est en phase de hausse (croissante).
   - Micro-direction baissière si la VWMA micro est en phase de baisse (décroissante).

2. **Seuil optionnel**
   - Optionnellement, on peut ignorer les cas où la variation de VWMA est trop faible (pour éviter le bruit).

## Événements détectés (ce que l’agent rapporte)

- Micro-direction détectée (haussière ou baissière) sur une bougie donnée.

## Usages typiques

- Filtre de validation pour:
  - un croisement Stochastique (éviter les croisements “contre le flux” court-terme),
  - un retournement de pullback (exiger que la reprise soit soutenue par un flux court-terme).



# 9) MacdMomentiumTwoTFCycleAgent

- **Fichier**: libs/agents/macd_momentum_two_tf_cycle_agent.py
- **But**: combiner deux timeframes (TF contexte + TF exécution) en utilisant:
  - `MacdHistTrancheAgent` sur TF contexte pour définir la tendance (sens + force),
  - `MacdHistTrancheAgent` sur TF exécution pour déclencher un signal au changement de signe,
  - `TripleCciTrancheAgent` sur TF contexte et TF exécution pour filtrer la validité via CCI global.

## Données attendues

- Deux séries de bougies:
  - `df_ctx` (TF contexte / supérieur)
  - `df_exec` (TF exécution)
- Pour chaque TF:
  - Bougies (temps, prix d’ouverture, plus haut, plus bas, clôture).
  - Histogramme MACD (`macd_hist`).
  - Trois CCIs (rapide, moyen, lent) nécessaires au calcul du CCI global.

## Logique

1. **Découpage en tranches MACD sur TF contexte (tendance)**
   - On segmente `df_ctx` en tranches haussières/baissières selon le signe de `macd_hist` via `MacdHistTrancheAgent`.
   - Chaque tranche contexte fournit:
     - un sens (`tranche_sign`),
     - une force normalisée (`force_mean_abs`).

2. **Découpage en tranches MACD sur TF exécution (signal)**
   - On segmente `df_exec` via `MacdHistTrancheAgent`.
   - Un **changement de signe** (début d’une nouvelle tranche) est le **déclencheur de signal**.

3. **Synchronisation exécution → contexte (anti-lookahead)**
   - Pour un instant d’exécution `t_exec`, on récupère la dernière bougie contexte clôturée `t_ctx` telle que `t_ctx <= t_exec`.
   - La tranche contexte active à `t_ctx` définit la **tendance** au moment du signal.

4. **Définition du candidat trade (TF exécution)**
   - Au début d’une tranche exécution:
     - si `exec_tranche_sign == '+'` alors candidat `LONG`,
     - si `exec_tranche_sign == '-'` alors candidat `SHORT`.
   - L’exécution du trade se fait sur la bougie suivante: `exec_i = signal_i + 1`.

5. **Filtres de validation**
   - **Alignement tendance/signal**:
     - Le trade n’est validé que si `exec_tranche_sign` est identique à `ctx_tranche_sign` au moment du signal.
   - **Force MACD min sur les deux TF**:
     - `ctx_force_mean_abs >= min_abs_force_ctx`
     - `exec_force_mean_abs >= min_abs_force_exec`
   - **Filtre CCI global (TF contexte = ne doit PAS être en extrême dans le sens du trade)**:
     - On calcule un CCI global par tranche contexte via `TripleCciTrancheAgent`.
     - Donc, pour valider un trade, le contexte doit être **hors extrême** (zone autorisée):
       - `LONG`: `cci_global_last_extreme < cci_global_extreme_level_ctx`
       - `SHORT`: `cci_global_last_extreme > -cci_global_extreme_level_ctx`
   - **Filtre CCI global (TF exécution = ne doit PAS être en extrême dans le sens du trade)**:
     - On calcule un CCI global par tranche exécution via `TripleCciTrancheAgent`.
     - Donc, pour valider un trade, l’exécution doit être **hors extrême** (zone autorisée):
       - `LONG`: `cci_global_last_extreme < cci_global_extreme_level_exec`
       - `SHORT`: `cci_global_last_extreme > -cci_global_extreme_level_exec`

## Événements détectés (ce que l’agent rapporte)

- Début d’une tranche MACD sur TF exécution (changement de signe) = **signal candidat**.
- Mapping du signal candidat vers la tranche TF contexte active (tendance au moment du signal).
- Décision finale par signal:
  - `ACCEPT` (trade validé)
  - `REJECT` (avec raison: non-alignement, force insuffisante, CCI contexte trop extrême, CCI exécution trop extrême, etc.)

## Critères de sélection (idée générale)

- Alignement du signe MACD entre TF contexte et TF exécution.
- Force normalisée MACD suffisante sur TF contexte et TF exécution.
- CCI global:
  - TF contexte: **pas** en extrême dans le sens du trade,
  - TF exécution: **pas** en extrême dans le sens du trade.

---

# 10) TripleMacdRolesAgent
 
- **Fichier**: libs/agents/triple_macd_roles_agent.py
- **But**: utiliser trois MACDs (slow/medium/fast) sur une même série pour combiner (i) un régime (tendance interne via zone MACD), (ii) un contexte (impulse vs respiration via hist par rapport à la zone), (iii) un déclencheur (croisement via changement de signe de hist), avec des filtres de force stricts et configurables par niveau.
 
## Données attendues
 
- Bougies (temps, clôture).
- Trois MACDs (slow/medium/fast), chacun avec:
  - `macd_line_*`, `macd_signal_*`, `macd_hist_*`.
 
## Logique
 
1. **Tendance interne (par niveau)**
   - On calcule une zone `zone_sign` uniquement via (macd_line, macd_signal) par rapport à 0:
     - `+1` si macd_line>0 et macd_signal>0.
     - `-1` si macd_line<0 et macd_signal<0.
     - `0` sinon (zone de transition). Par défaut l’agent rejette ces cas (`reject_zone_transition=True`).
 
2. **Croisements & tranches (par niveau)**
   - `hist = macd_line - macd_signal`.
   - Le **signal de croisement** est le changement de signe de `hist`:
     - bull cross: hist passe de ≤0 à >0.
     - bear cross: hist passe de ≥0 à <0.
   - Les “tranches” correspondent aux périodes où le signe de `hist` est constant (même principe que `MacdHistTrancheAgent`).
 
3. **Respiration vs impulse (par niveau)**
   - Si `zone_sign != 0`:
     - **impulse** si `hist_sign == zone_sign`.
     - **respiration** si `hist_sign == -zone_sign`.
   - Chaque niveau slow/medium/fast peut autoriser ou interdire un trade en respiration (`allow_trade_when_respiration`).
 
4. **Force (filtre strict, par niveau)**
   - Force normalisée: `force = abs(hist / close)` (même définition que `MacdHistTrancheAgent`).
   - Filtre min: `force >= min_abs_force` (par niveau).
   - Filtre d’“accélération”: exiger que la force soit strictement croissante sur les dernières bougies (`require_force_rising`, `force_rising_bars`).
 
5. **Macro (mode sélectionnable)**
   - La direction macro (`macro_mode`) peut être:
     - `slow_zone`: macro = `zone_sign_slow`.
     - `slow_hist`: macro = signe de `hist_slow`.
     - `slow_zone_and_hist`: macro non-nulle uniquement si zone et hist sont alignés.
 
6. **Déclencheur (niveau sélectionnable)**
   - Le niveau de déclenchement est sélectionnable:
     - explicitement via `entry_trigger_level` (slow/medium/fast),
     - ou via `style` (ex: scalping → fast, swing → medium, position → slow) si `entry_trigger_level` est vide.
   - Option `require_trigger_in_macro_dir`: exiger que le cross du niveau choisi soit dans le sens macro.
 
## Paramètres clés (exemples)
 
- `macro_mode`: slow_zone | slow_hist | slow_zone_and_hist
- `style`: scalp | swing | position (choisit le trigger si `entry_trigger_level` n’est pas fixé)
- `entry_trigger_level`: slow | medium | fast
- Par niveau (`slow`, `medium`, `fast`):
  - `enabled`
  - `reject_zone_transition`
  - `min_abs_force`
  - `require_force_rising`, `force_rising_bars`
  - `allow_trade_when_respiration`
  - `require_align_zone_to_macro`, `require_align_hist_to_macro`
- Global:
  - `reject_when_all_three_respire`

---

# 11) TripleCciRolesAgent
 
- **Fichier**: libs/agents/triple_cci_roles_agent.py
- **But**: utiliser trois CCIs (slow/medium/fast) sur une même série pour combiner (i) un régime de pression vs moyenne (signe du CCI), (ii) un contexte impulse vs respiration (pente du CCI vs signe), (iii) un déclencheur de timing (croisement 0, croisement de pente ou sortie d’extrême), avec des filtres de force stricts et configurables par niveau.
 
## Données attendues
 
- Bougies (temps).
- Par niveau slow/medium/fast:
  - un CCI (`cci_*`) ou, si absent, la possibilité de le calculer via OHLC.
 
Colonnes typiques:
 
- `ts`
- `high`, `low`, `close` (uniquement si on veut que l’agent calcule le CCI)
- `cci_30`, `cci_120`, `cci_300` (par défaut)
 
## Logique
 
1. **Tendance (par niveau)**
   - On définit une direction interne via le signe du CCI:
     - `trend_sign = sign(cci)` (autour de 0).
   - Option `reject_trend_transition`: par défaut on rejette `trend_sign == 0`.
 
2. **Force / impulsion (par niveau)**
   - On calcule une pente du CCI (linreg) sur `slope_window`:
     - `slope = linreg_slope(cci, window)`.
   - Force (configurable):
     - `force_mode=abs_slope`: `force = abs(slope)`.
     - `force_mode=abs_cci`: `force = abs(cci)`.
   - Filtre min: `force >= min_abs_force`.
   - Filtre d’“accélération”: force strictement croissante sur les dernières bougies (`require_force_rising`, `force_rising_bars`).
 
3. **Extrêmes (par niveau)**
   - On détecte l’état extrême via `extreme_level`:
     - extrême haut si `cci >= +extreme_level`.
     - extrême bas si `cci <= -extreme_level`.
   - On mesure la “stickiness” via `extreme_dwell` (nombre de bougies consécutives en extrême).
 
4. **Respiration vs impulse (par niveau)**
   - Respiration = perte d’impulsion contre la tendance:
     - si `trend_sign != 0` et `slope_sign == -trend_sign`.
   - Chaque niveau slow/medium/fast peut autoriser ou interdire un trade en respiration (`allow_trade_when_respiration`).
 
5. **Macro (mode sélectionnable)**
   - `macro_mode`:
     - `slow_sign`: macro = `trend_sign_slow`.
     - `slow_sign_and_slope`: macro non-nulle uniquement si `trend_sign_slow` et `slope_sign_slow` sont alignés.
 
6. **Déclencheur (niveau + mode sélectionnables)**
   - Niveau de déclenchement: `entry_trigger_level` (slow/medium/fast) ou `style` (scalp→fast, swing→medium, position→slow).
   - `trigger_mode`:
     - `zero_cross`: croisement de 0 du CCI.
     - `slope_cross`: croisement de 0 de la pente.
     - `extreme_exit`: sortie d’extrême (repasser au-dessus de `-extreme_level` ou en-dessous de `+extreme_level`).
     - `any`: union des 3.
   - Option `require_trigger_in_macro_dir`: exiger que le trigger soit dans le sens de la macro.
 
## Paramètres clés (exemples)
 
- `macro_mode`: slow_sign | slow_sign_and_slope
- `style`: scalp | swing | position
- `entry_trigger_level`: slow | medium | fast
- `trigger_mode`: zero_cross | slope_cross | extreme_exit | any
- Par niveau (`slow`, `medium`, `fast`):
  - `cci_col`, `cci_period`
  - `extreme_level`, `slope_window`
  - `reject_trend_transition`
  - `force_mode`, `min_abs_force`, `require_force_rising`, `force_rising_bars`
  - `allow_trade_when_respiration`, `allow_trade_when_extreme_in_macro_dir`
  - `require_align_trend_to_macro`, `require_align_slope_to_macro`
- Global:
  - `reject_when_all_three_respire`

---

# 12) TripleStochRolesAgent
 
- **Fichier**: libs/agents/triple_stoch_roles_agent.py
- **But**: utiliser trois Stochastics (slow/medium/fast) pour combiner (i) un régime (position au-dessus/au-dessous de 50 + stickiness en extrêmes), (ii) un momentum (position relative K vs D), (iii) un déclencheur de timing (croisement K/D, croisement de régime, sortie d’extrême), avec des filtres de force stricts et configurables par niveau.
 
## Données attendues
 
- Bougies (temps).
- Par niveau slow/medium/fast:
  - `K` et `D` (si déjà calculés), ou la possibilité de les calculer via OHLC.
 
Colonnes typiques:
 
- `ts`
- `high`, `low`, `close` (uniquement si on veut que l’agent calcule le stoch)
- `stoch_k_fast`, `stoch_d_fast`
- `stoch_k_medium`, `stoch_d_medium`
- `stoch_k_slow`, `stoch_d_slow`
 
## Logique
 
1. **Régime (par niveau)**
   - On définit un régime via `regime_pivot` (par défaut 50):
     - `regime_sign=+1` si `K>pivot` et `D>pivot`.
     - `regime_sign=-1` si `K<pivot` et `D<pivot`.
     - `0` sinon (transition). Par défaut l’agent rejette ces cas (`reject_regime_transition=True`).
 
2. **Momentum (par niveau)**
   - `momentum_sign = sign(K - D)`.
 
3. **Extrêmes (par niveau)**
   - Extrême haut si `K >= extreme_high` (par défaut 80), extrême bas si `K <= extreme_low` (par défaut 20).
   - “Stickiness” via `extreme_dwell` (nombre de bougies consécutives en extrême).
 
4. **Respiration vs impulse (par niveau)**
   - Respiration = momentum contre le régime:
     - si `regime_sign != 0` et `momentum_sign == -regime_sign`.
   - Chaque niveau slow/medium/fast peut autoriser ou interdire un trade en respiration (`allow_trade_when_respiration`).
 
5. **Force (filtre strict, par niveau)**
   - Force (configurable):
     - `force_mode=abs_spread`: `force = abs(K-D)`.
     - `force_mode=abs_k_slope`: `force = abs(linreg_slope(K))`.
   - Filtre min: `force >= min_abs_force`.
   - Filtre d’“accélération”: force strictement croissante sur les dernières bougies (`require_force_rising`, `force_rising_bars`).
 
6. **Macro (mode sélectionnable)**
   - `macro_mode`:
     - `slow_regime`: macro = `regime_sign_slow`.
     - `slow_regime_and_momentum`: macro non-nulle uniquement si régime et momentum slow sont alignés.
 
7. **Déclencheur (niveau + mode sélectionnables)**
   - Niveau de déclenchement: `entry_trigger_level` (slow/medium/fast) ou `style` (scalp→fast, swing→medium, position→slow).
   - `trigger_mode`:
     - `kd_cross`: croisement K/D.
     - `regime_cross`: croisement de `K` autour de `regime_pivot`.
     - `extreme_exit`: sortie d’extrême (K repasse au-dessus de `extreme_low` ou en-dessous de `extreme_high`).
     - `any`: union des 3.
   - Option `require_trigger_in_macro_dir`: exiger que le trigger soit dans le sens de la macro.
 
## Paramètres clés (exemples)
 
- `macro_mode`: slow_regime | slow_regime_and_momentum
- `style`: scalp | swing | position
- `entry_trigger_level`: slow | medium | fast
- `trigger_mode`: kd_cross | regime_cross | extreme_exit | any
- Par niveau (`slow`, `medium`, `fast`):
  - `k_col`, `d_col`, `k_period`, `d_period`
  - `regime_pivot`, `extreme_high`, `extreme_low`
  - `reject_regime_transition`
  - `force_mode`, `min_abs_force`, `require_force_rising`, `force_rising_bars`
  - `allow_trade_when_respiration`, `allow_trade_when_extreme_in_macro_dir`
  - `require_align_regime_to_macro`, `require_align_momentum_to_macro`
- Global:
  - `reject_when_all_three_respire`

---

# 13) DmiExhaustionAgent

- **Fichier**: libs/agents/dmi_exhaustion_agent.py
- **But**: détecter un signal “d’essoufflement” quand le marché est en **régime mature** (ADX dominant) et que **DX croise sous ADX** (cross-under), et retourner l’événement ainsi que la tendance au moment du croisement.

## Données attendues

- Bougies (temps): colonne `ts`.
- Indicateurs DMI/ADX (déjà calculés dans le DataFrame):
  - `adx`
  - `plus_di`
  - `minus_di`
  - `dx` (optionnel, sinon l’agent peut le reconstruire à partir de `plus_di` et `minus_di`).

## Logique

1. **Maturité (régime)**
   - Mode par défaut (`maturity_mode="di_max"`):
     - `ADX > max(+DI, -DI)`
   - Variante (`maturity_mode="adx_threshold"`):
     - `ADX >= adx_min_threshold`

2. **Déclencheur one-shot: cross-under DX sous ADX**
   - Détection d’un croisement descendant:
     - `dx_prev > adx_prev` et `dx_now <= adx_now`

3. **Tendance au moment du croisement**
   - `side="LONG"` si `plus_di > minus_di`.
   - `side="SHORT"` sinon.

## Événements détectés (ce que l’agent rapporte)

- Un événement `dmi_dx_cross_under_adx` avec:
  - position `pos`, timestamp `ts`, `dt`
  - `side` (tendance au moment du croisement)
  - `meta` contenant (prev/now): `adx`, `dx`, `plus_di`, `minus_di`, et la config de maturité.