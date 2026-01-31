# Stratégie Zerem

## Principe général

La stratégie Zerem détecte des structures haussières ou baissières basées sur des extrêmes (creux/sommets) identifiés via les zones d’extrême de l’indicateur CCI.

## Détection des structures

### Structure haussière
- Deux creux (valeurs minimales) successifs où le deuxième creux est plus haut que le premier.

### Structure baissière
- Deux sommets (valeurs maximales) successifs où le deuxième sommet est plus bas que le premier.

## Zones d’extrême avec le CCI

Le CCI ne fait que délimiter les zones d’extrême ; les valeurs extrêmes sont prises sur une série cible pendant ces zones.

### Creux (minimum)
- Condition CCI : CCI < -100 (par défaut).
- Zone d’extrême : débute lorsque CCI croise sous -100, se termine lorsque CCI repasse au-dessus de -100.
- Valeur extrême : valeur minimale de la série cible pendant cette zone.

### Sommet (maximum)
- Condition CCI : CCI > +100 (par défaut).
- Zone d’extrême : débute lorsque CCI croise au-dessus de +100, se termine lorsque CCI repasse sous +100.
- Valeur extrême : valeur maximale de la série cible pendant cette zone.

## Logique de structure

- On collecte les extrêmes successifs (creux ou sommets) selon la direction.
- Structure haussière : creux_2 > creux_1.
- Structure baissière : sommet_2 < sommet_1.
