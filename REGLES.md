# Règles du dépôt

## Objectif
Conserver un code simple et réutilisable pour le calcul d’indicateurs et les scripts associés.

## Organisation
- Les scripts exécutables restent à la racine (ex: `compare_indicators_bybit.py`).
- Les fonctions réutilisables vont dans `libs/`.
- Les sous-dossiers de `libs/` regroupent par domaine (ex: `libs/utils`, `libs/indicators`).
- Dans `libs/indicators/`, créer des sous-dossiers par domaine (ex: `moving_averages`, `momentum`, `volatility`, `volume`).
- Un module à la racine peut servir de façade (ré-export) pour compatibilité, mais la logique réutilisable reste dans `libs/`.

## Conventions
- Imports en haut de fichier.
- 1 fonction par fichier `.py` (hors `__init__.py`).
- Fonctions pures si possible (pas d’effets de bord hors I/O).
- Validation d’arguments: lever `ValueError` pour des entrées incohérentes.
- Nommage: `snake_case` pour fonctions/variables.

## Dépendances
- Ajouter une dépendance uniquement si elle est réellement nécessaire.
- Mettre à jour `requirements.txt` lors de l’ajout d’une dépendance.
