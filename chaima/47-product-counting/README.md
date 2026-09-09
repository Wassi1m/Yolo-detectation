# #47 — Product Counting (ligne de préparation)
 
 
## Objectif
Compter les produits qui défilent sur une ligne de préparation, en distinguant
plusieurs types (ex. bouteilles, boîtes), à partir d'une caméra de surveillance.
 
## Approche
1. **Détection + tracking** (`model.track()` de YOLO26) : chaque objet détecté reçoit
   un ID stable d'une frame à l'autre.
2. **Comptage par franchissement de ligne** : une ligne virtuelle est définie dans la
   config. Un objet n'est compté qu'une seule fois, au moment où son centre traverse
   cette ligne — pas à chaque frame où il est visible (ce qui compterait le même
   produit plusieurs fois).
3. Le compteur est **séparé par classe** (bouteille, boîte...) pour un total par type.

 
## Fichiers
- `src/product_counting/counting.py` — logique pure du comptage (testable sans caméra)
- `src/product_counting/runner.py` — pipeline complet (tracking, CSV, headless)
- `tests/test_counting.py` — 6 tests sur la logique de comptage
## Comment lancer
```bash
pip install -e .
 
product-counting --config config/baseline.yaml
product-counting --config config/baseline.yaml --headless
pytest tests/
```
 
## Configuration (config/baseline.yaml)
| Champ | Description |
|---|---|
| `classes_to_count` | Liste des classes COCO à compter, ex. `[bottle]` |
| `counting_line` | `[x1, y1, x2, y2]` — ligne virtuelle de comptage, à définir selon le cadrage réel |
| `headless` | Désactive l'affichage, obligatoire en déploiement serveur |
 

 