# Architecture commune — projets de détection YOLO (stage)

Ce document définit la structure que **chaque projet** (existant ou nouveau, dans `chaima/`, `ghada/`, `louay/`, `mohamed amine/`, `yassmine/`) doit respecter. Le but : pouvoir prendre n'importe quel projet, comprendre en 30 secondes comment le lancer, et l'intégrer dans un système plus large (ex. l'app de surveillance multi-caméras) sans devoir lire tout le code.

Elle a été établie à partir de l'analyse des 5 dossiers du dépôt (voir le `PLAN_AMELIORATION.md` de chacun) :
- `louay/` est aujourd'hui la référence à suivre (package installable, config déclarative, poids partagés).
- `yassmine/mov_vide_nuit/` apporte la brique manquante côté "sortie exploitable" (alertes, logs, preuves, enregistrement).
- `chaima/`, `ghada/` montrent ce qu'il faut éviter : code à plat sans config, notebooks non reproductibles, résultats non persistés, chemins Google Colab codés en dur.

---

## 1. Squelette de dossier standard

```
<projet>/
├── pyproject.toml              # package installable, dépendances épinglées, point d'entrée console
├── README.md                   # setup, run, sortie produite, description champ par champ de la config
├── model_card.yaml             # fiche d'identité du modèle (voir §5) — condition pour comparer les projets
├── config/
│   └── baseline.yaml           # TOUTE valeur ajustable vit ici, jamais en dur dans le code
├── src/<nom_du_package>/
│   ├── __init__.py
│   ├── config.py                # dataclass Settings + load_settings(path) -> Settings
│   ├── model.py                 # load_model(settings) -> modèle  — SEUL point de chargement du modèle
│   ├── <logique_metier>.py     # fonctions pures (pas d'I/O, pas de modèle) → testables sans caméra/GPU
│   └── runner.py                # CLI (argparse) + fonction run(settings, source) réutilisable en import
├── data/
│   └── sample.*                 # au moins un exemple d'entrée versionné : le projet doit tourner "out of the box"
├── tests/
│   └── test_*.py                # tests sur la logique pure de <logique_metier>.py
├── outputs/                     # gitignored — résultats générés
├── .venv/                       # gitignored
└── .gitignore
```

**Interdits** : script Python à plat sans `pyproject.toml` (comme `chaima/42-crowd-density-estimation/crowd_density.py`), notebook comme seule forme d'implémentation (comme `ghada/*`), valeurs de configuration codées en dur dans le code (seuils, chemins, URL).

Un notebook reste toléré **uniquement** comme brouillon d'exploration initiale (`notebooks/exploration.ipynb`, non versionné ou clairement marqué comme tel) — dès qu'un résultat fonctionne, la logique doit être extraite vers `src/`.

---

## 2. Contrat de configuration (`config/baseline.yaml`)

Champs obligatoires, avec ces noms exacts, pour que n'importe qui puisse lire n'importe quel `baseline.yaml` du dépôt sans documentation supplémentaire :

```yaml
project: <nom-du-projet>
model: <fichier-de-poids>.pt      # résolu contre le dossier de poids partagé, jamais un chemin absolu
device: cpu                        # ou "cuda:0" — jamais codé en dur dans le code
image_size: 640
source: data/sample.jpg            # image, vidéo, ou dossier de frames
output_dir: outputs/baseline
```

Chaque projet ajoute ses propres champs métier en dessous (ex. `fallen_aspect_ratio`, `keypoint_confidence`), documentés un par un dans le README, comme le fait déjà `louay/fall-detection/README.md`.

`config.py::load_settings(path)` charge ce fichier dans une `@dataclass(frozen=True) Settings` — jamais de `dict` brut passé de fonction en fonction (pattern déjà en place dans les 3 sous-projets de `louay/`).

---

## 3. Contrat de code — le point d'intégration

Pour qu'un projet soit "branchable" dans un système plus large (par exemple appelé depuis l'app multi-caméras de `yassmine/`), `runner.py` doit exposer une fonction **importable**, pas seulement un `main()` en CLI :

```python
# src/<package>/runner.py
def run(settings: Settings, source: Path) -> RunResult:
    """Exécute le pipeline complet et renvoie un résultat structuré.
    Doit pouvoir être appelée depuis un autre programme Python,
    pas seulement depuis la ligne de commande."""

def main() -> None:
    """Wrapper CLI (argparse) autour de run(), pour l'usage en standalone."""
```

`RunResult` (ou équivalent) doit toujours produire :
1. Un **fichier structuré** des détections (CSV ou JSON — pas seulement un affichage `cv2.imshow`). C'est le manque le plus critique identifié chez `chaima/42-crowd-density-estimation` : sans persistance, le projet est inutilisable en dehors d'une démo avec écran.
2. Un **mode headless** obligatoire (aucun `cv2.imshow`/`cv2.waitKey` bloquant par défaut) — indispensable pour tourner sur un serveur ou dans un pipeline automatisé.
3. Optionnellement, un média annoté (image/vidéo) si utile pour la revue humaine.

Pour les projets qui doivent aussi déclencher des alertes en temps réel (surveillance, fraude), réutiliser la brique `Notifier` de `yassmine/mov_vide_nuit/src/notifier.py` (cooldown, logs, preuve horodatée, callback) plutôt que d'en réécrire une variante par projet.

---

## 4. Poids de modèle mutualisés

Un seul dossier `models/` partagé à la racine du dépôt (généralisation du pattern déjà utilisé en interne par `louay/models/`), au lieu d'un `.pt` retéléchargé par projet :

```
Yolo stage/
├── models/                       # poids partagés, non versionnés (gitignored), un seul exemplaire par fichier
│   ├── yolo26n.pt
│   └── yolo26n-pose.pt
├── chaima/...
├── ghada/...
├── louay/...
├── mohamed amine/...
└── yassmine/...
```

`model.py::load_model(settings)` résout `settings.model` contre ce dossier partagé. Migration suggérée : déplacer `louay/models/` vers `Yolo stage/models/` et mettre à jour la résolution de chemin dans les 3 `config.py` de `louay/` (actuellement fragile, voir `louay/PLAN_AMELIORATION.md` §2).

---

## 5. `model_card.yaml` — comparer les projets entre eux

Pour que tu puisses "prendre le bon modèle" sans devoir relire chaque projet en détail, chaque projet fournit une fiche d'identité à sa racine :

```yaml
name: fall-detection
owner: louay
task: classification (debout / tombé)
input_type: image          # image | video | stream
model: yolo26n.pt
status: baseline            # baseline | trained | evaluated | production-ready
metrics: {}                  # rempli dès qu'une évaluation quantitative existe (precision, recall, ...)
depends_on: []               # autres projets/briques réutilisés, le cas échéant
```

Ce fichier est ce qui permet une comparaison rapide entre, par exemple, plusieurs approches de comptage de personnes développées par différents stagiaires, sans ouvrir le code de chacune.

---

## 6. Reproductibilité — non négociable

- `pyproject.toml` avec dépendances **versionnées** (`>=` a minima, idéalement bornées), package installable en mode éditable (`pip install -e .`), point d'entrée `[project.scripts]`.
- `requirements.txt` seul (sans `pyproject.toml`) est toléré uniquement en phase d'exploration très précoce, à migrer dès que la structure `src/` existe.
- Toujours un exemple d'entrée versionné dans `data/` pour que `git clone` + `pip install -e .` + commande du README suffisent à obtenir un résultat, sans dépendre d'un chemin local (`/content/drive/...`) ou d'un service externe non documenté (cf. le notebook Colab de `ghada/Comportement-client/`, inexécutable tel quel hors Colab).
- `.gitignore` type : `.venv/`, `__pycache__/`, `outputs/`, `*.pt`/`*.onnx` (sauf l'éventuel exemple explicitement whitelisté), données brutes volumineuses.

---

## 7. Tests minimums

Pas besoin de mocker YOLO : toute la logique métier (classification par seuil, calcul d'angle, association de boîtes, agrégation de densité) doit être écrite en fonctions pures dans un module séparé du chargement du modèle, et testée sans GPU ni vidéo réelle — c'est déjà le cas dans `louay/*/src/*/classify.py`, `separation.py`, `posture.py`. Un projet sans aucune fonction testable indépendamment du modèle est un signal que la logique métier et l'inférence ne sont pas assez découplées.

---

## 8. Migration des projets existants

| Projet | Écart principal par rapport à ce standard | Détail |
|---|---|---|
| `louay/*` | Quasi conforme — écarts mineurs (résolution de `models_root` fragile, un sous-projet sans `model.py`, pas de `data/sample.mp4`) | `louay/PLAN_AMELIORATION.md` |
| `yassmine/mov_vide_nuit` | Pas de `pyproject.toml`/`src/` layout ; bonne brique `Notifier` à réutiliser telle quelle ailleurs | `yassmine/PLAN_AMELIORATION.md` |
| `chaima/42-crowd-density-estimation` | Script à plat, config en dur, aucune sortie persistée, pas de mode headless | `chaima/PLAN_AMELIORATION.md` |
| `chaima/12-*`, `chaima/47-*` | Aucun code — à démarrer directement sur ce standard | `chaima/PLAN_AMELIORATION.md` |
| `ghada/*` | Notebooks non reproductibles, dépendance Colab en dur, un seul notebook fonctionnel (et non exécutable tel quel) | `ghada/PLAN_AMELIORATION.md` |
| `mohamed amine/` | Rien n'existe — à démarrer directement sur ce standard | `mohamed amine/PLAN_AMELIORATION.md` |

Chaque `PLAN_AMELIORATION.md` de projet détaille les actions concrètes ; ce document-ci fixe la cible commune vers laquelle elles convergent.
