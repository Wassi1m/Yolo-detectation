# Plan d'amélioration — `ghada/`

Analyse des 3 sous-projets de vision par ordinateur : `Comportement-client/` (comportement client / files d'attente en magasin), `detection-fraude-stock/` (détection de fraude sur stock), `detection-de-pollution/` (détection de pollution). Le README racine ne contient que le titre `# Yolo-detectation` — aucune description ni instruction.

État constaté : **aucun fichier `.py`, aucun `requirements.txt`, aucun README de sous-projet n'existe dans ce dossier.** Tout le travail est contenu dans 4 notebooks Jupyter (`.ipynb`), dont 3 sont **vides** (une seule cellule vierge) et 1 seul contient du code réel mais n'a jamais terminé son exécution avec succès.

---

## 1. État réel du contenu (à corriger avant tout le reste)

### 1.1 Trois notebooks sur quatre sont vides
`Comportement-client/Test.ipynb`, `detection-de-pollution/Test1.ipynb` et `detection-fraude-stock/Test2.ipynb` ne contiennent qu'une cellule vide (`initial_id`), sans code ni sortie. Concrètement, **2 des 3 sous-projets annoncés (fraude sur stock, pollution) n'ont aucune implémentation**, et le sous-projet "comportement client" n'a qu'une ébauche non aboutie dans un second notebook.
**Action** : soit démarrer réellement ces deux sous-projets, soit ne pas les lister comme des livrables tant qu'aucun code n'existe, pour éviter de donner l'impression qu'un travail a été fait.

### 1.2 Le seul notebook fonctionnel n'a jamais tourné jusqu'au bout
`Comportement-client/Analyzing_Retail_Queues_Using_Computer_Vision.ipynb` échoue dès la 2ᵉ cellule : `pip install inference supervision` se termine par `ERROR: No matching distribution found for inference`, puis `import inference` lève un `ModuleNotFoundError`. Toutes les cellules suivantes (chargement du modèle Roboflow, boucle de traitement vidéo complète, écriture du CSV) n'ont donc **jamais été exécutées avec succès** — les sorties visibles dans le fichier sont des tracebacks d'erreur, pas des résultats. Aucune preuve que la logique de comptage de temps en file d'attente fonctionne réellement.
**Action** : corriger l'installation (le paquet `inference` de Roboflow nécessite Python < 3.13 selon les contraintes visibles dans les logs de pip — vérifier la compatibilité avec l'environnement utilisé), puis ré-exécuter le notebook de bout en bout et ne conserver que des sorties valides.

---

## 2. Reproductibilité — le notebook fonctionnel est câblé pour Google Colab, pas pour un poste local

- `from google.colab import drive; drive.mount('/content/drive')` : le notebook ne peut s'exécuter que dans Google Colab.
- Le chemin de la vidéo source est codé en dur : `video_path = "/content/drive/MyDrive/Retail.mp4"`, tout comme le chemin de sortie `"/content/drive/MyDrive/output_video.mp4"`. Ces chemins n'existent que sur le Google Drive personnel de l'auteure — personne d'autre ne peut relancer ce notebook tel quel.
- La vidéo source `Retail.mp4` elle-même n'est pas fournie dans le dépôt (normal pour une vidéo volumineuse, mais rien n'indique où se la procurer).
- Le modèle utilisé (`yolov8n-640` via l'API `get_roboflow_model` du SDK `inference` de Roboflow) nécessite une clé API Roboflow (variable d'environnement `ROBOFLOW_API_KEY` généralement) qui n'est configurée nulle part dans le notebook — probablement une source d'échec supplémentaire une fois le paquet installé.
**Action** : remplacer le montage Google Drive par un chemin relatif configurable (variable d'environnement ou argument), documenter la clé API Roboflow nécessaire (sans jamais la committer en clair), et fournir soit un extrait de vidéo d'exemple soit un lien de téléchargement.

---

## 3. Qualité du code (dans le notebook fonctionnel)

### 3.1 Aucune fonction, tout est dans une seule cellule procédurale
La cellule 4 fait ~90 lignes : ouverture vidéo, tracking, détection des ROI, calcul de temps passé en file, écriture CSV — tout est en séquence dans une seule cellule, sans découpage en fonctions. Impossible de réutiliser une seule partie (par ex. le calcul du temps en file) ailleurs, ni de la tester isolément.
**Action** : extraire au minimum une fonction `is_in_roi(point, roi_coords)`, une fonction `update_queue_tracking(...)`, une fonction `process_video(...)`.

### 3.2 ROI codées en dur, spécifiques à une vidéo précise
`roi1_coords` et `roi2_coords` sont des coordonnées de pixels en dur (`[747, 622], [707, 38], ...`), valables uniquement pour le cadrage exact de `Retail.mp4`. Toute autre vidéo ou caméra nécessiterait de recalculer ces polygones à la main dans le code.
**Action** : externaliser les ROI dans un fichier de configuration (YAML/JSON) par vidéo/caméra, comme fait dans `louay/*/config/baseline.yaml` (`roi_polygon` dans `yassmine/mov_vide_nuit/configs/config.yaml` est un bon précédent dans ce dépôt).

### 3.3 `print()` de debug laissés dans le code, pas de logging
De nombreux `print("Found: ", ...)`, `print("Tracking:", ...)`, `print("outside:", ...)` polluent la sortie et ne peuvent pas être désactivés sans éditer le code.
**Action** : remplacer par le module `logging` standard avec un niveau configurable (voir `logger.info(...)` utilisé dans `louay/*/src/*/runner.py`).

### 3.4 Bug potentiel de division par zéro
`average = sum(timespent)/len(timespent)` (fin de cellule 4) plantera avec `ZeroDivisionError` si aucune personne n'a jamais quitté la file pendant toute la vidéo (`timespent` resterait vide). Cas plausible sur une vidéo courte ou sans mouvement de file.
**Action** : garder ce calcul dans un `if timespent:` ou fournir une valeur par défaut.

### 3.5 Fichier CSV ouvert manuellement sans `with`
`file = open(filename, 'w', newline='')` est ouvert au début et fermé uniquement à la fin via `file.close()` — si une exception survient en cours de boucle (par ex. le `ZeroDivisionError` ci-dessus), le fichier ne sera jamais proprement fermé/flush.
**Action** : utiliser un context manager `with open(filename, 'w', newline='') as file:` englobant toute la boucle.

---

## 4. Absence totale d'environnement reproductible

Aucun `requirements.txt`, `pyproject.toml` ni fichier d'environnement Colab (`!pip install ...` en première cellule seulement, sans version épinglée) n'existe pour aucun des 3 sous-projets. Impossible de savoir quelles versions de `opencv-python`, `pandas`, `numpy`, `supervision`, `inference` ont réellement été utilisées ni de recréer l'environnement de façon fiable ailleurs que dans la session Colab d'origine.
**Action** : dès qu'un notebook produit un résultat qui fonctionne, figer les dépendances dans un `requirements.txt` versionné à la racine de chaque sous-dossier.

---

## 5. Notebooks vs scripts — recommandation structurelle

Les notebooks Jupyter posent ici les mêmes problèmes classiques :
- **Diffs Git illisibles** : chaque exécution change les métadonnées de cellules et les sorties (traces d'erreur incluses, ce qui a permis cette analyse mais pollue l'historique), rendant la revue de code difficile.
- **Aucune fonction testable** : tout le comportement est écrit comme un script linéaire dans des cellules, impossible à appeler depuis un test unitaire ou un autre module.
- **Pas de point d'entrée standard** : rien d'équivalent au `[project.scripts]` + `runner.py` utilisé dans `louay/*`.

**Action recommandée** : une fois qu'un notebook produit un résultat validé, migrer sa logique vers un package Python installable (voir l'architecture commune proposée à la racine du dépôt, `ARCHITECTURE_COMMUNE.md`) — le notebook peut rester comme document d'exploration initiale, mais ne doit pas être le seul artefact livré.

---

## Priorisation suggérée

| Priorité | Action | Effort |
|---|---|---|
| 🔴 Immédiat | Faire tourner `Analyzing_Retail_Queues...ipynb` jusqu'au bout (corriger l'installation de `inference`) | 30-60 min |
| 🔴 Immédiat | Remplacer les chemins Google Drive en dur par des chemins configurables | 15 min |
| 🟠 Cette semaine | Corriger la division par zéro et l'ouverture de fichier sans `with` (3.4, 3.5) | 15 min |
| 🟠 Cette semaine | Créer un `requirements.txt` figé pour le sous-projet fonctionnel | 15 min |
| 🟡 Court terme | Démarrer réellement `detection-fraude-stock` et `detection-de-pollution` (actuellement vides) | à définir |
| 🟡 Court terme | Extraire le code en fonctions + migrer vers un package Python (voir architecture commune) | 3-4 h |
| 🟢 Continu | Documenter chaque sous-projet (README, provenance des données, clé API Roboflow) | 1 h |
