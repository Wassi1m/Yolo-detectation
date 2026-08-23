# Plan d'amélioration — `louay`

Le dossier contient **trois projets** de vision par ordinateur bâtis sur le même gabarit (package Python installable, `src/` layout, `config/baseline.yaml`, point d'entrée console, poids YOLO26 partagés) :

- `fall-detection` : classification debout/tombé par ratio largeur/hauteur de la boîte englobante.
- `posture-load-estimation` : estimation d'un angle de flexion du tronc à partir des points-clés de pose YOLO26-pose.
- `unaccompanied-child-detection` : suivi de personnes (ByteTrack) + heuristique enfant isolé.

C'est de loin le dossier le **mieux structuré** du dépôt (packaging propre, séparation config/modèle/logique métier/point d'entrée, `.gitignore` cohérent). Ce plan porte donc surtout sur la consolidation plutôt que sur des corrections de bugs bloquants.

---

## 1. Écarts entre sous-projets à corriger

### 1.1 `unaccompanied-child-detection` n'a pas de `model.py`
`fall-detection` et `posture-load-estimation` isolent le chargement du modèle dans un module dédié (`model.py::load_model`), mais `unaccompanied_child/runner.py` instancie `YOLO(str(settings.model))` directement en ligne 73, en dehors de toute fonction testable. Ceci casse la symétrie entre les trois projets et empêche de mocker facilement le chargement du modèle dans un test.
**Action** : ajouter `unaccompanied_child/model.py` avec une fonction `load_model(settings)`, identique au pattern des deux autres projets.

### 1.2 `unaccompanied-child-detection` ne peut pas tourner "out of the box"
Contrairement à `fall-detection` et `posture-load-estimation` qui livrent un `data/sample.jpg` versionné, `unaccompanied-child-detection` ne possède **aucun dossier `data/`** alors que son README indique un défaut `data/sample.mp4`. De plus, son `.gitignore` contient `*.mp4`, ce qui empêche de committer un jour un exemple. Résultat : un nouveau clone ne peut pas exécuter la commande `unaccompanied-child` sans fournir manuellement une vidéo, sans qu'aucune documentation n'indique où s'en procurer une.
**Action** : soit fournir un court extrait vidéo de démo (licence libre) et l'exclure de la règle `*.mp4` du `.gitignore` (`!data/sample.mp4`), soit documenter clairement dans le README comment obtenir/générer une vidéo de test.

---

## 2. Fragilité du chargement de configuration (les 3 sous-projets)

Chaque `config.py::load_settings` résout le chemin du modèle et des dossiers de données via une remontée d'arborescence codée en dur :
```python
project_root = path.resolve().parent.parent
models_root = project_root.parent / "models"
```
Ceci suppose que `--config` pointe toujours vers `<projet>/config/baseline.yaml` (deux niveaux exactement) et que `models/` est toujours le dossier frère direct de `louay/`. Si un utilisateur exécute `--config /autre/chemin/config.yaml` (fonctionnalité explicitement supportée par `argparse`, voir README « Pass `--config` to use a different configuration file »), le calcul de `models_root` devient silencieusement faux et pointe vers un dossier inexistant ou erroné, sans message d'erreur clair (juste un chemin de modèle invalide passé à `YOLO(...)`).
**Action** : ne pas dériver `models_root` de la position du fichier de config ; le rendre explicite dans `baseline.yaml` (`models_dir: ../../models`) ou via une variable d'environnement `LOUAY_MODELS_DIR`, avec un message d'erreur explicite si le chemin résolu n'existe pas.

## 3. Absence de tests (assumé dans le README, à planifier)

Le README de `louay/` liste lui-même « Add tests » dans les prochaines étapes — c'est donc un choix conscient, mais aucun n'existe encore alors que la logique métier est déjà bien isolée et facilement testable sans YOLO :
- `fall_detection.classify.classify_posture` : cas limites (`height == 0`, ratio pile égal au seuil).
- `unaccompanied_child.separation.SeparationTracker` : scénario complet (enfant isolé → alerte → adulte s'approche → reset → track disparaît → reset), sans dépendance à YOLO ni vidéo réelle.
- `posture_load.posture.trunk_flexion_angle` et `geometry.angle` : cas où aucun côté n'a une confiance suffisante, vecteurs nuls (division par zéro déjà gérée en retournant `nan`, à vérifier par un test).
**Action** : ajouter `pytest` en dépendance de dev dans chaque `pyproject.toml` (`[project.optional-dependencies] dev = ["pytest"]`) et un dossier `tests/` par sous-projet, en commençant par ces fonctions pures.

## 4. Modèles "zero-shot", pas encore adaptés au cas d'usage

Les trois projets utilisent des heuristiques géométriques simples sur un modèle YOLO26 générique non ré-entraîné (aspect ratio pour la chute, ratio de taille pour "enfant", angle de flexion pour la posture) — c'est assumé et documenté comme un baseline de bout en bout, pas un défaut de code. Points de vigilance pour la suite :
- `fall-detection` : un adulte penché/accroupi (sans être tombé) aura aussi un ratio largeur/hauteur élevé → faux positifs. La config a bien un champ `fallen_aspect_ratio` réglable mais pas encore de jeu d'évaluation pour le calibrer (le README mentionne Le2i/UP-Fall/UR Fall Detection comme sources possibles, mais rien n'est encore intégré).
- `unaccompanied-child-detection` : la classification "enfant" est **relative aux personnes présentes dans la même frame** (`classify_children`) — une frame où seuls des enfants sont visibles ne détectera aucun "enfant" (le plus grand des enfants sera classé adulte par construction). C'est une limite structurelle de l'heuristique à documenter clairement comme risque connu, pas un bug de code.
**Action** : dès qu'un jeu de données labellisé est disponible, ajouter une étape d'évaluation quantitative (precision/recall) avant de considérer un de ces trois baselines comme prêt pour autre chose qu'une démo.

## 5. Documentation — déjà bon niveau, un point à compléter

Les README sont clairs et bien écrits (setup, run, output, configuration détaillée champ par champ). Il manque cependant :
- Un README à la racine de `louay/models/` explique le partage des poids, mais aucun des 3 sous-projets n'explique **où récupérer `data/sample.mp4`** pour `unaccompanied-child-detection` (cf. 1.2).
- Aucune mention de la version de Python testée au-delà de `requires-python = ">=3.9"`, ni d'instructions Linux/macOS (les commandes `Setup`/`Run` utilisent uniquement la syntaxe Windows `.venv/Scripts/...` alors que l'environnement d'exécution observé est Linux — `.venv/bin/...` serait nécessaire).
**Action** : corriger les commandes d'installation pour couvrir Linux/macOS (`source .venv/bin/activate` ou `.venv/bin/pip install -e .`) en plus de Windows.

---

## Priorisation suggérée

| Priorité | Action | Effort |
|---|---|---|
| 🔴 Immédiat | Corriger les commandes Setup/Run pour Linux/macOS (5) | 10 min |
| 🟠 Cette semaine | Fournir `data/sample.mp4` ou documenter son absence (1.2) | 30 min |
| 🟠 Cette semaine | Ajouter `model.py` à `unaccompanied-child-detection` (1.1) | 15 min |
| 🟡 Court terme | Fiabiliser la résolution de `models_root` (2) | 1 h |
| 🟡 Court terme | Ajouter les premiers tests unitaires sur la logique pure (3) | 2-3 h |
| 🟢 Continu | Jeu d'évaluation quantitatif par projet avant mise en production (4) | variable |
