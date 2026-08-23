# Plan d'amélioration — `yassmine/mov_vide_nuit`

Analyse du projet de surveillance nocturne par détection de mouvement (OpenCV MOG2 + vérification YOLOv8), disponible en deux modes : script autonome (`main.py`) et application web multi-caméras (`app.py` + Flask + SSE).

Architecture actuelle : `src/detector.py` (soustraction de fond + CLAHE + ROI), `src/classifier.py` (vérification YOLO), `src/notifier.py` (alertes, enregistrement vidéo, webhook), `configs/config.yaml`, `templates/index.html`.

---

## 1. Bugs bloquants (à corriger en premier)

### 1.1 `main.py` plante toujours au démarrage
`main.py` instancie `Notifier` avec l'argument `discord_webhook_url=config.get("discord_webhook_url", "")` (lignes 60-69), mais `Notifier.__init__` (`src/notifier.py`) n'accepte qu'un paramètre `webhook_url`. Résultat : `TypeError: __init__() got an unexpected keyword argument 'discord_webhook_url'` dès l'exécution. De plus, `config.yaml` déclare la clé `webhook_url`, pas `discord_webhook_url` — donc même après correction du nom d'argument, le webhook ne serait jamais lu.
**Action** : renommer en `webhook_url=config.get("webhook_url", "")` dans `main.py`.

### 1.2 `main.py` plante à la fermeture si la source est un dossier d'images
Le bloc `finally` appelle `cap.release()` sans condition (ligne 229), mais `cap` n'est jamais défini quand `video_source` est un répertoire d'images (`is_dir=True`, lignes 79-91). `app.py` gère correctement ce cas (`if not is_dir and 'cap' in locals()...`) — il suffit d'aligner `main.py` sur la même garde.

### 1.3 `requirements.txt` incomplet — l'app ne s'installe/démarre pas telle quelle
- `flask` est importé dans `app.py` mais absent du fichier de dépendances.
- `requests` est importé dans `src/notifier.py` (utilisé pour le webhook) mais absent également — donc même `main.py` seul ne peut pas démarrer après un `pip install -r requirements.txt` propre.
- À l'inverse, `pandas` et `matplotlib` figurent dans `requirements.txt` mais ne sont utilisés nulle part dans le code (dépendances mortes, alourdissent l'installation inutilement).
**Action** : ajouter `flask` et `requests`, retirer `pandas`/`matplotlib` (sauf besoin futur explicite).

### 1.4 Condition de course sur le redémarrage multi-caméras (`app.py`)
`restart_loop` est une variable globale **partagée par tous les threads caméra**. Chaque thread la remet à `False` dès qu'il relit la config (`surveillance_loop`, ligne 68), avant que les autres threads aient eu le temps de la consulter dans leur boucle interne (`while running and not restart_loop`). Conséquence : lors d'une mise à jour de config avec plusieurs caméras actives, certaines caméras peuvent ne jamais redémarrer avec les nouveaux paramètres.
**Action** : utiliser un flag de redémarrage par caméra (dict `restart_flags[camera_id]`) au lieu d'une variable globale unique.

---

## 2. Sécurité

### 2.1 Endpoint `/config` non authentifié et non validé (`app.py`)
- N'importe qui sur le réseau (l'app écoute sur `0.0.0.0:5001`) peut réécrire la configuration via un `POST /config` sans authentification, y compris pointer `evidence_dir` ou `log_file` vers un chemin arbitraire, ou modifier `video_path` (RTSP, chemin local, etc.).
- `request.json` n'est pas validé : si le corps est vide ou n'est pas un objet JSON, `current.items()`/`new_config.items()` lève une exception non gérée → 500.
**Action** : ajouter une authentification minimale (clé API, session), valider le schéma reçu (types, clés autorisées), englober l'analyse JSON dans un `try/except` avec réponse 400 propre.

### 2.2 Webhook et futurs secrets stockés en clair dans `config.yaml`
Le fichier de configuration (avec l'URL du webhook Discord) est versionnable et modifiable via l'API `/config` sans contrôle d'accès. Si l'app est exposée, ce champ pourrait être détourné pour exfiltrer les alertes vers un autre serveur.
**Action** : déplacer les secrets vers une variable d'environnement ou un fichier `.env` non versionné, et ne les exposer jamais tels quels via `GET /config`.

---

## 3. Dette technique / duplication

### 3.1 Duplication massive entre `main.py` et `app.py`
`load_config`, `is_night_time`, et toute la logique de détection/alerte (~150 lignes) sont dupliquées presque à l'identique entre les deux fichiers. Toute correction de bug doit être faite deux fois (comme le montre le bug 1.1, présent uniquement dans `main.py`).
**Action** : extraire la logique commune dans `src/pipeline.py` (une fonction `run_detection_cycle(frame, detector, classifier, notifier, config)` réutilisable) et dans `src/config_utils.py` (`load_config`, `save_config`, `is_night_time`).

### 3.2 `tests/code_test1.py` n'est pas un test
Le fichier contient un script procédural complet (ouverture de `walk.mp4`, boucle `cv2.imshow`, `waitKey`) sans aucune fonction, assertion, ni usage de `pytest`/`unittest`. Il s'exécute (et ouvre une fenêtre, bloque) dès la collecte par un lanceur de tests, et référence un chemin relatif `walk.mp4` qui n'existe pas dans `tests/`. C'est en réalité une ancienne version prototype du système, mal rangée.
**Action** : soit le déplacer vers `archive/` ou le supprimer, soit écrire de vrais tests unitaires (voir section 4).

### 3.3 Incohérence des modèles YOLO par défaut
`main.py` utilise par défaut `yolov8n.pt` (modèle générique Ultralytics) tandis que `app.py` utilise par défaut `mod_facile.pt` (modèle custom, absent du dépôt car `*.pt` est dans `.gitignore`). Un nouveau clone du dépôt ne peut pas faire tourner `app.py` sans que quelqu'un fournisse ce fichier hors-git.
**Action** : documenter dans le README où récupérer/entraîner `mod_facile.pt`, ou stocker le modèle via Git LFS / un lien de téléchargement, et harmoniser les deux valeurs par défaut.

### 3.4 Magic numbers dupliqués
La résolution `(1280, 720)` est codée en dur à plusieurs endroits (masque ROI dans `detector.py`, frame vide dans `app.py::generate_frames`, position du timestamp). Un changement de résolution cible nécessite de modifier plusieurs fichiers en cohérence.
**Action** : centraliser dans `config.yaml` (`target_resolution: [1280, 720]`) et propager cette valeur partout.

---

## 4. Tests

Aucun test automatisé exploitable n'existe actuellement (voir 3.2). Compte tenu de la logique déjà bien découpée en classes (`MotionDetector`, `YoloClassifier`, `Notifier`), des tests unitaires sont réalisables sans caméra réelle :
- `MotionDetector.process_frame` : vérifier la détection sur des frames synthétiques (image statique vs image avec un rectangle mobile).
- `Notifier.trigger_alert` : vérifier le respect du cooldown, la création des fichiers de preuve, l'appel du callback (mock `cv2.imwrite`/filesystem via `tmp_path`).
- `is_night_time` : cas limites (minuit, plage qui traverse minuit, horaires invalides).
- `boxes_overlap` (classifier.py) : cas de chevauchement/non-chevauchement.
**Action** : ajouter `pytest` aux dépendances de dev, créer `tests/test_detector.py`, `tests/test_notifier.py`, `tests/test_utils.py`.

---

## 5. Configuration par défaut suspecte

`config.yaml` définit `night_start_time: '2:00'` et `night_end_time: '04:00'`, soit une fenêtre active de seulement 2h (2h-4h du matin). Le commentaire dans le code (`main.py`) donne pourtant l'exemple standard `22:00 à 06:00`. Cette valeur ressemble à un réglage de test oublié plutôt qu'à une valeur de production.
**Action** : vérifier avec l'auteure si c'est intentionnel ; sinon remettre une plage nocturne réaliste par défaut (ex. `22:00`–`06:00`).

---

## 6. Documentation

Le `README.md` du dépôt principal contient uniquement le titre `# Yolo-detectation`. Rien n'explique : comment installer, comment lancer `main.py` vs `app.py`, où placer les vidéos (`data/` est gitignoré, donc absent d'un clone frais), comment obtenir/entraîner `mod_facile.pt`, ni la structure du projet.
**Action** : rédiger un README avec installation (`pip install -r requirements.txt`), lancement des deux modes, structure des dossiers, et schéma de `config.yaml` (chaque champ expliqué).

---

## 7. Performance / robustesse (secondaire)

- **Inférence YOLO sur image complète** à chaque alerte, pour chaque caméra active en parallèle (threads) : sur CPU, ceci limitera le nombre de caméras simultanées supportées. Envisager de restreindre l'inférence aux régions de mouvement (crop) plutôt qu'à la frame entière, ou d'ajouter un paramètre `device` (GPU) configurable.
- **Rechargement complet du modèle YOLO** à chaque redémarrage de config (`start_surveillance_threads`/`restart_loop`) pour chaque caméra : pourrait être mutualisé si plusieurs caméras utilisent le même modèle, pour éviter de le charger N fois en mémoire.

---

## Priorisation suggérée

| Priorité | Action | Effort |
|---|---|---|
| 🔴 Immédiat | Corriger bug `discord_webhook_url` (1.1) | 2 min |
| 🔴 Immédiat | Corriger `cap.release()` non défini (1.2) | 5 min |
| 🔴 Immédiat | Compléter `requirements.txt` (1.3) | 5 min |
| 🟠 Cette semaine | Sécuriser `/config` (2.1) | 1-2 h |
| 🟠 Cette semaine | Corriger la race condition multi-caméras (1.4) | 1 h |
| 🟡 Court terme | Mutualiser code dupliqué main.py/app.py (3.1) | 2-3 h |
| 🟡 Court terme | Nettoyer/remplacer `tests/code_test1.py` par de vrais tests (3.2, 4) | 3-4 h |
| 🟢 Continu | README + doc config (6) | 1-2 h |
