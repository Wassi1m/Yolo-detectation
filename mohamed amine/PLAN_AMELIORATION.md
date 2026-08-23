# Plan d'amélioration — `mohamed amine`

## État actuel

Le dossier ne contient qu'un `README.md` d'une ligne (`# Yolo-detectation`, hérité du nom du dépôt) et aucun code, configuration, ni sous-projet. C'est le seul dossier de stagiaire du dépôt qui n'a pas encore démarré.

## Ce qu'il faut faire

Il n'y a rien à corriger, mais tout à initialiser. Pour éviter de repartir dans une structure isolée (comme `ghada/`, en notebooks non reproductibles) ou incomplète (comme les sous-projets `chaima/` sans code), ce projet doit démarrer **directement** avec la structure commune définie dans `../ARCHITECTURE_COMMUNE.md` à la racine du dépôt.

### Checklist de démarrage

- [ ] Choisir un nom de projet clair et un objectif précis (type de détection, entrée attendue image/vidéo/flux, sortie attendue).
- [ ] Créer la structure de package standard : `pyproject.toml`, `src/<nom_package>/`, `config/baseline.yaml`, `tests/`, `data/` (avec un exemple versionné), `.gitignore` — voir le gabarit détaillé dans `ARCHITECTURE_COMMUNE.md`.
- [ ] Séparer dès le départ : chargement de config (`config.py`), chargement du modèle (`model.py`), logique métier pure et testable (`<logique>.py`), point d'entrée CLI + fonction réutilisable (`runner.py`).
- [ ] Utiliser le modèle YOLO26 (ou la version en cours d'usage par les autres stagiaires — voir `louay/models/README.md`) pointé depuis un dossier de poids partagé, plutôt que de télécharger une copie locale.
- [ ] Écrire un `README.md` avec setup, run, format de sortie, et la description de chaque champ de config — sur le modèle de `louay/fall-detection/README.md`, qui est la meilleure référence actuelle du dépôt.
- [ ] Ajouter un `model_card.yaml` (voir architecture commune) dès la première baseline fonctionnelle, pour que le projet soit comparable aux autres.
- [ ] Ajouter des tests unitaires sur la logique métier pure dès qu'elle existe (pas besoin d'attendre la fin du projet).

## Priorisation

| Priorité | Action |
|---|---|
| 🔴 Immédiat | Cadrer l'objectif du projet et créer le squelette conforme à `ARCHITECTURE_COMMUNE.md` |
| 🟠 Cette semaine | Premier pipeline de bout en bout (baseline zero-shot comme dans `louay/`) |
| 🟡 Court terme | Tests + README complet + `model_card.yaml` |
