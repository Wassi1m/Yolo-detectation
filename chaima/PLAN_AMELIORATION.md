# Plan d'amélioration — `chaima/`

Le dossier `chaima/` contient trois sous-projets de vision par ordinateur, chacun désigné par un numéro de ticket :

| Sous-projet | État actuel |
|---|---|
| `42-crowd-density-estimation` | Code fonctionnel (script unique), en cours |
| `12-inventory-fraud-detection` | **Vide** — aucun contenu, même pas de README rempli |
| `47-product-counting` | **Vide** — aucun contenu, même pas de README rempli |

Le `README.md` à la racine de `chaima/` est également vide (juste le titre générique `# Yolo-detectation` hérité du dépôt).

---

## 1. `42-crowd-density-estimation` — script `crowd_density.py`

Détecte les personnes avec YOLO26 (Ultralytics), compte celles dont le centre tombe dans une zone rectangulaire, et classe la densité en LOW/MEDIUM/HIGH.

### 1.1 🔴 Zone et seuils codés en dur, non exploitables sans modifier le code
`ZONE`, `LOW_THRESHOLD` et `MEDIUM_THRESHOLD` sont des constantes de module (lignes 18-24), pas des arguments CLI ni des champs de config. Le README affirme pourtant : *"the same code used for local testing will connect to the real surveillance camera by simply changing the `--source` argument — no logic changes needed"* — c'est faux dès qu'on change de caméra, car `ZONE` (les coordonnées pixels de la zone à surveiller) dépend entièrement du cadrage de la caméra et doit être réédité dans le fichier source à chaque déploiement.
**Action** : sortir `zone`, `low_threshold`, `medium_threshold` vers un fichier `config/baseline.yaml` (voir architecture commune) ou au minimum des arguments `argparse` (`--zone x1,y1,x2,y2`, `--low`, `--medium`).

### 1.2 🔴 Aucune persistance des résultats — inutilisable en surveillance réelle sans surveillant humain
Le script n'écrit ni logs, ni CSV, ni image de preuve : la seule sortie est l'affichage `cv2.imshow` (lignes 85-88). Pour un cas d'usage de détection de risque de bousculade, si personne ne regarde l'écran au moment d'un pic HIGH, l'événement est perdu — rien n'est enregistré ni notifié.
**Action** : au minimum écrire un CSV horodaté (`timestamp, count, level`) à chaque frame ou changement de niveau, et sauvegarder un instantané image lors d'un passage en HIGH (cf. `Notifier` de `yassmine/mov_vide_nuit/src/notifier.py`, qui fait déjà ce travail et pourrait être réutilisé/adapté).

### 1.3 🟠 `cv2.imshow` empêche tout déploiement headless (serveur / RTSP en production)
Le README cible explicitement un flux RTSP de caméra de surveillance réelle (`rtsp://<camera-ip>:554/stream1`), un scénario typiquement exécuté sur un serveur sans écran. Or le script appelle `cv2.imshow`/`cv2.waitKey` sans condition (lignes 85, 87) : sur une machine headless, ceci lève une erreur OpenCV (absence de backend GUI) et empêche le script de tourner.
**Action** : ajouter un flag `--headless` qui désactive l'affichage et ne fait qu'écrire les résultats (logs/CSV/évidence), à l'image du choix déjà fait dans `yassmine/mov_vide_nuit/app.py` (streaming HTTP au lieu de fenêtre locale).

### 1.4 🟠 Pas de reconnexion sur perte de flux RTSP
La boucle principale (lignes 53-57) s'arrête définitivement (`break`) dès que `cap.read()` échoue une fois. Pour un fichier vidéo local c'est correct (fin de vidéo), mais pour un flux RTSP réel visé par ce projet, une coupure réseau transitoire est fréquente et ferait s'arrêter tout le monitoring sans réessai. `yassmine/mov_vide_nuit/app.py` gère déjà ce cas (reconnexion avec délai) — s'en inspirer.
**Action** : différencier "fin de fichier vidéo" (arrêt normal) de "flux réseau/caméra" (tenter une reconnexion avec backoff), par exemple selon le type de `source` comme le fait déjà `yassmine`.

### 1.5 🟡 Modèle téléchargé/chargé individuellement, pas de dossier partagé
`model = YOLO("yolo26n.pt")` (ligne 46) télécharge son propre exemplaire du modèle dans le dossier courant. Si plusieurs projets de vision (comme ceux de `louay/`) tournent sur la même machine, chacun retélécharge et stocke sa propre copie du même poids `yolo26n.pt`.
**Action** : adopter le même principe que `louay/models/` — un dossier de poids partagé référencé par chemin relatif depuis la config (voir section architecture commune ci-dessous).

### 1.6 🟡 `requirements.txt` minimal, pas de version figée pour la reproductibilité
Seul `ultralytics>=8.4.0` a une borne ; `opencv-python` n'a aucune contrainte de version, ce qui peut casser silencieusement le script lors d'une mise à jour majeure d'OpenCV (comme observé avec `opencv-python==5.0.0.93` dans `yassmine`, une version très récente aux API parfois changeantes).
**Action** : au moins fixer une borne majeure (`opencv-python>=4.9,<5`) et ajouter `numpy` explicitement (dépendance transitive actuellement implicite).

### 1.7 🟡 Aucun test
Les fonctions pures `point_in_zone` et `classify_density` (lignes 27-40) sont facilement testables unitairement (pas besoin de caméra ni de modèle) mais aucun test n'existe.
**Action** : ajouter `tests/test_crowd_density.py` couvrant les cas limites (`count == LOW_THRESHOLD`, `count == MEDIUM_THRESHOLD + 1`, point exactement sur le bord de la zone, `zone=None`).

---

## 2. `12-inventory-fraud-detection` — dossier vide

Le `README.md` existe mais est **totalement vide** (0 octet de contenu utile), et aucun autre fichier n'est présent. Il n'y a pas même de cahier des charges à partir duquel démarrer.
**Actions avant tout développement** :
1. Rédiger un README minimal décrivant l'objectif (à préciser avec l'auteure : quel type de fraude en inventaire — vol en rayon, erreur de scan, échange d'étiquettes ?), les entrées attendues (flux caméra, données de caisse ?) et les sorties souhaitées.
2. Une fois le besoin clarifié, démarrer directement avec l'architecture commune ci-dessous plutôt qu'un script unique, pour éviter de reproduire les problèmes 1.1-1.4.

## 3. `47-product-counting` — dossier vide

Même constat que ci-dessus : `README.md` présent mais vide, aucun code.
**Actions avant tout développement** : identique à la section 2 (clarifier le besoin — comptage de produits en rayon ? en sortie de caisse ? détection de rupture de stock ?), puis démarrer avec l'architecture commune.

---

## Priorisation suggérée

| Priorité | Action | Effort |
|---|---|---|
| 🔴 Immédiat | Sortir `ZONE`/seuils du code vers une config (1.1) | 30 min |
| 🔴 Immédiat | Ajouter la persistance des résultats (1.2) | 1-2 h |
| 🟠 Cette semaine | Mode headless pour déploiement serveur (1.3) | 1 h |
| 🟠 Cette semaine | Reconnexion sur perte de flux RTSP (1.4) | 1 h |
| 🟡 Court terme | Dossier de poids partagé (1.5) | 30 min |
| 🟡 Court terme | Fixer les versions dans `requirements.txt` (1.6) | 15 min |
| 🟡 Court terme | Tests unitaires sur les fonctions pures (1.7) | 1 h |
| 🔵 Avant de coder | Clarifier le besoin de `12-inventory-fraud-detection` et `47-product-counting` (sections 2, 3) | à planifier avec l'auteure |
