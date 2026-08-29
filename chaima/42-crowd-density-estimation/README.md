# #42 — Crowd Density Estimation
 
**Status:** baseline (voir model_card.yaml)
 
## Setup
```bash
pip install -e .
```
 
## Run
```bash
# avec l'exemple fourni
crowd-density --config config/baseline.yaml
 
# avec une source différente (fichier, webcam, ou RTSP)
crowd-density --config config/baseline.yaml --source rtsp://<camera-ip>:554/stream1
 
# sur un serveur sans écran
crowd-density --config config/baseline.yaml --headless
```
 
## Sortie produite
- `outputs/baseline/detections.csv` — une ligne par frame : timestamp, nombre de personnes dans la zone, niveau de densité
- `outputs/baseline/high_<timestamp>.jpg` — instantané sauvegardé à chaque transition vers le niveau HIGH
## Configuration (config/baseline.yaml)
| Champ | Description |
|---|---|
| `zone` | `[x1, y1, x2, y2]` en pixels, la zone surveillée. `null` = image entière. À redéfinir selon le cadrage réel de la caméra. |
| `low_threshold` | Nombre de personnes max pour le niveau LOW |
| `medium_threshold` | Nombre de personnes max pour le niveau MEDIUM (au-delà = HIGH) |
| `headless` | `true` désactive l'affichage cv2.imshow (obligatoire en déploiement serveur) |
| `model` | Nom du fichier de poids, résolu contre le dossier `models/` partagé à la racine du repo |
 
## Tests
```bash
pytest tests/
```
 