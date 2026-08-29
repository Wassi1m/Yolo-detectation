from pathlib import Path
 
from ultralytics import YOLO
 
from .config import Settings
 

SHARED_MODELS_DIR = Path(__file__).resolve().parents[4] / "models"
 
 
def load_model(settings: Settings) -> YOLO:

    weights_path = SHARED_MODELS_DIR / settings.model
    SHARED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
 

    if not weights_path.exists():
        import os
 
        cwd = os.getcwd()
        os.chdir(SHARED_MODELS_DIR)
        try:
            model = YOLO(settings.model)
        finally:
            os.chdir(cwd)
    else:
        model = YOLO(str(weights_path))
 
    return model
 