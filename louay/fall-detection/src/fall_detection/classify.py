def classify_posture(width: float, height: float, fallen_aspect_ratio: float) -> str:
    if height <= 0:
        return "unknown"
    return "fallen" if (width / height) >= fallen_aspect_ratio else "standing"
