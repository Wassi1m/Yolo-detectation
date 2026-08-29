from typing import Optional
 
 
def point_in_zone(cx: float, cy: float, zone: Optional[list]) -> bool:
    """Vérifie si un point (cx, cy) est à l'intérieur du rectangle zone [x1, y1, x2, y2].
 
    zone=None signifie "toute l'image" : toujours True.
    """
    if zone is None:
        return True
    x1, y1, x2, y2 = zone
    return x1 <= cx <= x2 and y1 <= cy <= y2
 
 
def classify_density(count: int, low_threshold: int, medium_threshold: int) -> tuple[str, tuple]:
    """Classe un nombre de personnes en niveau de densité + couleur BGR pour l'affichage."""
    if count <= low_threshold:
        return "LOW", (0, 200, 0)
    elif count <= medium_threshold:
        return "MEDIUM", (0, 165, 255)
    else:
        return "HIGH - RISK", (0, 0, 255)
