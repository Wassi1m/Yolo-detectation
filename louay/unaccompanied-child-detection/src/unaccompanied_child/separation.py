from typing import Dict, List, Sequence, Set, Tuple

Box = Tuple[float, float, float, float]


def bbox_height(xyxy: Sequence[float]) -> float:
    _, y1, _, y2 = xyxy
    return y2 - y1


def bbox_center(xyxy: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2, (y1 + y2) / 2


def classify_children(heights: Sequence[float], child_height_ratio: float) -> List[bool]:
    if not heights:
        return []
    tallest = max(heights)
    return [h < tallest * child_height_ratio for h in heights]


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class SeparationTracker:

    def __init__(self, unaccompanied_seconds: float, association_distance_ratio: float):
        self.unaccompanied_seconds = unaccompanied_seconds
        self.association_distance_ratio = association_distance_ratio
        self._unaccompanied_since: Dict[int, float] = {}
        self._alerted: Set[int] = set()

    def update(self, timestamp: float, tracks: List[dict]) -> List[dict]:
        children = [t for t in tracks if t["is_child"]]
        adults = [t for t in tracks if not t["is_child"]]
        active_ids = {t["id"] for t in children}

        alerts = []
        for child in children:
            threshold = child["height"] * self.association_distance_ratio
            has_adult_nearby = any(
                _distance(child["center"], adult["center"]) <= threshold for adult in adults
            )
            if has_adult_nearby:
                self._unaccompanied_since.pop(child["id"], None)
                self._alerted.discard(child["id"])
                continue

            since = self._unaccompanied_since.setdefault(child["id"], timestamp)
            duration = timestamp - since
            if duration >= self.unaccompanied_seconds and child["id"] not in self._alerted:
                self._alerted.add(child["id"])
                alerts.append(
                    {
                        "track_id": child["id"],
                        "timestamp": round(timestamp, 2),
                        "unaccompanied_seconds": round(duration, 2),
                    }
                )

        # a track that vanished restarts its timer if it reappears later
        for track_id in list(self._unaccompanied_since):
            if track_id not in active_ids:
                self._unaccompanied_since.pop(track_id, None)
                self._alerted.discard(track_id)

        return alerts
