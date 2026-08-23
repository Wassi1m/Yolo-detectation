from math import degrees

import numpy as np


def angle(first: np.ndarray, vertex: np.ndarray, last: np.ndarray) -> float:
    first_vector = first - vertex
    last_vector = last - vertex
    denominator = np.linalg.norm(first_vector) * np.linalg.norm(last_vector)
    if denominator == 0:
        return float("nan")
    cosine = np.dot(first_vector, last_vector) / denominator
    return degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
