from typing import Optional

import numpy as np

from posture_load.geometry import angle

RIGHT_SIDE = (6, 12, 14)
LEFT_SIDE = (5, 11, 13)


def trunk_flexion_angle(
    keypoints_xy: np.ndarray, keypoints_conf: np.ndarray, min_confidence: float
) -> Optional[float]:
    for shoulder, hip, knee in (RIGHT_SIDE, LEFT_SIDE):
        if min(keypoints_conf[shoulder], keypoints_conf[hip], keypoints_conf[knee]) >= min_confidence:
            return angle(keypoints_xy[shoulder], keypoints_xy[hip], keypoints_xy[knee])
    return None
