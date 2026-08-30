from dataclasses import dataclass
from typing import Optional


@dataclass
class CustomerSession:
    customer_id: int
    entry_time: Optional[float] = None
    exit_time: Optional[float] = None


def point_inside_roi(
    x: float,
    y: float,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> bool:

    return x1 <= x <= x2 and y1 <= y <= y2


def calculate_dwell_time(
    entry_time: float,
    exit_time: float,
) -> float:

    return max(0.0, exit_time - entry_time)