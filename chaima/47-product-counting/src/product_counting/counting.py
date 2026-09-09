from dataclasses import dataclass, field
 
 
def _side_of_line(point: tuple, line_start: tuple, line_end: tuple) -> float:
    
    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
 
 
def has_crossed_line(prev_point: tuple, curr_point: tuple, line_start: tuple, line_end: tuple) -> bool:
    s1 = _side_of_line(prev_point, line_start, line_end)
    s2 = _side_of_line(curr_point, line_start, line_end)
    return s1 * s2 < 0
 
 
@dataclass
class _TrackState:
    last_point: tuple
    already_counted: bool = False
 
 
@dataclass
class LineCounter:
    
 
    line_start: tuple
    line_end: tuple
    counts: dict = field(default_factory=dict)
    _tracks: dict = field(default_factory=dict)
 
    def update(self, track_id: int, class_name: str, point: tuple) -> bool:
        
        prev = self._tracks.get(track_id)
 
        if prev is not None and not prev.already_counted:
            if has_crossed_line(prev.last_point, point, self.line_start, self.line_end):
                self.counts[class_name] = self.counts.get(class_name, 0) + 1
                self._tracks[track_id] = _TrackState(point, already_counted=True)
                return True
 
        already_counted = prev.already_counted if prev is not None else False
        self._tracks[track_id] = _TrackState(point, already_counted=already_counted)
        return False
 
    def total(self) -> int:
        return sum(self.counts.values())
 