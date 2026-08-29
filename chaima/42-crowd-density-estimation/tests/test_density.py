 
from crowd_density.density import classify_density, point_in_zone
 
 
def test_classify_density_low():
    level, _ = classify_density(count=3, low_threshold=5, medium_threshold=15)
    assert level == "LOW"
 
 
def test_classify_density_exactly_at_low_threshold():
    # cas limite demandé par le plan : count == LOW_THRESHOLD
    level, _ = classify_density(count=5, low_threshold=5, medium_threshold=15)
    assert level == "LOW"
 
 
def test_classify_density_medium():
    level, _ = classify_density(count=10, low_threshold=5, medium_threshold=15)
    assert level == "MEDIUM"
 
 
def test_classify_density_high_just_above_medium():
    # cas limite demandé par le plan : count == MEDIUM_THRESHOLD + 1
    level, _ = classify_density(count=16, low_threshold=5, medium_threshold=15)
    assert level == "HIGH - RISK"
 
 
def test_point_in_zone_none_means_whole_image():
    assert point_in_zone(cx=999, cy=999, zone=None) is True
 
 
def test_point_in_zone_inside():
    assert point_in_zone(cx=50, cy=50, zone=[0, 0, 100, 100]) is True
 
 
def test_point_in_zone_outside():
    assert point_in_zone(cx=150, cy=50, zone=[0, 0, 100, 100]) is False
 
 
def test_point_in_zone_exactly_on_edge():
    # cas limite demandé par le plan : point exactement sur le bord de la zone
    assert point_in_zone(cx=100, cy=100, zone=[0, 0, 100, 100]) is True
 