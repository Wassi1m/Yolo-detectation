from fall_detection.classify import classify_posture


def test_zero_height_is_unknown():
    assert classify_posture(width=10, height=0, fallen_aspect_ratio=1.0) == "unknown"


def test_ratio_exactly_at_threshold_is_fallen():
    assert classify_posture(width=10, height=10, fallen_aspect_ratio=1.0) == "fallen"


def test_ratio_just_below_threshold_is_standing():
    assert classify_posture(width=9.9, height=10, fallen_aspect_ratio=1.0) == "standing"


def test_clearly_standing_box():
    assert classify_posture(width=20, height=60, fallen_aspect_ratio=1.0) == "standing"


def test_clearly_fallen_box():
    assert classify_posture(width=60, height=20, fallen_aspect_ratio=1.0) == "fallen"
