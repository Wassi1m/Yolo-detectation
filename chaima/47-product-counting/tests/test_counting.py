from product_counting.counting import LineCounter, has_crossed_line
 
 
def test_has_crossed_line_true_when_moving_across():
    # ligne horizontale à y=300, objet passe de y=250 (au-dessus) à y=350 (en-dessous)
    assert has_crossed_line((100, 250), (100, 350), (0, 300), (640, 300)) is True
 
 
def test_has_crossed_line_false_when_staying_on_same_side():
    assert has_crossed_line((100, 100), (100, 150), (0, 300), (640, 300)) is False
 
 
def test_line_counter_counts_once_on_crossing():
    counter = LineCounter(line_start=(0, 300), line_end=(640, 300))
 
    counter.update(track_id=1, class_name="bottle", point=(100, 250)) 
    counted = counter.update(track_id=1, class_name="bottle", point=(100, 350))  
 
    assert counted is True
    assert counter.counts["bottle"] == 1
 
 
def test_line_counter_does_not_double_count_same_track():
    counter = LineCounter(line_start=(0, 300), line_end=(640, 300))
 
    counter.update(track_id=1, class_name="bottle", point=(100, 250))
    counter.update(track_id=1, class_name="bottle", point=(100, 350))  # compté ici
    counted_again = counter.update(track_id=1, class_name="bottle", point=(100, 400))  # même objet, plus loin
 
    assert counted_again is False
    assert counter.counts["bottle"] == 1  # toujours 1, pas 2
 
 
def test_line_counter_counts_different_tracks_separately():
    counter = LineCounter(line_start=(0, 300), line_end=(640, 300))
 
    counter.update(track_id=1, class_name="bottle", point=(100, 250))
    counter.update(track_id=1, class_name="bottle", point=(100, 350))
    counter.update(track_id=2, class_name="bottle", point=(200, 250))
    counter.update(track_id=2, class_name="bottle", point=(200, 350))
 
    assert counter.counts["bottle"] == 2
    assert counter.total() == 2
 
 
def test_line_counter_separates_classes():
    counter = LineCounter(line_start=(0, 300), line_end=(640, 300))
 
    counter.update(track_id=1, class_name="bottle", point=(100, 250))
    counter.update(track_id=1, class_name="bottle", point=(100, 350))
    counter.update(track_id=2, class_name="box", point=(200, 250))
    counter.update(track_id=2, class_name="box", point=(200, 350))
 
    assert counter.counts == {"bottle": 1, "box": 1}
    assert counter.total() == 2
 