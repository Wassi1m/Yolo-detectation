from ultralytics import YOLO
import cv2

def boxes_overlap(box1, box2):
    """
    Checks if box1 (x, y, w, h) overlaps with box2 (x, y, w, h).
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Check if they do NOT overlap
    if (x1 + w1 < x2) or (x2 + w2 < x1) or (y1 + h1 < y2) or (y2 + h2 < y1):
        return False
    return True

class YoloClassifier:
    """
    Handles secondary object classification on motion regions
    using the YOLO architecture to verify specific targets (like 'person').
    """
    def __init__(self, model_path="yolov8n.pt"):
        # Load the YOLO model
        self.model = YOLO(model_path)
        
    def classify(self, frame, motion_boxes, conf_thresh=0.5, target_classes=[0]):
        """
        Runs YOLO inference on the full frame, then filters detections to check if
        any targets overlap with the regions of active motion.
        
        Returns:
            targets_detected: List of dicts representing matched YOLO targets.
            annotated_frame: Frame with YOLO detection bounding boxes overlayed.
        """
        if not motion_boxes:
            return [], frame.copy()

        # Run YOLO inference
        results = self.model(frame, verbose=False)
        annotated_frame = frame.copy()
        targets_detected = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # Check if class is in target classes and confidence is high enough
                if class_id in target_classes and conf >= conf_thresh:
                    # Get box coordinates [x1, y1, x2, y2]
                    coords = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = coords
                    
                    yolo_box = (x1, y1, x2 - x1, y2 - y1)
                    
                    # Verify that this YOLO detection actually overlaps with an active OpenCV motion box
                    has_overlap = False
                    for m_box in motion_boxes:
                        if boxes_overlap(yolo_box, m_box):
                            has_overlap = True
                            break
                            
                    if not has_overlap:
                        # Ignore stationary targets that are not the source of the active motion
                        continue
                    
                    label = self.model.names[class_id]
                    
                    targets_detected.append({
                        "box": yolo_box,
                        "class_id": class_id,
                        "label": label,
                        "confidence": conf
                    })
                    
                    # Draw verification box (blue/purple) to distinguish from raw motion (green)
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),  # Blue
                        2
                    )
                    
                    cv2.putText(
                        annotated_frame,
                        f"VERIFIED: {label.upper()} ({conf:.2f})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2
                    )
                    
        return targets_detected, annotated_frame
