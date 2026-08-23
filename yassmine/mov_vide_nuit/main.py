import cv2
import yaml
import time
import os
from datetime import datetime, time as datetime_time
from src.detector import MotionDetector
from src.classifier import YoloClassifier
from src.notifier import Notifier

def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def is_night_time(start_str, end_str):
    """
    Checks if the current system time falls within the night window.
    Handles scheduling that crosses midnight (e.g. 22:00 to 06:00).
    """
    try:
        start_h, start_m = map(int, start_str.split(':'))
        end_h, end_m = map(int, end_str.split(':'))
        start = datetime_time(start_h, start_m)
        end = datetime_time(end_h, end_m)
        
        now = datetime.now().time()
        
        if start <= end:
            return start <= now <= end
        else: # Crosses midnight
            return now >= start or now <= end
    except Exception as e:
        print(f"Error parsing night schedule times: {e}")
        return True # Default to active safety if config parse fails

def main():
    # Load configuration
    config = load_config()
    
    # 1. Initialize Motion Detector with Noise reduction and ROI Masking
    detector = MotionDetector(
        history=config.get("mog2_history", 500),
        var_threshold=config.get("mog2_var_threshold", 40),
        detect_shadows=config.get("detect_shadows", True),
        min_area=config.get("min_area", 1000),
        blur_size=config.get("blur_size", 5),
        roi_enabled=config.get("roi_enabled", False),
        roi_polygon=config.get("roi_polygon", None),
        clahe_enabled=config.get("clahe_enabled", False),
        clahe_clip_limit=config.get("clahe_clip_limit", 2.0),
        clahe_grid_size=config.get("clahe_grid_size", 8)
    )
    
    # 2. Initialize YOLO Classifier (Secondary Object Detection)
    classifier = None
    if config.get("yolo_enabled", True):
        model_path = config.get("yolo_model", "yolov8n.pt")
        classifier = YoloClassifier(model_path=model_path)
        
    # 3. Initialize Notifier with Discord Webhook and Video Recording integrations
    notifier = Notifier(
        cooldown=config.get("alert_cooldown", 10),
        evidence_dir=config.get("evidence_dir", "results/evidence"),
        log_file=config.get("log_file", "results/surveillance.log"),
        discord_webhook_url=config.get("discord_webhook_url", ""),
        record_video=config.get("record_video", True),
        video_duration=config.get("video_duration", 10),
        pre_alarm_duration=config.get("pre_alarm_duration", 3),
        fps=20.0  # Will be dynamically set below
    )
    
    # Open Video Source
    video_source = config.get("video_path", "data/Walk.mp4")
    
    is_dir = False
    frame_files = []
    frame_index = 0
    fps = 20.0
    
    if isinstance(video_source, str) and os.path.isdir(video_source):
        is_dir = True
        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        frame_files = sorted([
            os.path.join(video_source, f)
            for f in os.listdir(video_source)
            if f.lower().endswith(extensions)
        ])
        if not frame_files:
            print(f"Error: No images found in directory {video_source}")
            return
        fps = 10.0  # Standard FPS for frame sequence playback
        print(f"Surveillance system started on frame directory: {video_source} ({len(frame_files)} frames)")
    else:
        # If it's a digit, treat it as webcam index
        if isinstance(video_source, str) and video_source.isdigit():
            video_source = int(video_source)
            
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return
        
        # Retrieve video file FPS property
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 100:
            fps = 20.0
        print(f"Surveillance system started on: {video_source} (FPS: {fps:.2f})")
        
    # Dynamically scale pre-alarm buffer capacity to match source frame rate
    notifier.set_fps(fps)
    print("Press ESC to exit.")
    
    movement_frames = 0
    required_frames = config.get("required_frames", 5)

    try:
        while True:
            if is_dir:
                if frame_index >= len(frame_files):
                    break
                frame = cv2.imread(frame_files[frame_index])
                frame_index += 1
                ret = frame is not None
            else:
                ret, frame = cap.read()
                
            if not ret:
                # End of video or stream disconnected
                break
                
            # 1. Motion Detection & ROI masking
            resized_frame, motion_mask, motion_boxes, movement_detected = detector.process_frame(frame)
            
            # Keep a copy of the clean frame in the pre-alarm ring buffer
            notifier.keep_frame(resized_frame)
            # 3. Persistence Filter
            if movement_detected:
                movement_frames += 1
            else:
                movement_frames = 0
                
            # 4. Schedule Verification
            is_active_hours = True
            if config.get("night_mode_enabled", True):
                start_t = config.get("night_start_time", "22:00")
                end_t = config.get("night_end_time", "06:00")
                is_active_hours = is_night_time(start_t, end_t)
                
            # 5. Alert Triggering & Drawing
            status_text = "AREA SECURE"
            status_color = (0, 255, 0) # Green
            
            # Start with a clean copy of the frame for display
            display_frame = resized_frame.copy()
            
            if movement_frames >= required_frames:
                status_text = "ALERT: MOVEMENT DETECTED"
                status_color = (0, 0, 255) # Red
                
                # Active Surveillance Mode (within night window)
                if is_active_hours:
                    if classifier:
                        # Run YOLO verification on clean resized_frame for max precision
                        targets, annotated_frame = classifier.classify(
                            resized_frame, 
                            motion_boxes,
                            conf_thresh=config.get("yolo_confidence", 0.5),
                            target_classes=config.get("target_classes", [0])
                        )
                        
                        # Use the YOLO annotated frame as display base
                        display_frame = annotated_frame
                        
                        if targets:
                            status_text = f"ALERT: VERIFIED {targets[0]['label'].upper()}"
                            details = f"Verified {len(targets)} target(s): " + ", ".join([f"{t['label']} ({t['confidence']:.2f})" for t in targets])
                        else:
                            status_text = "ALERT: UNVERIFIED MOTION"
                            status_color = (0, 255, 255) # Yellow/Orange
                            details = "Unverified persistent motion"
                            
                        # Overlay the green motion boxes on top of the YOLO output
                        for (x, y, w, h) in motion_boxes:
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            
                        # Trigger alert on any persistent motion during active hours
                        notifier.trigger_alert(display_frame, details=details)
                    else:
                        # Draw green motion boxes
                        for (x, y, w, h) in motion_boxes:
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # Trigger alert directly without YOLO classification
                        notifier.trigger_alert(display_frame, details="Significant persistent motion")
                else:
                    # Draw green motion boxes
                    for (x, y, w, h) in motion_boxes:
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Daylight Mode: Log passive event, but do not raise alarms or write alerts
                    status_text = "BYPASS: MOTION DETECTED (DAYTIME)"
                    status_color = (255, 255, 0) # Cyan/Light Blue
            else:
                # If there's motion but it hasn't persisted yet, still draw green boxes for visual feedback
                for (x, y, w, h) in motion_boxes:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw Status Header on frame
            cv2.putText(
                display_frame,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2
            )
            
            # Write annotated display frame to video clip if recording is active
            if notifier.is_recording:
                notifier.write_recording_frame(display_frame)
                
            # Show feeds
            cv2.imshow("Surveillance Feed", display_frame)
            cv2.imshow("Surveillance Motion Mask", motion_mask)
            
            # Break loop on ESC key
            if cv2.waitKey(30) & 0xFF == 27:
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Safe release of video writer if still recording on shutdown
        if 'notifier' in locals() and notifier.is_recording:
            if notifier.video_writer:
                notifier.video_writer.release()
                print("Released active video writer during shutdown to save the file.")
        print("Surveillance system stopped.")

if __name__ == "__main__":
    main()
