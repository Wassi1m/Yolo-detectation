import cv2
import yaml
import time
import os
import threading
import queue
from datetime import datetime, time as datetime_time
from flask import Flask, render_template, Response, request, jsonify, send_from_directory, redirect, url_for

from src.detector import MotionDetector
from src.classifier import YoloClassifier
from src.notifier import Notifier

app = Flask(__name__)

# Threading and Queues for Multi-Camera Support
camera_frames = {}   # {camera_id: latest_frame}
camera_locks = {}    # {camera_id: threading.Lock()}
active_threads = {}  # {camera_id: threading.Thread()}
alert_queue = queue.Queue()
running = True
restart_loop = False

def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def save_config(config, config_path="configs/config.yaml"):
    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file, default_flow_style=False)

def is_night_time(start_str, end_str):
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
        return True

def handle_local_alert(message, img_path, metadata=None):
    # Parse human-readable details
    filename = os.path.basename(img_path)
    alert_data = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "message": message,
        "image": filename
    }
    if metadata:
        alert_data.update(metadata)
    alert_queue.put(alert_data)

def surveillance_loop(camera_config):
    global camera_frames, camera_locks, restart_loop, running
    camera_id = camera_config["id"]
    camera_name = camera_config.get("name", camera_id)
    
    while running:
        config = load_config()
        restart_loop = False
        
        # Check if camera still exists in config
        this_cam_config = None
        for cam in config.get("cameras", []):
            if cam["id"] == camera_id:
                this_cam_config = cam
                break
        if not this_cam_config:
            print(f"Camera {camera_name} not found in updated config. Stopping thread.")
            break
            
        video_source = this_cam_config["video_path"]
        
        # Initialize Motion Detector
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
        
        # Initialize YOLO Classifier
        classifier = None
        if config.get("yolo_enabled", True):
            model_path = config.get("yolo_model", "mod_facile.pt")
            classifier = YoloClassifier(model_path=model_path)
            
        # Initialize Notifier
        notifier = Notifier(
            cooldown=config.get("alert_cooldown", 10),
            evidence_dir=config.get("evidence_dir", "results/evidence"),
            log_file=config.get("log_file", "results/surveillance.log"),
            record_video=config.get("record_video", True),
            video_duration=config.get("video_duration", 10),
            pre_alarm_duration=config.get("pre_alarm_duration", 3),
            fps=20.0,
            on_alert_callback=handle_local_alert
        )
        
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
            fps = 10.0
        else:
            # Connection helper
            def connect_camera(source):
                if isinstance(source, str) and source.isdigit():
                    source = int(source)
                return cv2.VideoCapture(source)
                
            cap = connect_camera(video_source)
            if not cap.isOpened():
                print(f"Error: Could not open video source {video_source} for camera {camera_name}. Retrying in 5s...")
                time.sleep(5)
                continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or fps > 100:
                fps = 20.0
                
        notifier.set_fps(fps)
        movement_frames = 0
        required_frames = config.get("required_frames", 5)
        
        # Frame timing for realistic playback
        frame_delay = 1.0 / fps
        
        while running and not restart_loop:
            start_time = time.time()
            
            if is_dir:
                if frame_index >= len(frame_files):
                    frame_index = 0  # Loop directory
                frame = cv2.imread(frame_files[frame_index])
                frame_index += 1
                ret = frame is not None
            else:
                ret, frame = cap.read()
                if not ret:
                    # Loop video files for continuous web demo, reconnect for RTSP
                    if isinstance(video_source, str) and not video_source.isdigit() and not video_source.startswith("rtsp"):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret:
                            time.sleep(1)
                            continue
                    else:
                        print(f"Connection lost for camera {camera_name}. Reconnecting in 5s...")
                        cap.release()
                        time.sleep(5)
                        cap = connect_camera(video_source)
                        continue
                        
            if not ret:
                time.sleep(0.1)
                continue
                
            # Process frame
            resized_frame, motion_mask, motion_boxes, movement_detected = detector.process_frame(frame)
            notifier.keep_frame(resized_frame)
            
            if movement_detected:
                movement_frames += 1
            else:
                movement_frames = 0
                
            is_active_hours = True
            if config.get("night_mode_enabled", True):
                start_t = config.get("night_start_time", "22:00")
                end_t = config.get("night_end_time", "06:00")
                is_active_hours = is_night_time(start_t, end_t)
                
            status_text = "AREA SECURE"
            status_color = (0, 255, 0)
            display_frame = resized_frame.copy()
            
            # Draw Burn-in Timestamp for evidence/monitoring readability
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(display_frame, f"{camera_name} - {timestamp_str}", (11, 706), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(display_frame, f"{camera_name} - {timestamp_str}", (10, 705), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if movement_frames >= required_frames:
                status_text = "ALERT: MOVEMENT DETECTED"
                status_color = (0, 0, 255)
                
                if is_active_hours:
                    if classifier:
                        targets, annotated_frame = classifier.classify(
                            resized_frame,
                            motion_boxes,
                            conf_thresh=config.get("yolo_confidence", 0.5),
                            target_classes=config.get("target_classes", [0])
                        )
                        # Re-draw the timestamp on the annotated frame
                        display_frame = annotated_frame
                        cv2.putText(display_frame, f"{camera_name} - {timestamp_str}", (11, 706), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cv2.putText(display_frame, f"{camera_name} - {timestamp_str}", (10, 705), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        if targets:
                            status_text = f"ALERT: VERIFIED {targets[0]['label'].upper()}"
                            details = f"[{camera_name}] Verified {len(targets)} target(s): " + ", ".join([f"{t['label']} ({t['confidence']:.2f})" for t in targets])
                            status_color = (0, 0, 255)
                        else:
                            status_text = "ALERT: UNVERIFIED MOTION"
                            status_color = (0, 255, 255)
                            details = f"[{camera_name}] Unverified persistent motion"
                            
                        for (x, y, w, h) in motion_boxes:
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        notifier.trigger_alert(
                            display_frame, 
                            details=details,
                            metadata={
                                "camera_id": camera_name,
                                "alert_type": "Verified Target" if targets else "Unverified Motion",
                                "yolo_conf": float(targets[0]['confidence']) if targets else 0.0,
                                "opencv_area": int(max([b[2] * b[3] for b in motion_boxes])) if motion_boxes else 0
                            }
                        )
                    else:
                        for (x, y, w, h) in motion_boxes:
                            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        notifier.trigger_alert(
                            display_frame, 
                            details=f"[{camera_name}] Significant persistent motion",
                            metadata={
                                "camera_id": camera_name,
                                "alert_type": "Motion Detected",
                                "yolo_conf": 0.0,
                                "opencv_area": int(max([b[2] * b[3] for b in motion_boxes])) if motion_boxes else 0
                            }
                        )
                else:
                    for (x, y, w, h) in motion_boxes:
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    status_text = "BYPASS: MOTION DETECTED (DAYTIME)"
                    status_color = (255, 255, 0)
            else:
                for (x, y, w, h) in motion_boxes:
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
            cv2.putText(
                display_frame,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2
            )
            
            if notifier.is_recording:
                notifier.write_recording_frame(display_frame)
                
            # Update latest frame in dictionary
            with camera_locks[camera_id]:
                camera_frames[camera_id] = display_frame.copy()
                
            elapsed = time.time() - start_time
            sleep_time = max(0.005, frame_delay - elapsed)
            time.sleep(sleep_time)
            
        if not is_dir and 'cap' in locals() and cap is not None:
            cap.release()
            
    if 'notifier' in locals() and notifier.is_recording:
        if notifier.video_writer:
            notifier.video_writer.release()

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames(camera_id):
    global camera_frames, camera_locks
    while True:
        frame = None
        if camera_id in camera_locks:
            with camera_locks[camera_id]:
                if camera_frames.get(camera_id) is not None:
                    frame = camera_frames[camera_id].copy()
                    
        if frame is None:
            # Provide a blank frame if none is available yet
            import numpy as np
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, f"CONNECTING {camera_id.upper()}...", (350, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)  # ~25 FPS streaming

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    return Response(generate_frames(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def legacy_video_feed():
    config = load_config()
    cameras = config.get("cameras", [])
    if cameras:
        return redirect(url_for('video_feed', camera_id=cameras[0]['id']))
    return "No cameras configured"

@app.route('/stream')
def stream_alerts():
    def event_stream():
        # Yield connection status
        yield f"data: {{\"type\": \"system\", \"message\": \"Connected to Surveillance SSE Feed\"}}\n\n"
        while True:
            try:
                alert = alert_queue.get(timeout=1.0)
                # Format to JSON
                import json
                yield f"data: {json.dumps(alert)}\n\n"
            except queue.Empty:
                # Heartbeat to keep connection alive
                yield ": ping\n\n"
    return Response(event_stream(), mimetype="text/event-stream")

def start_surveillance_threads():
    global active_threads, camera_locks, camera_frames, running
    config = load_config()
    running = True
    for camera in config.get("cameras", []):
        camera_id = camera["id"]
        if camera_id not in camera_locks:
            camera_locks[camera_id] = threading.Lock()
            camera_frames[camera_id] = None
        if camera_id not in active_threads or not active_threads[camera_id].is_alive():
            t = threading.Thread(target=surveillance_loop, args=(camera,), daemon=True)
            active_threads[camera_id] = t
            t.start()
            print(f"Started thread for camera: {camera.get('name', camera_id)}")

@app.route('/config', methods=['GET', 'POST'])
def config_api():
    global restart_loop
    if request.method == 'GET':
        return jsonify(load_config())
    else:
        new_config = request.json
        current = load_config()
        # Update keys
        for k, v in new_config.items():
            current[k] = v
        save_config(current)
        restart_loop = True  # Signal threads to reload
        start_surveillance_threads()  # Start threads for any new cameras
        return jsonify({"status": "success", "message": "Config updated and system reloading"})

@app.route('/evidence/<path:filename>')
def serve_evidence(filename):
    evidence_dir = os.path.abspath(load_config().get("evidence_dir", "results/evidence"))
    return send_from_directory(evidence_dir, filename)

if __name__ == '__main__':
    start_surveillance_threads()
    
    # Run Flask
    app.run(host='0.0.0.0', port=5001, debug=False)
