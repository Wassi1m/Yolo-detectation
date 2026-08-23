import os
import time
import cv2
import logging
import requests
from collections import deque

class Notifier:
    """
    Handles alerts throttling, logging, saving evidence images,
    maintaining a pre-alarm frame buffer, and writing MP4 video recordings.
    """
    def __init__(self, cooldown=10, evidence_dir="results/evidence", log_file="results/surveillance.log",
                 webhook_url="", record_video=False, video_duration=10, pre_alarm_duration=3, fps=20.0,
                 on_alert_callback=None):
        self.cooldown = cooldown
        self.evidence_dir = evidence_dir
        self.log_file = log_file
        self.webhook_url = webhook_url
        self.on_alert_callback = on_alert_callback
        self.last_alert_time = 0
        
        # Video Recording Config
        self.record_video = record_video
        self.video_duration = video_duration
        self.pre_alarm_duration = pre_alarm_duration
        self.fps = fps
        
        # Recording State
        self.is_recording = False
        self.video_writer = None
        self.frames_remaining = 0
        
        # Pre-Alarm Circular Frame Buffer (clean pre-intrusion frames)
        buffer_len = max(1, int(self.pre_alarm_duration * self.fps))
        self.pre_alarm_buffer = deque(maxlen=buffer_len)
        
        # Ensure directories exist
        os.makedirs(self.evidence_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file) or '.', exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            filename=self.log_file,
            filemode='a',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        
        # Also log to console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(console_handler)

    def set_fps(self, fps):
        """
        Dynamically update the FPS and resize the circular buffer accordingly.
        """
        if fps <= 0:
            return
        self.fps = fps
        buffer_len = max(1, int(self.pre_alarm_duration * self.fps))
        
        # Copy elements to new buffer size
        old_buffer = list(self.pre_alarm_buffer)
        self.pre_alarm_buffer = deque(old_buffer, maxlen=buffer_len)

    def keep_frame(self, frame):
        """
        Stores a clean frame in the pre-alarm ring buffer.
        """
        if self.record_video:
            # Store a copy to avoid modification conflicts
            self.pre_alarm_buffer.append(frame.copy())

    def send_webhook_notification(self, message, file_path):
        """
        Sends an alert message and the evidence screenshot to the configured Webhook URL.
        Runs safely inside try-except to prevent surveillance loop crashes.
        """
        if not self.webhook_url:
            return
            
        try:
            payload = {
                "content": f"🚨 **SURVEILLANCE ALERT** 🚨\n{message}"
            }
            with open(file_path, 'rb') as f:
                files = {
                    "file": (os.path.basename(file_path), f, "image/jpeg")
                }
                response = requests.post(
                    self.webhook_url,
                    data=payload,
                    files=files,
                    timeout=5
                )
                if response.status_code == 204 or response.status_code == 200:
                    logging.info("Successfully sent webhook notification.")
                else:
                    logging.warning(f"Failed to send webhook. Status code: {response.status_code}")
        except Exception as e:
            logging.error(f"Error sending webhook notification: {e}")

    def trigger_alert(self, frame, details="Movement", metadata=None):
        """
        Triggers an alert if cooldown has expired.
        Saves the frame, logs details, sends webhook alerts, and starts video recording if enabled.
        
        Returns:
            alert_sent: Boolean indicating if the alert was successfully fired (not throttled).
        """
        current_time = time.time()
        
        if current_time - self.last_alert_time < self.cooldown:
            return False  # Cooldown active
            
        self.last_alert_time = current_time
        timestamp = int(current_time)
        
        # 1. Save screenshot
        img_path = os.path.join(self.evidence_dir, f"alert_{timestamp}.jpg")
        cv2.imwrite(img_path, frame)
        
        # 2. Log the incident
        message = f"Surveillance Alert: {details}. Evidence saved to {img_path}"
        logging.info(message)
        
        # Invoke web app callback if registered
        if self.on_alert_callback:
            try:
                self.on_alert_callback(message, img_path, metadata)
            except Exception as e:
                logging.error(f"Error invoking alert callback: {e}")
        
        # 3. Send webhook notification (if configured)
        if self.webhook_url:
            self.send_webhook_notification(message, img_path)
            
        # 4. Start Video Recording
        if self.record_video and not self.is_recording:
            h, w = frame.shape[:2]
            vid_path = os.path.join(self.evidence_dir, f"alert_{timestamp}.mp4")
            
            # Use standard mp4v codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(vid_path, fourcc, self.fps, (w, h))
            
            if self.video_writer.isOpened():
                self.is_recording = True
                total_frames = int(self.video_duration * self.fps)
                
                # Write pre-alarm buffered frames first
                written_frames = 0
                while self.pre_alarm_buffer:
                    buf_frame = self.pre_alarm_buffer.popleft()
                    self.video_writer.write(buf_frame)
                    written_frames += 1
                
                # Set remaining frames count
                self.frames_remaining = max(1, total_frames - written_frames)
                logging.info(f"Started video recording. Saved {written_frames} pre-alarm frames. Writing {self.frames_remaining} remaining frames to {vid_path}")
            else:
                logging.error(f"Failed to open VideoWriter for {vid_path}")
                self.video_writer = None
        
        return True

    def write_recording_frame(self, frame):
        """
        Writes a frame (usually the annotated display frame) to the active video recording.
        Closes the video file when the duration limit is reached.
        """
        if not self.is_recording or not self.video_writer:
            return
            
        self.video_writer.write(frame)
        self.frames_remaining -= 1
        
        if self.frames_remaining <= 0:
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            logging.info("Video clip recording completed successfully.")
