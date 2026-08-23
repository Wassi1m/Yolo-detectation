import cv2
import numpy as np

class MotionDetector:
    """
    Handles motion detection using background subtraction (MOG2),
    CLAHE low-light contrast enhancement, Gaussian blur noise reduction,
    ROI masking, thresholding, morphology operations, and contour analysis.
    """
    def __init__(self, history=500, var_threshold=40, detect_shadows=True, min_area=1000,
                 blur_size=5, roi_enabled=False, roi_polygon=None,
                 clahe_enabled=False, clahe_clip_limit=2.0, clahe_grid_size=8):
        self.min_area = min_area
        self.blur_size = blur_size
        self.roi_enabled = roi_enabled
        self.roi_polygon = roi_polygon
        
        # CLAHE Contrast Enhancer Config
        self.clahe_enabled = clahe_enabled
        self.clahe = None
        if self.clahe_enabled:
            self.clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit, 
                tileGridSize=(clahe_grid_size, clahe_grid_size)
            )
        
        self.background = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        self.kernel = np.ones((5, 5), np.uint8)
        
        # Initialize ROI mask if enabled
        self.roi_mask = None
        if self.roi_enabled and self.roi_polygon:
            # Mask matches target size (1280, 720)
            self.roi_mask = np.zeros((720, 1280), dtype=np.uint8)
            pts = np.array(self.roi_polygon, dtype=np.int32)
            cv2.fillPoly(self.roi_mask, [pts], 255)

    def apply_clahe(self, bgr_image):
        """
        Enhances the local luminance contrast of the BGR image using CLAHE.
        Operates on the L channel of LAB color space to preserve color profiles.
        """
        if not self.clahe:
            return bgr_image
            
        # Convert BGR to LAB color space
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to the Lightness channel
        cl = self.clahe.apply(l)
        
        # Merge and convert back to BGR
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def process_frame(self, frame, target_size=(1280, 720)):
        """
        Resizes frame, applies CLAHE contrast boosting, applies blur noise reduction, 
        filters shadows, restricts detection to ROI if enabled, cleans mask using morphology,
        and finds active contours.
        
        Returns:
            resized_frame: The contrast-enhanced, resized image frame.
            motion_mask: The clean binary mask of motion regions.
            boxes: List of tuples (x, y, w, h) representing motion bounding boxes.
            movement_detected: Boolean indicating if significant movement was detected.
        """
        # Resize frame for uniform processing size
        resized_frame = cv2.resize(frame, target_size)

        # 1. Low-Light Preprocessing: Apply CLAHE to boost contrast in dark regions
        if self.clahe_enabled:
            resized_frame = self.apply_clahe(resized_frame)

        # 2. Noise Reduction: Apply Gaussian Blur to smooth out high-frequency IR sensor noise
        processed_frame = resized_frame
        if self.blur_size > 0:
            # Kernel size must be odd
            k_size = self.blur_size if self.blur_size % 2 == 1 else self.blur_size + 1
            processed_frame = cv2.GaussianBlur(resized_frame, (k_size, k_size), 0)

        # 3. Background Subtraction
        motion_mask = self.background.apply(processed_frame)

        # 4. Remove shadows (MOG2 marks shadows as gray, i.e., value 127)
        _, motion_mask = cv2.threshold(
            motion_mask,
            200,
            255,
            cv2.THRESH_BINARY
        )

        # 5. Apply ROI Masking (Zero out movement outside designated ROI)
        if self.roi_enabled and self.roi_mask is not None:
            motion_mask = cv2.bitwise_and(motion_mask, motion_mask, mask=self.roi_mask)

        # 6. Morphological opening (removes small noise particles)
        motion_mask = cv2.morphologyEx(
            motion_mask,
            cv2.MORPH_OPEN,
            self.kernel
        )

        # 7. Morphological dilation (merges nearby movement parts)
        motion_mask = cv2.dilate(
            motion_mask,
            self.kernel,
            iterations=2
        )

        # 8. Find external contours
        contours, _ = cv2.findContours(
            motion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []
        movement_detected = False

        for contour in contours:
            area = cv2.contourArea(contour)
            # Ignore tiny changes
            if area < self.min_area:
                continue

            movement_detected = True
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))

        # Annotate ROI zone visually on resized_frame if enabled for feedback
        if self.roi_enabled and self.roi_polygon:
            pts = np.array(self.roi_polygon, dtype=np.int32)
            cv2.polylines(resized_frame, [pts], isClosed=True, color=(255, 255, 0), thickness=2)

        return resized_frame, motion_mask, boxes, movement_detected
