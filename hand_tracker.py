import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
from typing import Tuple, Optional, List

class HandTracker:
    """
    Module 1: Perception Engine
    Handles real-time webcam frame acquisition, 3D hand landmark tracking using 
    the modern MediaPipe Tasks API (on CPU), and noise reduction using an 
    Exponential Moving Average (EMA) filter.
    """
    # Standard hand skeletal connections for drawing overlays
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),         # Index
        (5, 9), (9, 10), (10, 11), (11, 12),     # Middle
        (9, 13), (13, 14), (14, 15), (15, 16),   # Ring
        (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky
        (0, 17)                                 # Palm base
    ]

    def __init__(
        self, 
        alpha: float = 0.15, 
        model_path: str = "hand_landmarker.task",
        min_hand_detection_confidence: float = 0.7,
        min_hand_presence_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7
    ) -> None:
        """
        Initializes the perception engine.

        Args:
            alpha: Exponential Moving Average smoothing factor. Range: (0, 1].
                   Tuned default is 0.15, balancing low-latency with tremor filtering.
            model_path: Path to the downloaded hand_landmarker.task model.
            min_hand_detection_confidence: Detection confidence threshold.
            min_hand_presence_confidence: Presence confidence threshold.
            min_tracking_confidence: Tracking confidence threshold.
        """
        self.alpha = alpha
        
        # Configure MediaPipe Tasks HandLandmarker options
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Initialize detector
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Filter state for Landmark 8 (INDEX_FINGER_TIP) -> np.array([x, y, z])
        self._filtered_pos: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Resets the internal state of the EMA filter."""
        self._filtered_pos = None

    def close(self) -> None:
        """Closes the underlying hand landmarker detector."""
        self.detector.close()

    def draw_landmarks(self, frame: np.ndarray, landmarks) -> None:
        """
        Manually draws hand landmarks and skeletal connections on the frame.

        Args:
            frame: Image frame to draw on.
            landmarks: List of normalized hand landmarks.
        """
        h, w, _ = frame.shape
        # Draw connections
        for connection in self.HAND_CONNECTIONS:
            pt1 = landmarks[connection[0]]
            pt2 = landmarks[connection[1]]
            px1, py1 = int(pt1.x * w), int(pt1.y * h)
            px2, py2 = int(pt2.x * w), int(pt2.y * h)
            cv2.line(frame, (px1, py1), (px2, py2), (180, 105, 255), 2)  # Pink skeleton line

        # Draw joints
        for idx, lm in enumerate(landmarks):
            px, py = int(lm.x * w), int(lm.y * h)
            if idx == 8:
                # Highlight index tip separately
                cv2.circle(frame, (px, py), 6, (0, 255, 255), -1)
            else:
                cv2.circle(frame, (px, py), 4, (255, 0, 0), -1)

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, float]:
        """
        Extracts index finger tip landmark, calculates pinch distance to thumb,
        draws debugging overlays, and filters coordinate noise.

        Args:
            frame: Raw BGR input frame from camera.

        Returns:
            A tuple of:
                - filtered_position: np.ndarray([x, y, z]) or None
                - raw_position: np.ndarray([x, y, z]) or None
                - pinch_distance: float representing normalized Euclidean distance between thumb & index tip.
                - palm_scale: float representing 2D distance between wrist and middle MCP.
                               Returns 1.0 if tracking is lost.
        """
        # Convert BGR frame to RGB and wrap in MediaPipe's Image object
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Run detection
        results = self.detector.detect(mp_image)

        if not results.hand_landmarks:
            self.reset()
            return None, None, 1.0, 0.0

        # Isolate primary hand landmarks
        hand_landmarks = results.hand_landmarks[0]
        
        # Extract Landmark 0 (WRIST), Landmark 9 (MIDDLE_FINGER_MCP), Landmark 4 (THUMB_TIP) and Landmark 8 (INDEX_FINGER_TIP)
        wrist = hand_landmarks[0]
        mcp = hand_landmarks[9]
        thumb = hand_landmarks[4]
        tip = hand_landmarks[8]
        
        # Calculate palm scale in image plane (2D distance between wrist and middle finger knuckle)
        # Closer hand -> larger scale -> maps to larger Z value (lower down in MuJoCo)
        # Further hand -> smaller scale -> maps to smaller Z value (higher up in MuJoCo)
        d_palm = float(np.linalg.norm(np.array([wrist.x - mcp.x, wrist.y - mcp.y])))
        
        # Linearly map palm scale from range [0.09, 0.22] to depth coordinate in range [-0.3, 0.1]
        # Maps 1 meter distance (d_palm = 0.09) to top Z and 30 cm distance (d_palm = 0.22) to floor Z
        d_min, d_max = 0.1, 0.4
        depth_ratio = np.clip((d_palm - d_min) / (d_max - d_min), 0.0, 1.0)
        z_depth_proxy = -0.3 + 0.3 * depth_ratio  # Maps [0, 1] to [-0.3, 0]
        
        thumb_pos = np.array([thumb.x, thumb.y, thumb.z], dtype=np.float64)
        raw_pos = np.array([mcp.x, mcp.y, z_depth_proxy], dtype=np.float64)
        tip_raw = np.array([tip.x, tip.y, tip.z], dtype=np.float64)

        # Calculate Euclidean distance in normalized space
        pinch_distance = float(np.linalg.norm(thumb_pos - tip_raw))

        # Draw hand skeleton overlay
        self.draw_landmarks(frame, hand_landmarks)

        # Draw a custom pinch feedback line between thumb and index tip
        h, w, _ = frame.shape
        px_thumb = (int(thumb.x * w), int(thumb.y * h))
        px_index = (int(tip.x * w), int(tip.y * h))
        
        # Linear color blend: Red (open, ratio=1) -> Green (pinched, ratio=0)
        # Pinch ranges typically from 0.04 (touching) to 0.15+ (open)
        ratio = np.clip((pinch_distance - 0.05) / 0.10, 0.0, 1.0)
        line_color = (0, int(255 * (1.0 - ratio)), int(255 * ratio))
        
        cv2.line(frame, px_thumb, px_index, line_color, 3)
        cv2.circle(frame, px_thumb, 6, line_color, -1)
        cv2.circle(frame, px_index, 6, line_color, -1)

        # Apply Exponential Moving Average (EMA) filter
        if self._filtered_pos is None:
            self._filtered_pos = raw_pos.copy()
        else:
            self._filtered_pos = self.alpha * raw_pos + (1.0 - self.alpha) * self._filtered_pos

        return self._filtered_pos.copy(), raw_pos.copy(), pinch_distance, d_palm


if __name__ == "__main__":
    # Isolated unit test loop
    print("[INFO] Starting Module 1 Unit Test (MediaPipe Tasks API with Pinch)...")
    print("[INFO] Initializing webcam capture (Device Index 0)...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam source. Checking if index 0 is available.")
        print("[WARNING] Exiting test. Please connect a webcam or check camera permissions.")
        exit(1)

    # Instantiate tracker with alpha=0.15
    tracker = HandTracker(alpha=0.15)
    
    prev_time = time.time()
    print("[INFO] Perception loop active. Press 'q' to quit.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from webcam.")
                break

            # Mirror frame horizontally for intuitive human-in-the-loop operation
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Process frame
            filtered, raw, pinch_dist, palm_scale = tracker.process_frame(frame)

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time

            # Render overlays
            cv2.putText(
                frame, f"FPS: {fps:.1f} (Target: 60)", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

            if filtered is not None and raw is not None:
                # Draw tracking debug circles
                # Raw coordinate (Red)
                raw_px = (int(raw[0] * w), int(raw[1] * h))
                cv2.circle(frame, raw_px, 8, (0, 0, 255), -1)
                
                # Filtered coordinate (Green)
                fil_px = (int(filtered[0] * w), int(filtered[1] * h))
                cv2.circle(frame, fil_px, 12, (0, 255, 0), 2)

                # Visual feedback for pinch state
                pinch_status = "CLOSED" if pinch_dist < 0.15 else "OPEN" if pinch_dist > 0.18 else "PROPORTIONAL"
                cv2.putText(
                    frame, f"Pinch Dist: {pinch_dist:.3f} | Gripper: {pinch_status}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                )

                # Print coordinate comparison to terminal
                print(
                    f"Raw: [{raw[0]:.3f}, {raw[1]:.3f}, {raw[2]:.3f}] | "
                    f"EMA-Filtered: [{filtered[0]:.3f}, {filtered[1]:.3f}, {filtered[2]:.3f}] | "
                    f"Pinch: {pinch_dist:.3f}", 
                    end="\r"
                )
            else:
                cv2.putText(
                    frame, "Hand Lost", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )

            # Display window
            cv2.imshow("Module 1 Test - Perception Engine", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print("\n[INFO] Closing Perception Engine unit test.")
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
