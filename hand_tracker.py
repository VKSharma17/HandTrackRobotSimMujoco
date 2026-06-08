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
            num_hands=2,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Initialize detector
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Filter state for Left and Right hands
        self._filtered_pos = {"Left": None, "Right": None}

    def reset(self, hand_label: Optional[str] = None) -> None:
        """Resets the internal state of the EMA filter for a specific hand or all hands."""
        if hand_label:
            self._filtered_pos[hand_label] = None
        else:
            self._filtered_pos = {"Left": None, "Right": None}

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

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Extracts hand landmarks for up to 2 hands, tracks them using independent EMA filters,
        calculates pinch distances and palm scales, and draws skeletal overlays.

        Args:
            frame: Raw BGR input frame from camera.

        Returns:
            A dictionary mapping "Left" and/or "Right" to their respective hand data:
                {
                    "Left": {
                        "filtered_pos": np.array([x, y, z]),
                        "raw_pos": np.array([x, y, z]),
                        "pinch_dist": float,
                        "d_palm": float
                    },
                    "Right": { ... }
                }
        """
        # Convert BGR frame to RGB and wrap in MediaPipe's Image object
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Run detection
        results = self.detector.detect(mp_image)

        hands_data = {}

        if not results.hand_landmarks:
            self.reset()
            return hands_data

        detected_hands = []

        for idx, hand_landmarks in enumerate(results.hand_landmarks):
            # Resolve handedness. Flipped frame swaps Left and Right.
            category = results.handedness[idx][0]
            mp_label = getattr(category, 'category_name', getattr(category, 'label', 'Right'))
            hand_label = "Left" if mp_label == "Right" else "Right"
            detected_hands.append(hand_label)

            # Extract landmarks: 0 (WRIST), 9 (MIDDLE_MCP), 4 (THUMB_TIP), 8 (INDEX_TIP)
            wrist = hand_landmarks[0]
            mcp = hand_landmarks[9]
            thumb = hand_landmarks[4]
            tip = hand_landmarks[8]
            
            # Calculate palm scale in image plane (2D distance between wrist and middle knuckle)
            d_palm = float(np.linalg.norm(np.array([wrist.x - mcp.x, wrist.y - mcp.y])))
            
            # Linearly map palm scale from range [0.1, 0.4] to depth coordinate in range [-0.3, 0]
            d_min, d_max = 0.1, 0.4
            depth_ratio = np.clip((d_palm - d_min) / (d_max - d_min), 0.0, 1.0)
            z_depth_proxy = -0.3 + 0.3 * depth_ratio
            
            thumb_pos = np.array([thumb.x, thumb.y, thumb.z], dtype=np.float64)
            raw_pos = np.array([mcp.x, mcp.y, z_depth_proxy], dtype=np.float64)
            tip_raw = np.array([tip.x, tip.y, tip.z], dtype=np.float64)

            # Calculate Euclidean distance in normalized space
            pinch_distance = float(np.linalg.norm(thumb_pos - tip_raw))

            # Draw hand skeleton overlay
            self.draw_landmarks(frame, hand_landmarks)

            # Draw custom pinch feedback line
            h, w, _ = frame.shape
            px_thumb = (int(thumb.x * w), int(thumb.y * h))
            px_index = (int(tip.x * w), int(tip.y * h))
            
            ratio = np.clip((pinch_distance - 0.05) / 0.10, 0.0, 1.0)
            line_color = (0, int(255 * (1.0 - ratio)), int(255 * ratio))
            
            cv2.line(frame, px_thumb, px_index, line_color, 3)
            cv2.circle(frame, px_thumb, 6, line_color, -1)
            cv2.circle(frame, px_index, 6, line_color, -1)

            # Add hand label text above the knuckle
            px_label = (int(mcp.x * w), int(mcp.y * h) - 15)
            cv2.putText(frame, hand_label, px_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Apply Exponential Moving Average (EMA) filter
            if self._filtered_pos[hand_label] is None:
                self._filtered_pos[hand_label] = raw_pos.copy()
            else:
                self._filtered_pos[hand_label] = self.alpha * raw_pos + (1.0 - self.alpha) * self._filtered_pos[hand_label]

            # Populate hand data
            hands_data[hand_label] = {
                "filtered_pos": self._filtered_pos[hand_label].copy(),
                "raw_pos": raw_pos.copy(),
                "pinch_dist": pinch_distance,
                "d_palm": d_palm
            }

        # Reset any hands that were not detected in this frame
        for label in ["Left", "Right"]:
            if label not in detected_hands:
                self.reset(label)

        return hands_data


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
            hands_data = tracker.process_frame(frame)

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 60.0
            prev_time = curr_time

            # Render overlays
            cv2.putText(
                frame, f"FPS: {fps:.1f} (Target: 60)", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

            if hands_data:
                y_offset = 80
                term_msg = ""
                for hand_label, data in hands_data.items():
                    filtered = data["filtered_pos"]
                    raw = data["raw_pos"]
                    pinch_dist = data["pinch_dist"]
                    
                    # Draw tracking debug circles
                    # Raw coordinate (Red)
                    raw_px = (int(raw[0] * w), int(raw[1] * h))
                    cv2.circle(frame, raw_px, 8, (0, 0, 255), -1)
                    
                    # Filtered coordinate (Green)
                    fil_px = (int(filtered[0] * w), int(filtered[1] * h))
                    cv2.circle(frame, fil_px, 12, (0, 255, 0), 2)

                    pinch_status = "CLOSED" if pinch_dist < 0.15 else "OPEN" if pinch_dist > 0.18 else "PROPORTIONAL"
                    cv2.putText(
                        frame, f"{hand_label} Pinch: {pinch_dist:.3f} | {pinch_status}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                    )
                    y_offset += 30
                    term_msg += f"{hand_label}: [{raw[0]:.2f}, {raw[1]:.2f}, {raw[2]:.2f}] "
                
                print(term_msg, end="\r")
            else:
                cv2.putText(
                    frame, "No Hands Detected", (20, 80), 
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
