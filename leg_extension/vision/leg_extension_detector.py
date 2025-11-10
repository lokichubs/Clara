import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Thresholds
EXTENSION_THRESHOLD = 165
FLEXION_THRESHOLD = 120
SMOOTHING_WINDOW = 5

# 🎨 Colors
CLARA_COLOR = (228, 203, 121)  # turquoise when extended
WHITE = (255, 255, 255)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class LegExtensionDetector:
    def __init__(self):
        self.leg_extended = False
        self.reps = 0
        self.knee_angles = deque(maxlen=SMOOTHING_WINDOW)
        self.pose = mp_pose.Pose(min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine, -1, 1)))

    def process_frame(self, frame):
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return frame, self.reps, None

        lm = results.pose_landmarks.landmark
        left_z = lm[mp_pose.PoseLandmark.LEFT_ANKLE].z
        right_z = lm[mp_pose.PoseLandmark.RIGHT_ANKLE].z
        side = "left" if left_z < right_z else "right"

        def coord(part):
            p = lm[getattr(mp_pose.PoseLandmark, f"{side.upper()}_{part}")]
            return np.array([p.x * w, p.y * h])

        hip, knee, ankle = coord("HIP"), coord("KNEE"), coord("ANKLE")

        angle = self.calculate_angle(hip, knee, ankle)
        self.knee_angles.append(angle)
        smooth_angle = np.mean(self.knee_angles)

        fully_extended = smooth_angle > EXTENSION_THRESHOLD
        fully_bent = smooth_angle < FLEXION_THRESHOLD

        # State machine for rep counting
        if fully_extended:
            self.leg_extended = True
        elif fully_bent and self.leg_extended:
            self.reps += 1
            self.leg_extended = False

        # Change color based on state
        color = CLARA_COLOR if fully_extended else WHITE

        # Draw lines
        cv2.line(frame, tuple(hip.astype(int)), tuple(knee.astype(int)), color, 4)
        cv2.line(frame, tuple(knee.astype(int)), tuple(ankle.astype(int)), color, 4)

        # Draw joints
        cv2.circle(frame, tuple(hip.astype(int)), 10, color, -1)
        cv2.circle(frame, tuple(knee.astype(int)), 10, color, -1)
        cv2.circle(frame, tuple(ankle.astype(int)), 10, color, -1)

        # Display reps
        cv2.putText(frame, f"Reps: {self.reps}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return frame, self.reps, smooth_angle


if __name__ == "__main__":
    print("Starting Leg Extension Detector test... Press 'q' to quit.")
    detector = LegExtensionDetector()

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Leg Extension Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Leg Extension Test", 980, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        frame, reps, angle = detector.process_frame(frame)

        # Display current knee angle (if detected)
        if angle is not None:
            color = CLARA_COLOR if angle > EXTENSION_THRESHOLD else WHITE
            # This is very distracting and keeps changing, so commented out
            # cv2.putText(frame, f"Angle: {angle:.1f}", (30, 150),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Leg Extension Test", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting test.")
            break

    cap.release()
    cv2.destroyAllWindows()
