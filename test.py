import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize Mediapipe pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Constants
EXTENSION_THRESHOLD = 165   # Knee considered extended above this angle
FLEXION_THRESHOLD = 120     # Knee considered bent below this angle
SMOOTHING_WINDOW = 5        # Moving average window for smoothing

# State variables
leg_extended = False  # True if leg was fully extended
reps = 0
knee_angles = deque(maxlen=SMOOTHING_WINDOW)

# Helper function to compute joint angles
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1, 1)
    return np.degrees(np.arccos(cosine_angle))

def draw_leg_line(img, p1, p2, color, thickness=4):
    p1 = tuple(map(int, p1))
    p2 = tuple(map(int, p2))
    cv2.line(img, p1, p2, color, thickness)

# Open webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow('Leg Extension Detector', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Leg Extension Detector', 1280, 720)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Pick the leg closer to the camera
            left_z = lm[mp_pose.PoseLandmark.LEFT_ANKLE].z
            right_z = lm[mp_pose.PoseLandmark.RIGHT_ANKLE].z
            side = "left" if left_z < right_z else "right"

            # Get coordinates
            hip = np.array([
                lm[getattr(mp_pose.PoseLandmark, f"{side.upper()}_HIP")].x * w,
                lm[getattr(mp_pose.PoseLandmark, f"{side.upper()}_HIP")].y * h
            ])
            knee = np.array([
                lm[getattr(mp_pose.PoseLandmark, f"{side.upper()}_KNEE")].x * w,
                lm[getattr(mp_pose.PoseLandmark, f"{side.upper()}_KNEE")].y * h
            ])
            ankle = np.array([
                lm[getattr(mp_pose.PoseLandmark, f"{side.upper()}_ANKLE")].x * w,
                lm[getattr(mp_pose.PoseLandmark, f"{side.upper()}_ANKLE")].y * h
            ])
            heel = np.array([
                lm[getattr(mp_pose.PoseLandmark, f"{side.upper()}_HEEL")].x * w,
                lm[getattr(mp_pose.PoseLandmark, f"{side.upper()}_HEEL")].y * h
            ])
            foot = np.array([
                lm[getattr(mp_pose.PoseLandmark, f"{side.upper()}_FOOT_INDEX")].x * w,
                lm[getattr(mp_pose.PoseLandmark, f"{side.upper()}_FOOT_INDEX")].y * h
            ])

            # Calculate knee angle (for seated extension)
            knee_angle = calculate_angle(hip, knee, ankle)
            knee_angles.append(knee_angle)
            knee_angle_smooth = np.mean(knee_angles)

            # Determine leg state
            fully_extended = knee_angle_smooth > EXTENSION_THRESHOLD
            fully_bent = knee_angle_smooth < FLEXION_THRESHOLD

            # Count reps only when leg goes from fully extended -> fully bent
            if fully_extended:
                leg_extended = True
            elif fully_bent and leg_extended:
                reps += 1
                leg_extended = False

            # Determine color: green if straight, white if bent
            color = (0, 255, 0) if fully_extended else (255, 255, 255)

            # Draw leg lines
            draw_leg_line(image, hip, knee, color)
            draw_leg_line(image, knee, ankle, color)
            draw_leg_line(image, ankle, foot, (255, 255, 255))

            # Draw joints
            for pt in [hip, knee, ankle, heel, foot]:
                cv2.circle(image, tuple(map(int, pt)), 6, (255, 255, 255), -1)

            # Display feedback
            cv2.putText(image, f"Leg: {'STRAIGHT' if fully_extended else 'BENT'}",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.putText(image, f"Reps: {reps}", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Leg Extension Detector', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
