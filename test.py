import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time


engine = pyttsx3.init()

all_green_once = False
hold_start_time = None
hold_message_played = False
relax_message_played = False

tol = 15
target_hip = 0
target_knee = 180
target_heel = 90

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Vertex
    c = np.array(c)  # Second point
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1, 1)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

LEG_LANDMARKS = {
    "left": {
        "hip": mp_pose.PoseLandmark.LEFT_HIP.value,
        "knee": mp_pose.PoseLandmark.LEFT_KNEE.value,
        "ankle": mp_pose.PoseLandmark.LEFT_ANKLE.value,
        "heel": mp_pose.PoseLandmark.LEFT_HEEL.value,
        "foot_index": mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value
    },
    "right": {
        "hip": mp_pose.PoseLandmark.RIGHT_HIP.value,
        "knee": mp_pose.PoseLandmark.RIGHT_KNEE.value,
        "ankle": mp_pose.PoseLandmark.RIGHT_ANKLE.value,
        "heel": mp_pose.PoseLandmark.RIGHT_HEEL.value,
        "foot_index": mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value
    }
}


def draw_rect_from_points(img, pt1, pt2, color=(255, 255, 255), thickness=2):
    # Draw rectangle around two points with some width
    pt1 = np.array(pt1).astype(int)
    pt2 = np.array(pt2).astype(int)
    width = 20  # rectangle thickness approx
    # Create perpendicular vector
    vec = pt2 - pt1
    length = np.linalg.norm(vec)
    if length == 0:
        return
    unit_vec = vec / length
    perp_vec = np.array([-unit_vec[1], unit_vec[0]])  # Perpendicular vector

    pts = np.array([
        pt1 + perp_vec * width,
        pt2 + perp_vec * width,
        pt2 - perp_vec * width,
        pt1 - perp_vec * width
    ], np.int32)

    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)


cap = cv2.VideoCapture(0)

cv2.namedWindow('Leg Angles with Rigid Bodies', cv2.WINDOW_NORMAL)

with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        # Convert to RGB for Mediapipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        # Convert back to BGR to display with OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            left_ankle_z = lm[LEG_LANDMARKS["left"]["ankle"]].z
            right_ankle_z = lm[LEG_LANDMARKS["right"]["ankle"]].z
            closest = "left" if left_ankle_z < right_ankle_z else "right"

            # Extract coords scaled to pixel space
            hip = [lm[LEG_LANDMARKS[closest]["hip"]].x * w, lm[LEG_LANDMARKS[closest]["hip"]].y * h]
            knee = [lm[LEG_LANDMARKS[closest]["knee"]].x * w, lm[LEG_LANDMARKS[closest]["knee"]].y * h]
            ankle = [lm[LEG_LANDMARKS[closest]["ankle"]].x * w, lm[LEG_LANDMARKS[closest]["ankle"]].y * h]
            heel = [lm[LEG_LANDMARKS[closest]["heel"]].x * w, lm[LEG_LANDMARKS[closest]["heel"]].y * h]
            foot = [lm[LEG_LANDMARKS[closest]["foot_index"]].x * w, lm[LEG_LANDMARKS[closest]["foot_index"]].y * h]

            hip_angle = calculate_angle(ankle, hip, knee)
            knee_angle = calculate_angle(hip, knee, ankle)
            heel_angle = calculate_angle(ankle, heel, foot)

            tol = 15

            # Define target angles
            target_hip = 0
            target_knee = 180
            target_heel = 90

            # Check if angles are within tolerance
            thigh_color = (0, 255, 0) if abs(hip_angle - target_hip) <= tol else (255, 255, 255)
            calf_color = (0, 255, 0) if abs(knee_angle - target_knee) <= tol else (255, 255, 255)
            foot_color = (0, 255, 0) if abs(heel_angle - target_heel) <= tol else (255, 255, 255)

            # Draw rectangles with condition colors
            draw_rect_from_points(image, hip, knee, color=thigh_color, thickness=2)
            draw_rect_from_points(image, knee, ankle, color=calf_color, thickness=2)
            # draw_rect_from_points(image, ankle, foot, color=foot_color, thickness=2)

            # Draw joint circles
            joint_color = (255, 255, 255)  # white
            radius = 6
            cv2.circle(image, tuple(map(int, hip)), radius, joint_color, -1)
            cv2.circle(image, tuple(map(int, knee)), radius, joint_color, -1)
            cv2.circle(image, tuple(map(int, ankle)), radius, joint_color, -1)
            cv2.circle(image, tuple(map(int, heel)), radius, joint_color, -1)
            cv2.circle(image, tuple(map(int, foot)), radius, joint_color, -1)

            # Draw rectangles approx thigh, calf, foot rigid bodies
            # Thigh rectangle (hip to knee)
            draw_rect_from_points(image, hip, knee, color=(255, 255, 255), thickness=1)
            # Calf rectangle (knee to ankle)
            draw_rect_from_points(image, knee, ankle, color=(255, 255, 255), thickness=1)
            # # Foot rectangle (ankle to foot_index)
            # draw_rect_from_points(image, ankle, foot, color=(255, 255, 255), thickness=1)

            all_green = abs(hip_angle - target_hip) <= tol and abs(knee_angle - target_knee) <= tol # and abs(heel_angle - target_heel) <= tol

            if all_green and not all_green_once:
                engine.say("Keep your leg fully straight")
                engine.runAndWait()
                all_green_once = True
                hold_start_time = time.time()

            if all_green_once:
                elapsed = time.time() - hold_start_time
                if elapsed >= 0 and elapsed < 5 and not hold_message_played:
                    engine.say("Hold for 5 seconds")
                    engine.runAndWait()
                    hold_message_played = True
                elif elapsed >= 5 and not relax_message_played:
                    engine.say("You can relax... what... stop doing the running man")
                    engine.runAndWait()
                    relax_message_played = True

            # Write angles on top right corner in white
            # text_x = int(w * 0.75)
            # cv2.putText(image, f'Hip Angle: {int(hip_angle)} deg', (text_x, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # cv2.putText(image, f'Knee Angle: {int(knee_angle)} deg', (text_x, 60),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # cv2.putText(image, f'Heel Angle: {int(heel_angle)} deg', (text_x, 90),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


        cv2.imshow('Leg Angles with Rigid Bodies', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
