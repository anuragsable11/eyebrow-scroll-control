import cv2
import mediapipe as mp
import pyautogui
import time

# Setup Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Facial landmark indices (MediaPipe)
LEFT_EYEBROW_IDX = 65
RIGHT_EYEBROW_IDX = 295
LEFT_EYE_IDX = 159
RIGHT_EYE_IDX = 386

# Webcam
cap = cv2.VideoCapture(0)

# Initialization
baseline_samples = []
baseline_distance = None
scroll_cooldown = 1.0  # Seconds between scrolls
last_scroll_time = time.time()
scroll_triggered = False
frame_counter = 0

def get_landmark_y(landmarks, idx, height):
    return int(landmarks[idx].y * height)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        # Get Y positions
        left_eyebrow_y = get_landmark_y(landmarks, LEFT_EYEBROW_IDX, h)
        left_eye_y = get_landmark_y(landmarks, LEFT_EYE_IDX, h)
        right_eyebrow_y = get_landmark_y(landmarks, RIGHT_EYEBROW_IDX, h)
        right_eye_y = get_landmark_y(landmarks, RIGHT_EYE_IDX, h)

        # Calculate distances
        left_dist = abs(left_eyebrow_y - left_eye_y)
        right_dist = abs(right_eyebrow_y - right_eye_y)
        avg_dist = (left_dist + right_dist) / 2

        # Calibration: collect baseline samples (30 frames)
        if baseline_distance is None and frame_counter < 30:
            baseline_samples.append(avg_dist)
            frame_counter += 1
            cv2.putText(frame, f"Calibrating... {30 - frame_counter}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        elif baseline_distance is None:
            baseline_distance = sum(baseline_samples) / len(baseline_samples)
            print(f"[INFO] Baseline set at: {baseline_distance:.2f}")
        else:
            # Detect eyebrow raise
            diff = avg_dist - baseline_distance
            current_time = time.time()

            # Show debug text
            cv2.putText(frame, f"Diff: {diff:.2f}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if diff < -5 and not scroll_triggered and current_time - last_scroll_time > scroll_cooldown:
                pyautogui.press('down')  # Simulate arrow down key
                print("✅ Eyebrows Raised - Pressed DOWN key")
                scroll_triggered = True
                last_scroll_time = current_time

            elif diff > -3:
                scroll_triggered = False

    # Show the webcam feed
    cv2.imshow("Eyebrow Scroll", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
