import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

if not os.path.exists('data'):
    os.makedirs('data')

asl_signs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'] # Add more signs as needed
current_sign_index = 0
recording = False
frame_count = 0

def create_white_image(height, width):
    return np.ones((height, width, 3), dtype=np.uint8) * 255

def get_hand_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        return results.multi_hand_landmarks[0]
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)  # Mirror the frame
    white_image = create_white_image(frame.shape[0], frame.shape[1])

    hand_landmarks = get_hand_landmarks(frame)

    if hand_landmarks:
        mp_drawing.draw_landmarks(
            white_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2)
        )

    cv2.putText(frame, f"Current sign: {asl_signs[current_sign_index]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'r' to start/stop recording", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'n' for next sign", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if recording and hand_landmarks:
        cv2.putText(frame, "Recording...", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        sign_dir = os.path.join('data', asl_signs[current_sign_index])
        if not os.path.exists(sign_dir):
            os.makedirs(sign_dir)

        cv2.imwrite(os.path.join(sign_dir, f'{frame_count}.jpg'), white_image)
        frame_count += 1

    cv2.imshow('ASL Data Collection', frame)
    cv2.imshow('Hand Landmarks', white_image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        recording = not recording
        if recording:
            frame_count = 0
        else:
            print(f"Recorded {frame_count} frames for sign '{asl_signs[current_sign_index]}'")
    elif key == ord('n'):
        current_sign_index = (current_sign_index + 1) % len(asl_signs)
        recording = False
        frame_count = 0

cap.release()
cv2.destroyAllWindows()
