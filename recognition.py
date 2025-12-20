import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
from collections import deque
import time
from gtts import gTTS
import os
from difflib import get_close_matches

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Word list for autocorrect
WORD_LIST = [
    "hello", "world", "thank", "you", "please", "help", "good", "morning",
    "evening", "night", "friend", "computer", "program", "python", "code",
    "sign", "language", "learn", "study", "practice", "perfect", "great"
]

# Load the trained model
try:
    model = load_model('asl_recognition_model3.keras')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the class names
try:
    with open('class_names.txt', 'r') as f:
        asl_signs = [line.strip() for line in f]
    print(f"Loaded {len(asl_signs)} signs: {asl_signs}")
except Exception as e:
    print(f"Error loading class names: {e}")
    asl_signs = []

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

# Initialize temporal smoothing filter
prediction_queue = deque(maxlen=10)

# Initialize word formation variables
current_word = ""
last_recognized_letter = None
letter_recognition_start_time = None
letter_captured = False
captured_letters = []
no_hand_start_time = None
word_spoken = False

def create_white_image(height, width):
    return np.ones((height, width, 3), dtype=np.uint8) * 255

def get_hand_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        return results.multi_hand_landmarks[0]
    return None

def autocorrect_word(word, min_confidence=0.6):
    if word.lower() in WORD_LIST:
        return word

    matches = get_close_matches(word.lower(), WORD_LIST, n=1, cutoff=min_confidence)
    if matches:
        corrected_word = matches[0]
        print(f"Autocorrected '{word}' to '{corrected_word}'")
        return corrected_word
    return word

def speak_word(word):
    tts = gTTS(text=word, lang='en')
    tts.save("temp.mp3")
    os.system("afplay temp.mp3")

def create_glowing_effect(image, t, hand_in_view_duration):
    h, w = image.shape[:2]
    border_size = 60

    if hand_in_view_duration >= 0.5:
        glow_speed = 2.0
        color_intensity = 1.5
    else:
        glow_speed = 1.0
        color_intensity = 1.0

    glow_image = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(border_size):
        alpha = (border_size - i) / border_size
        pink = np.array([255, 128, 255])
        blue = np.array([255, 165, 0])
        color = (pink * (np.sin(t * glow_speed) * 0.5 + 0.5) + blue * (np.cos(t * glow_speed) * 0.5 + 0.5)) * color_intensity
        color = color * alpha
        cv2.rectangle(glow_image, (i, i), (w-i, h-i), color.tolist(), thickness=1)

    glow_image = cv2.GaussianBlur(glow_image, (101, 101), 0)
    combined_image = cv2.addWeighted(image, 1, glow_image, 0.8, 0)

    return combined_image

# Main loop
t = 0
frame_count = 0
hand_in_view_start_time = None
hand_in_view_duration = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Frame shape: {frame.shape}")

    frame = cv2.flip(frame, 1)
    white_image = create_white_image(frame.shape[0], frame.shape[1])

    hand_landmarks = get_hand_landmarks(frame)

    if hand_landmarks:
        word_spoken = False  # Reset the flag when hand is detected again
        if hand_in_view_start_time is None:
            hand_in_view_start_time = time.time()

        hand_in_view_duration = time.time() - hand_in_view_start_time

        mp_drawing.draw_landmarks(
            white_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2)
        )

        if model is not None and asl_signs:
            gray_image = cv2.cvtColor(white_image, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray_image, (96, 96))  # Resize to 96x96
            normalized_image = resized_image.astype('float32') / 255
            input_image = normalized_image.reshape(1, 96, 96, 1)  # Reshape to (1, 96, 96, 1)

            prediction = model.predict(input_image)
            predicted_index = np.argmax(prediction)
            confidence = prediction[0][predicted_index]

            prediction_queue.append(predicted_index)
            smoothed_prediction = max(set(prediction_queue), key=prediction_queue.count)

            if confidence > 0.7:
                predicted_sign = asl_signs[smoothed_prediction]

                # Letter capture logic
                if predicted_sign != last_recognized_letter:
                    letter_recognition_start_time = time.time()
                    letter_captured = False
                    last_recognized_letter = predicted_sign
                elif not letter_captured and (time.time() - letter_recognition_start_time) >= 2.0:
                    captured_letters.append(predicted_sign)
                    letter_captured = True
                    current_word = ''.join(captured_letters)
                    print(f"Letter captured: {predicted_sign}")

                cv2.putText(frame, f"Predicted: {predicted_sign}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.putText(white_image, f"Word: {current_word}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                if not letter_captured and letter_recognition_start_time:
                    hold_time = time.time() - letter_recognition_start_time
                    if hold_time < 1.0:
                        cv2.putText(white_image, f"Hold for: {1.0 - hold_time:.1f}s",
                                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    else:
        if hand_in_view_start_time is not None:  # Hand just disappeared
            if current_word and not word_spoken:
                corrected_word = autocorrect_word(current_word)
                speak_word(corrected_word)
                word_spoken = True
                captured_letters = []
                current_word = ""
        hand_in_view_start_time = None
        hand_in_view_duration = 0

    white_image_resized = cv2.resize(white_image, (frame.shape[1], frame.shape[0]))
    combined_image = np.vstack((frame, white_image_resized))
    glowing_frame = create_glowing_effect(combined_image, t, hand_in_view_duration)
    cv2.imshow('ASL Recognition', glowing_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    t += 0.1

cap.release()
cv2.destroyAllWindows()
