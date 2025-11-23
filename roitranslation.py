# ASL Live Translation with Hugging Face Inference API
# Save this file and run. Set HUGGINGFACE_TOKEN as an environment variable or replace with your token string.

import os
import time
import math
import json
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import requests
import pygame

# ------------------- Configuration -------------------
IMG_SIZE = 200
HISTORY_LENGTH = 25
CONFIDENCE_THRESHOLD = 0.75
DETECTION_THRESHOLD = 0.5
FRAMES_THRESHOLD = 10
TOLERANCE_THRESHOLD = 0.95
WORD_TIMEOUT = 1.5
SENTENCE_TIMEOUT = 5.0
CLEAR_TIMEOUT = 6.0
DOUBLE_LETTER_TIME = 3.0
DOUBLE_LETTER_STABILITY = 0.95
HF_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
HF_TOKEN = "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"  # <-- put your token here or set env var
HF_TIMEOUT = 30  # seconds for HF API call

# ------------------- Audio feedback -------------------
pygame.mixer.init()
try:
    boop_sound = pygame.mixer.Sound("boop.wav")
except Exception:
    # generate a short beep if boop.wav not available
    sample_rate = 44100
    freq = 500
    duration = 0.12
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * freq * t) * 32767
    audio_data = audio_data.astype(np.int16)
    boop_sound = pygame.mixer.Sound(audio_data.tobytes())
    boop_sound.set_volume(0.5)

# ------------------- Mediapipe / Model setup -------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8)

# Load your trained ASL classification model and class names
try:
    model = tf.keras.models.load_model('v1exhaustiveroiasl_landmark_model.keras')
    with open('class_names.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print("Model and class names loaded successfully. Classes:", class_names)
except Exception as e:
    print(f"Error loading model or class names: {e}")
    raise

# ------------------- Utilities -------------------

def query_hf_llm(letters: str) -> str:
    """
    Uses Hugging Face Router API to convert ASL finger-spelled letters
    into a natural English sentence using Llama-3-8B-Instruct (novita).
    """
    if not letters:
        return ""

    API_URL = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    prompt = (
        "You are a helpful assistant that converts ASL finger-spelled letters "
        "into a natural English sentence.\n\n"
        f"Input letters: {letters}\n\n"
        "Interpret the letters as English text and output a single concise, "
        "natural English sentence. If needed, split into multiple words. "
        "Return ONLY the sentence."
    )

    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct:novita",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 128,
        "temperature": 0.2
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=HF_TIMEOUT)
    except Exception as e:
        print(f"HF API request failed: {e}")
        return ""

    try:
        data = resp.json()
    except:
        print("HF: Failed to parse JSON")
        return ""

    if resp.status_code != 200:
        print("HF API error:", data)
        return ""

    try:
        return data["choices"][0]["message"]["content"].strip()
    except:
        print("HF API unexpected response format:", data)
        return ""

# ------------------- Landmark processing (kept mostly as your previous implementation) -------------------

def get_hand_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    return results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None


def calculate_slope_and_adjust(hand_landmarks, image_shape, target_size=(200, 200)):
    if not hand_landmarks:
        return None
    h, w = image_shape[:2]
    target_h, target_w = target_size
    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
    x_coords = [pt[0] for pt in landmarks]
    y_coords = [pt[1] for pt in landmarks]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    padding = 20
    scale = min((target_w - 2 * padding) / (x_max - x_min) if x_max > x_min else 1,
                (target_h - 2 * padding) / (y_max - y_min) if y_max > y_min else 1)
    offset_x = (target_w - ((x_max - x_min) * scale)) // 2
    offset_y = (target_h - ((y_max - y_min) * scale)) // 2
    final_landmarks = []
    for x, y in landmarks:
        new_x = int((x - x_min) * scale + offset_x)
        new_y = int((y - y_min) * scale + offset_y)
        final_landmarks.append((new_x, new_y))
    landmark_image = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    connections = mp_hands.HAND_CONNECTIONS
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(final_landmarks) and end_idx < len(final_landmarks):
            cv2.line(landmark_image, final_landmarks[start_idx], final_landmarks[end_idx], (0, 0, 0), 1)
    for point in final_landmarks:
        cv2.circle(landmark_image, point, 3, (0, 0, 0), 1)
    if len(final_landmarks) >= 18:
        cv2.line(landmark_image, final_landmarks[0], final_landmarks[17], (255, 0, 0), 2)
    return landmark_image


def preprocess_image(landmark_image):
    if landmark_image is None:
        return None
    gray = cv2.cvtColor(landmark_image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype('float32') / 255.0
    return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

# ------------------- Sequence processing (simplified) -------------------

prediction_history = deque(maxlen=HISTORY_LENGTH)
current_letters = []
all_raw_letters = []
is_recording = False
last_confident_time = None
last_letter_time = None
current_letter = ""
sentence_text = ""


def process_letter(prediction_history, current_time):
    global current_letters, all_raw_letters, is_recording, last_confident_time, last_letter_time, current_letter
    if not prediction_history or len(prediction_history) < FRAMES_THRESHOLD:
        return

    most_common = max(set(prediction_history), key=prediction_history.count)
    confidence_ratio = prediction_history.count(most_common) / len(prediction_history)
    unique_predictions = len(set(prediction_history))
    is_chaotic = confidence_ratio < 0.5 or (unique_predictions / HISTORY_LENGTH > 0.5)

    if confidence_ratio >= TOLERANCE_THRESHOLD and most_common in class_names:
        if not is_recording:
            is_recording = True
            current_letters = []
            print(f"Started recording at {current_time}")

        last_letter = current_letters[-1] if current_letters else None

        if last_letter != most_common:
            # New letter detected → append
            current_letters.append(most_common)
            all_raw_letters.append(most_common)
            current_letter = most_common
            last_letter_time = current_time
            try:
                boop_sound.play()
            except:
                pass
        else:
            # Same letter as before → check if we should repeat it
            if last_letter_time and (current_time - last_letter_time >= DOUBLE_LETTER_TIME):
                # Stable long hold → repeat letter
                stability = prediction_history.count(most_common) / len(prediction_history)
                if not is_chaotic and stability >= DOUBLE_LETTER_STABILITY:
                    current_letters.append(most_common)
                    all_raw_letters.append(most_common)
                    current_letter = most_common
                    last_letter_time = current_time
                    try:
                        boop_sound.play()
                    except:
                        pass

        last_confident_time = current_time

# ------------------- Main loop -------------------

cap = cv2.VideoCapture(0)
print("Live ASL Translation (HF API) - Press 'q' to Quit")
last_active_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    hand_landmarks = get_hand_landmarks(frame)
    landmark_image = calculate_slope_and_adjust(hand_landmarks, frame.shape) if hand_landmarks else None
    current_time = time.time()

    if landmark_image is not None:
        processed_image = preprocess_image(landmark_image)
        if processed_image is not None:
            prediction = model.predict(processed_image, verbose=0)
            predicted_index = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            if confidence > CONFIDENCE_THRESHOLD:
                current_prediction = class_names[predicted_index] if predicted_index < len(class_names) else ""
                prediction_history.append(current_prediction)
                last_confident_time = current_time
            else:
                current_prediction = ""
        else:
            current_prediction = ""
    else:
        current_prediction = ""
        prediction_history.clear()

    # Process letter sequence
    process_letter(prediction_history, current_time)

    # Timeouts: when a word finishes, query HF LLM to craft sentence
    if last_confident_time and (current_time - last_confident_time > WORD_TIMEOUT):
        if is_recording:
            is_recording = False
            # Gather the letters we've seen
            combined = ''.join(all_raw_letters)  # only all_raw_letters, do not add current_letters again
            if combined:
                print(f"Word finished: {combined} -> querying HF LLM...")
                sentence = query_hf_llm(combined)
                if sentence:
                    sentence_text = sentence
                    print("HF LLM returned:", sentence_text)
                # clear collected letters after sending to HF
                all_raw_letters = []
                current_letters = []
            last_letter_time = None

    # When a longer pause occurs, treat as end of sentence and re-query full history
    if last_confident_time and (current_time - last_confident_time > SENTENCE_TIMEOUT):
        combined = ''.join(all_raw_letters)  # only use all_raw_letters
        if combined:
            print(f"Sentence timeout: {combined} -> querying HF LLM for full sentence...")
            sentence = query_hf_llm(combined)
            if sentence:
                sentence_text = sentence
                print("HF LLM returned (sentence timeout):", sentence_text)
            all_raw_letters = []
            current_letters = []

    # Clear everything after long inactivity
    if last_confident_time and (current_time - last_confident_time > CLEAR_TIMEOUT):
        all_raw_letters = []
        current_letters = []
        sentence_text = ""
        current_letter = ""
        is_recording = False
        last_confident_time = None
        prediction_history.clear()
        print("Cleared all state due to inactivity")

    # Display UI overlays
    bar_height = 120
    cv2.rectangle(frame, (0, frame.shape[0] - bar_height), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.putText(frame, f"Prediction: {current_letter}", (20, frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.putText(frame, f"Raw letters: {''.join(all_raw_letters)}", (20, frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame, f"Sentence: {sentence_text}", (20, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)

    if hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('ASL Live Translation', frame)
    if landmark_image is not None:
        cv2.imshow('Landmark Image', cv2.resize(landmark_image, (400,400), interpolation=cv2.INTER_NEAREST))
    else:
        cv2.imshow('Landmark Image', np.ones((400,400,3), dtype=np.uint8) * 255)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
