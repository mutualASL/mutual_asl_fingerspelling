# ASL Live Translation with Hugging Face Inference API + UI
# Save this file and run. Set HUGGINGFACE_TOKEN as an environment variable or replace with your token string.

import os
import time
import math
from collections import deque
import threading

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import requests
import pygame
import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk

# ------------------- Configuration -------------------
IMG_SIZE = 200
HISTORY_LENGTH = 16
FRAMES_THRESHOLD = 4
CONFIDENCE_THRESHOLD = 0.70
TOLERANCE_THRESHOLD = 0.50
WORD_TIMEOUT = 0.35
SENTENCE_TIMEOUT = 5.0
CLEAR_TIMEOUT = 6.0
DOUBLE_LETTER_TIME = 0.7
DOUBLE_LETTER_STABILITY = 0.92
HF_TIMEOUT = 30

# ------------------- Audio feedback -------------------
pygame.mixer.init()
try:
    boop_sound = pygame.mixer.Sound("boop.wav")
except Exception:
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

# Load your trained ASL model and class names
try:
    model = tf.keras.models.load_model('v1exhaustiveroiasl_landmark_model.keras')
    with open('class_names.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print("Model and class names loaded successfully. Classes:", class_names)
except Exception as e:
    print(f"Error loading model or class names: {e}")
    raise

# ------------------- Startup Animation -------------------
def play_startup_animation(root, on_finish, video_filename="Startup copy.mp4", fade_frames=15):
    """Fixed startup animation - plays in main UI window, centered, not stretched"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(base_dir, video_filename)

    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Warning: Video file not found at {video_path}")
        print("Skipping startup animation...")
        on_finish()
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video file at {video_path}")
        print("Skipping startup animation...")
        on_finish()
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 120:  # fallback if fps detection fails
        fps = 60
    frame_delay = int(1000 / fps)  # Convert to milliseconds

    # Get original video dimensions
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {orig_width}x{orig_height} @ {fps} fps")

    # Make sure root window is visible
    root.deiconify()
    root.update()

    # Get window dimensions (1200x700 from ASLTranslationUI)
    window_w = 1200
    window_h = 700

    # Calculate scaled dimensions to fit in window while maintaining aspect ratio
    aspect_ratio = orig_width / orig_height
    if window_w / window_h > aspect_ratio:
        # Window is wider than video - fit to height
        display_h = window_h
        display_w = int(display_h * aspect_ratio)
    else:
        # Window is taller than video - fit to width
        display_w = window_w
        display_h = int(display_w / aspect_ratio)

    # Create canvas that covers the entire window
    canvas = tk.Canvas(root, width=window_w, height=window_h, highlightthickness=0, bg='white')
    canvas.place(x=0, y=0, relwidth=1, relheight=1)
    img_container = canvas.create_image(window_w//2, window_h//2, anchor=tk.CENTER)

    # Read all frames and resize them
    frames = []
    print("Loading startup animation frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize to fit window while maintaining aspect ratio
        frame = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()

    if not frames:
        print("Warning: No frames loaded from video")
        canvas.destroy()
        on_finish()
        return

    print(f"Loaded {len(frames)} frames @ {fps} fps. Starting animation...")

    # Convert to PhotoImage and store references
    photo_frames = []
    for f in frames:
        photo_frames.append(ImageTk.PhotoImage(f))

    # Keep references to prevent garbage collection
    canvas._photo_refs = photo_frames
    canvas._pil_frames = frames

    def show_frame(i=0):
        if i < len(photo_frames):
            canvas.itemconfig(img_container, image=photo_frames[i])
            # Use video's actual frame rate
            root.after(frame_delay, lambda: show_frame(i + 1))
        else:
            print("Animation complete. Starting fade out...")
            fade_out(0)

    def fade_out(step):
        if step >= fade_frames:
            print("Fade complete. Starting main app...")
            # Cleanup with delay to prevent warnings
            def cleanup():
                canvas.destroy()
                canvas._photo_refs = None
                canvas._pil_frames = None
                on_finish()
            root.after(50, cleanup)
            return

        alpha = step / fade_frames
        last_frame = np.array(frames[-1]).astype(np.float32)
        white = np.ones_like(last_frame) * 255
        blended = (last_frame * (1 - alpha) + white * alpha).astype(np.uint8)
        img = ImageTk.PhotoImage(Image.fromarray(blended))
        canvas.itemconfig(img_container, image=img)
        canvas._fade_ref = img
        root.after(25, lambda: fade_out(step + 1))

    # Start animation immediately
    show_frame(0)


# ------------------- Hugging Face relay server query -------------------
# This replaces the local HF token usage with your Railway server
HF_RELAY_URL = "https://hfrelay-production.up.railway.app/translate"  # <--- replace with your deployed server URL

def query_hf_llm(letters: str) -> str:
    if not letters:
        return ""

    try:
        response = requests.post(
            HF_RELAY_URL,
            headers={"Content-Type": "application/json"},
            json={"text": letters},
            timeout=HF_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        return data.get("result", "")
    except Exception as e:
        print(f"HF Relay server request failed: {e}")
        return ""
# ------------------- Hand landmarks processing -------------------
def get_hand_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    return results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None

def check_wrist_slope(hand_landmarks, image_shape):
    if not hand_landmarks:
        return None
    h, w = image_shape[:2]
    wrist = (int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h))
    pinky_mcp = (int(hand_landmarks.landmark[17].x * w), int(hand_landmarks.landmark[17].y * h))
    dx = pinky_mcp[0] - wrist[0]
    dy = pinky_mcp[1] - wrist[1]
    slope = float('inf') if dx == 0 else dy / dx
    return slope, wrist, pinky_mcp

def calculate_slope_and_adjust(hand_landmarks, image_shape, target_size=(200, 200)):
    if not hand_landmarks:
        return None
    h, w = image_shape[:2]
    target_h, target_w = target_size
    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

    wrist = landmarks[0]
    index_mcp, middle_mcp, ring_mcp, pinky_mcp = landmarks[5], landmarks[9], landmarks[13], landmarks[17]
    x_range_min, x_range_max = min(index_mcp[0], middle_mcp[0], ring_mcp[0], pinky_mcp[0]), max(index_mcp[0], middle_mcp[0], ring_mcp[0], pinky_mcp[0])
    x_range = x_range_max - x_range_min

    slope_info = check_wrist_slope(hand_landmarks, image_shape)
    slope = slope_info[0] if slope_info else None

    is_vertical = x_range_min <= wrist[0] <= x_range_max
    is_horizontal = (wrist[0] < x_range_min or wrist[0] > x_range_max) or (slope is not None and -0.5 <= slope <= 0.5)

    if is_vertical and not is_horizontal:
        fixed_wrist = (target_w // 2, target_h - 50)
        mcp_center = ((index_mcp[0] + middle_mcp[0] + ring_mcp[0] + pinky_mcp[0]) // 4,
                      (index_mcp[1] + middle_mcp[1] + ring_mcp[1] + pinky_mcp[1]) // 4)
        dx, dy = mcp_center[0] - wrist[0], mcp_center[1] - wrist[1]
        rotation_angle = (-math.pi / 2 - math.atan2(dy, dx)) if dx != 0 or dy != 0 else 0
        cos_theta, sin_theta = math.cos(rotation_angle), math.sin(rotation_angle)
        transformed_landmarks = [(int((x - wrist[0]) * cos_theta - (y - wrist[1]) * sin_theta + fixed_wrist[0]),
                                  int((x - wrist[0]) * sin_theta + (y - wrist[1]) * cos_theta + fixed_wrist[1]))
                                 for x, y in landmarks]
    elif is_horizontal and slope is not None and -0.5 <= slope <= 0.5:
        dx, dy = pinky_mcp[0] - wrist[0], pinky_mcp[1] - wrist[1]
        rotation_angle = (0 - math.atan2(dy, dx)) if dx != 0 or dy != 0 else 0
        cos_theta, sin_theta = math.cos(rotation_angle), math.sin(rotation_angle)
        transformed_landmarks = [(int((x - wrist[0]) * cos_theta - (y - wrist[1]) * sin_theta + wrist[0]),
                                  int((x - wrist[0]) * sin_theta + (y - wrist[1]) * cos_theta + wrist[1]))
                                 for x, y in landmarks]
    else:
        transformed_landmarks = landmarks

    x_coords = [pt[0] for pt in transformed_landmarks]
    y_coords = [pt[1] for pt in transformed_landmarks]
    x_min, x_max, y_min, y_max = min(x_coords), max(x_coords), min(y_coords), max(y_coords)
    padding = 20
    scale = min((target_w - 2 * padding) / (x_max - x_min if x_max > x_min else 1),
                (target_h - 2 * padding) / (y_max - y_min if y_max > y_min else 1))
    offset_x, offset_y = (target_w - ((x_max - x_min) * scale)) // 2, (target_h - ((y_max - y_min) * scale)) // 2
    final_landmarks = [(int((x - x_min) * scale + offset_x), int((y - y_min) * scale + offset_y)) for x, y in transformed_landmarks]

    landmark_image = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    for start_idx, end_idx in mp_hands.HAND_CONNECTIONS:
        if start_idx < len(final_landmarks) and end_idx < len(final_landmarks):
            cv2.line(landmark_image, final_landmarks[start_idx], final_landmarks[end_idx], (0, 0, 0), 1)
    for x, y in final_landmarks:
        cv2.circle(landmark_image, (x, y), 3, (0, 0, 0), 1)
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

# ------------------- Sequence processing -------------------
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
        last_letter = current_letters[-1] if current_letters else None
        if last_letter != most_common:
            current_letters.append(most_common)
            all_raw_letters.append(most_common)
            current_letter = most_common
            last_letter_time = current_time
            try: boop_sound.play()
            except: pass
        else:
            if last_letter_time and (current_time - last_letter_time >= DOUBLE_LETTER_TIME):
                stability = prediction_history.count(most_common) / len(prediction_history)
                if not is_chaotic and stability >= DOUBLE_LETTER_STABILITY:
                    current_letters.append(most_common)
                    all_raw_letters.append(most_common)
                    current_letter = most_common
                    last_letter_time = current_time
                    try: boop_sound.play()
                    except: pass
        last_confident_time = current_time

# ------------------- UI Setup -------------------
class ASLTranslationUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Live Translation")
        self.root.geometry("1200x700")
        self.root.configure(bg='white')

        self.top_frame = tk.Frame(root, bg='white')
        self.top_frame.pack(fill=tk.BOTH, expand=True)

        self.word_label = tk.Label(
            self.top_frame,
            text="",
            bg='white',
            fg='black',
            font=tkfont.Font(family="Helvetica", size=48, weight="bold"),
            wraplength=750
        )
        self.word_label.pack(pady=(100, 20))

        self.letter_label = tk.Label(
            self.top_frame,
            text="Last Letter: ",
            bg='white',
            fg='black',
            font=tkfont.Font(family="Helvetica", size=24)
        )
        self.letter_label.pack(pady=10)

        self.raw_letters_label = tk.Label(
            self.top_frame,
            text="Raw: ",
            bg='white',
            fg='black',
            font=tkfont.Font(family="Helvetica", size=16),
            wraplength=750
        )
        self.raw_letters_label.pack(pady=10)

        self.bottom_frame = tk.Frame(root, bg='#808080', height=180)
        self.bottom_frame.pack(fill=tk.BOTH, expand=True)
        self.bottom_frame.pack_propagate(False)

        self.sentence_label = tk.Label(
            self.bottom_frame,
            text="",
            bg='#808080',
            fg='white',
            font=tkfont.Font(family="Helvetica", size=20),
            wraplength=750,
            justify=tk.LEFT
        )
        self.sentence_label.pack(pady=40, padx=20)

        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_display(self, last_word, last_letter, full_sentence, raw_letters):
        self.word_label.config(text=last_word)
        self.letter_label.config(text=f"Last Letter: {last_letter}")
        self.sentence_label.config(text=full_sentence)
        self.raw_letters_label.config(text=f"Raw: {raw_letters}")

    def on_closing(self):
        self.running = False
        self.root.destroy()

# ------------------- Main loop -------------------
def main():
    global cap
    cap = None
    global current_letter, sentence_text, all_raw_letters, current_letters
    global is_recording, last_confident_time, last_letter_time, prediction_history

    root = tk.Tk()
    ui = ASLTranslationUI(root)
    root.withdraw()
    root.update_idletasks()
    root.update()

    last_word = ""
    current_landmark_img = None

    def start_main_app():
        # Show UI after animation
        root.deiconify()
        root.attributes('-topmost', False)

        global cap
        cap = cv2.VideoCapture(0)
        print("Live ASL Translation (HF API) - Close window to quit")

        def update_frame():
            global current_letter, sentence_text, all_raw_letters, current_letters
            global is_recording, last_confident_time, last_letter_time, prediction_history

            nonlocal last_word, current_landmark_img

            if not ui.running:
                if cap is not None:
                    cap.release()
                cv2.destroyAllWindows()
                pygame.mixer.quit()
                return

            ret, frame = cap.read()
            if not ret:
                root.after(10, update_frame)
                return

            frame = cv2.flip(frame, 1)
            hand_landmarks = get_hand_landmarks(frame)
            landmark_image = calculate_slope_and_adjust(hand_landmarks, frame.shape) if hand_landmarks else None
            current_landmark_img = landmark_image
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

            process_letter(prediction_history, current_time)

            if last_confident_time and (current_time - last_confident_time > WORD_TIMEOUT):
                if is_recording:
                    is_recording = False
                    combined = ''.join(all_raw_letters)
                    if combined:
                        print(f"Word finished: {combined} -> querying HF LLM...")
                        def query_thread():
                            global sentence_text
                            nonlocal last_word
                            sentence = query_hf_llm(combined)
                            if sentence:
                                sentence_text = sentence
                                words = sentence.split()
                                last_word = words[-1] if words else ""
                                print("HF LLM returned:", sentence_text)
                        threading.Thread(target=query_thread, daemon=True).start()
                        all_raw_letters = []
                        current_letters = []
                    last_letter_time = None

            if last_confident_time and (current_time - last_confident_time > SENTENCE_TIMEOUT):
                combined = ''.join(all_raw_letters)
                if combined:
                    print(f"Sentence timeout: {combined} -> querying HF LLM for full sentence...")
                    def query_thread():
                        global sentence_text
                        nonlocal last_word
                        sentence = query_hf_llm(combined)
                        if sentence:
                            sentence_text = sentence
                            words = sentence.split()
                            last_word = words[-1] if words else ""
                            print("HF LLM returned (sentence timeout):", sentence_text)
                    threading.Thread(target=query_thread, daemon=True).start()
                    all_raw_letters = []
                    current_letters = []

            if last_confident_time and (current_time - last_confident_time > CLEAR_TIMEOUT):
                all_raw_letters = []
                current_letters = []
                sentence_text = ""
                current_letter = ""
                last_word = ""
                is_recording = False
                last_confident_time = None
                prediction_history.clear()
                print("Cleared all state due to inactivity")

            raw_display = ''.join(all_raw_letters)
            ui.update_display(last_word, current_letter, sentence_text, raw_display)

            if current_landmark_img is not None:
                cv2.imshow('Landmark Image (Rotation Adjusted)',
                          cv2.resize(current_landmark_img, (500, 500), interpolation=cv2.INTER_NEAREST))
            else:
                blank = np.ones((500, 500, 3), dtype=np.uint8) * 255
                cv2.imshow('Landmark Image (Rotation Adjusted)', blank)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                ui.on_closing()
                return

            root.after(10, update_frame)

        update_frame()

    play_startup_animation(
        root,
        on_finish=start_main_app,
        video_filename="Startup copy.mp4"
    )

    root.mainloop()

    if cap is not None:
        cap.release()

    cv2.destroyAllWindows()
    pygame.mixer.quit()


if __name__ == "__main__":
    main()
