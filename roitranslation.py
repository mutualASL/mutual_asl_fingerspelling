# ASL Live Translation — Mac version
# Uses TFLite (via tensorflow package) + cv2.VideoCapture webcam

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
from PIL import Image, ImageTk, ImageDraw

# ------------------- Configuration -------------------
IMG_SIZE                = 200
HISTORY_LENGTH          = 16
FRAMES_THRESHOLD        = 2
CONFIDENCE_THRESHOLD    = 0.70
TOLERANCE_THRESHOLD     = 0.50
WORD_TIMEOUT            = 0.35
SENTENCE_TIMEOUT        = 5.0
CLEAR_TIMEOUT           = 9.0
DOUBLE_LETTER_TIME      = 1.2
DOUBLE_LETTER_STABILITY = 0.92
DOUBLE_LETTER_MAX       = 2.2
HF_TIMEOUT              = 30
CAPSULE_LINGER          = 2.0

MOTION_LETTERS          = {'Z', 'J'}
MOTION_FRAMES_THRESHOLD = 8
MOTION_MIN_RATIO        = 0.65

# Panel constants
PANEL_W   = 340
PANEL_H_RATIO = 0.91   # panel height = window_height * this ratio
PANEL_PAD = 20         # gap from right edge when open
PANEL_R   = 32

# ------------------- Audio feedback -------------------
pygame.mixer.init()
try:
    boop_sound = pygame.mixer.Sound("boop.wav")
except Exception:
    sample_rate = 44100
    freq        = 500
    duration    = 0.12
    t           = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data  = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    boop_sound  = pygame.mixer.Sound(audio_data.tobytes())
    boop_sound.set_volume(0.5)

# ------------------- Mediapipe / Model setup -------------------
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                             min_detection_confidence=0.8)

try:
    _interpreter = tf.lite.Interpreter(model_path='v1exhaustiveroiasl_landmark_model.tflite')
    _interpreter.allocate_tensors()
    _input_details  = _interpreter.get_input_details()
    _output_details = _interpreter.get_output_details()
    with open('class_names.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print("TFLite model and class names loaded. Classes:", class_names)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

def tflite_predict(processed_image):
    _interpreter.set_tensor(_input_details[0]['index'], processed_image)
    _interpreter.invoke()
    return _interpreter.get_tensor(_output_details[0]['index'])

# ------------------- Startup Animation (commented out) -------------------
# def play_startup_animation(...): ...

# ------------------- HF Relay -------------------
HF_RELAY_URL = "https://hfrelay-production.up.railway.app/translate"

def query_hf_llm(letters: str) -> str:
    if not letters:
        return ""
    try:
        r = requests.post(HF_RELAY_URL,
                          headers={"Content-Type": "application/json"},
                          json={"text": letters}, timeout=HF_TIMEOUT)
        r.raise_for_status()
        return r.json().get("result", "")
    except Exception as e:
        print(f"HF relay error: {e}")
        return ""

# ------------------- Hand landmark processing -------------------
def get_hand_landmarks(image, already_rgb=False):
    rgb     = image if already_rgb else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    return results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None

def check_wrist_slope(hand_landmarks, image_shape):
    if not hand_landmarks:
        return None
    h, w   = image_shape[:2]
    wrist  = (int(hand_landmarks.landmark[0].x * w),  int(hand_landmarks.landmark[0].y * h))
    pinky  = (int(hand_landmarks.landmark[17].x * w), int(hand_landmarks.landmark[17].y * h))
    dx, dy = pinky[0] - wrist[0], pinky[1] - wrist[1]
    slope  = float('inf') if dx == 0 else dy / dx
    return slope, wrist, pinky

def calculate_slope_and_adjust(hand_landmarks, image_shape, target_size=(200, 200)):
    if not hand_landmarks:
        return None
    h, w   = image_shape[:2]
    th, tw = target_size
    lm     = [(int(l.x * w), int(l.y * h)) for l in hand_landmarks.landmark]

    wrist                       = lm[0]
    i_mcp, m_mcp, r_mcp, p_mcp = lm[5], lm[9], lm[13], lm[17]
    xr_min = min(i_mcp[0], m_mcp[0], r_mcp[0], p_mcp[0])
    xr_max = max(i_mcp[0], m_mcp[0], r_mcp[0], p_mcp[0])

    slope_info  = check_wrist_slope(hand_landmarks, image_shape)
    slope       = slope_info[0] if slope_info else None
    is_vertical = xr_min <= wrist[0] <= xr_max
    is_horiz    = (wrist[0] < xr_min or wrist[0] > xr_max) or \
                  (slope is not None and -0.5 <= slope <= 0.5)

    if is_vertical and not is_horiz:
        fw      = (tw // 2, th - 50)
        mc      = ((i_mcp[0]+m_mcp[0]+r_mcp[0]+p_mcp[0])//4,
                   (i_mcp[1]+m_mcp[1]+r_mcp[1]+p_mcp[1])//4)
        dx, dy  = mc[0]-wrist[0], mc[1]-wrist[1]
        ang     = (-math.pi/2 - math.atan2(dy, dx)) if (dx or dy) else 0
        ca, sa  = math.cos(ang), math.sin(ang)
        xlm     = [(int((x-wrist[0])*ca-(y-wrist[1])*sa+fw[0]),
                    int((x-wrist[0])*sa+(y-wrist[1])*ca+fw[1])) for x, y in lm]
    elif is_horiz and slope is not None and -0.5 <= slope <= 0.5:
        dx, dy  = p_mcp[0]-wrist[0], p_mcp[1]-wrist[1]
        ang     = (0 - math.atan2(dy, dx)) if (dx or dy) else 0
        ca, sa  = math.cos(ang), math.sin(ang)
        xlm     = [(int((x-wrist[0])*ca-(y-wrist[1])*sa+wrist[0]),
                    int((x-wrist[0])*sa+(y-wrist[1])*ca+wrist[1])) for x, y in lm]
    else:
        xlm = lm

    xs, ys   = [p[0] for p in xlm], [p[1] for p in xlm]
    xmn, xmx = min(xs), max(xs)
    ymn, ymx = min(ys), max(ys)
    pad      = 20
    sc       = min((tw-2*pad) / (xmx-xmn if xmx>xmn else 1),
                   (th-2*pad) / (ymx-ymn if ymx>ymn else 1))
    ox       = (tw - (xmx-xmn)*sc) // 2
    oy       = (th - (ymx-ymn)*sc) // 2
    flm      = [(int((x-xmn)*sc+ox), int((y-ymn)*sc+oy)) for x, y in xlm]

    img = np.ones((th, tw, 3), dtype=np.uint8) * 255
    for s, e in mp_hands.HAND_CONNECTIONS:
        if s < len(flm) and e < len(flm):
            cv2.line(img, flm[s], flm[e], (0, 0, 0), 1)
    for x, y in flm:
        cv2.circle(img, (x, y), 3, (0, 0, 0), 1)
    if len(flm) >= 18:
        cv2.line(img, flm[0], flm[17], (255, 0, 0), 2)
    return img

def preprocess_image(lm_img):
    if lm_img is None:
        return None
    gray = cv2.cvtColor(lm_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    return (resized.astype('float32') / 255.0).reshape(1, IMG_SIZE, IMG_SIZE, 1)

def get_hand_crop_with_landmarks(frame_bgr, hand_landmarks, padding=50):
    if hand_landmarks is None or frame_bgr is None:
        return None
    h, w = frame_bgr.shape[:2]
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]
    x1 = max(0, int(min(xs)) - padding)
    x2 = min(w, int(max(xs)) + padding)
    y1 = max(0, int(min(ys)) - padding)
    y2 = min(h, int(max(ys)) + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    frame_copy = frame_bgr.copy()
    mp_drawing.draw_landmarks(
        frame_copy, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 200, 80),  thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(255, 80,  0),  thickness=2)
    )
    return frame_copy[y1:y2, x1:x2]

# ------------------- Sequence processing -------------------
prediction_history  = deque(maxlen=HISTORY_LENGTH)
current_letters     = []
all_raw_letters     = []
is_recording        = False
last_confident_time = None
last_letter_time    = None
current_letter      = ""
sentence_text       = ""

_recent_letter_intervals = deque(maxlen=6)
_last_commit_time        = None

def _adaptive_double_time():
    if len(_recent_letter_intervals) < 2:
        return DOUBLE_LETTER_TIME
    avg_interval = sum(_recent_letter_intervals) / len(_recent_letter_intervals)
    adaptive = DOUBLE_LETTER_TIME + max(0.0, (0.4 - avg_interval) * 3.0)
    return min(DOUBLE_LETTER_MAX, max(DOUBLE_LETTER_TIME, adaptive))

def process_letter(prediction_history, current_time):
    global current_letters, all_raw_letters, is_recording
    global last_confident_time, last_letter_time, current_letter
    global _last_commit_time

    if not prediction_history or len(prediction_history) < FRAMES_THRESHOLD:
        return None

    most_common      = max(set(prediction_history), key=prediction_history.count)
    confidence_ratio = prediction_history.count(most_common) / len(prediction_history)
    unique_preds     = len(set(prediction_history))
    is_chaotic       = confidence_ratio < 0.5 or (unique_preds / HISTORY_LENGTH > 0.5)

    is_motion   = most_common.upper() in MOTION_LETTERS
    req_frames  = MOTION_FRAMES_THRESHOLD if is_motion else FRAMES_THRESHOLD
    req_ratio   = MOTION_MIN_RATIO        if is_motion else TOLERANCE_THRESHOLD

    if len(prediction_history) < req_frames:
        return None

    if confidence_ratio >= req_ratio and most_common in class_names:
        if not is_recording:
            is_recording    = True
            current_letters = []

        last_letter = current_letters[-1] if current_letters else None

        if last_letter != most_common:
            current_letters.append(most_common)
            all_raw_letters.append(most_common)
            current_letter = most_common
            if _last_commit_time is not None:
                interval = current_time - _last_commit_time
                if interval < 5.0:
                    _recent_letter_intervals.append(interval)
            _last_commit_time   = current_time
            last_letter_time    = current_time
            last_confident_time = current_time
            try: boop_sound.play()
            except: pass
            return most_common

        else:
            dynamic_threshold = _adaptive_double_time()
            if last_letter_time and (current_time - last_letter_time >= dynamic_threshold):
                stability = prediction_history.count(most_common) / len(prediction_history)
                if not is_chaotic and stability >= DOUBLE_LETTER_STABILITY:
                    current_letters.append(most_common)
                    all_raw_letters.append(most_common)
                    current_letter = most_common
                    if _last_commit_time is not None:
                        interval = current_time - _last_commit_time
                        if interval < 5.0:
                            _recent_letter_intervals.append(interval)
                    _last_commit_time   = current_time
                    last_letter_time    = current_time
                    last_confident_time = current_time
                    try: boop_sound.play()
                    except: pass
                    return most_common

        last_confident_time = current_time

    return None

# ------------------- Capsule spring physics -------------------
def _spring(pos, vel, target, k, d):
    f   = (target - pos) * k
    vel = vel * d + f
    return pos + vel, vel


class Pill:
    W = 40
    H = 40

    def __init__(self, letter, tx):
        self.letter  = letter
        self.tx      = float(tx)
        self.x       = float(tx)
        self.y       = -float(self.H) * 1.8
        self.ty      = 0.0
        self.vx      = 0.0
        self.vy      = 0.0
        self.scale   = 0.0
        self.vs      = 0.0
        self.alpha   = 1.0
        self.exiting = False
        self.done    = False

    def tick(self):
        self.x,     self.vx = _spring(self.x,     self.vx, self.tx,  0.42, 0.60)
        self.y,     self.vy = _spring(self.y,     self.vy, self.ty,  0.42, 0.60)
        ts = 0.0 if self.exiting else 1.0
        self.scale, self.vs = _spring(self.scale, self.vs, ts,       0.36, 0.56)
        self.scale = max(0.0, min(self.scale, 1.6))
        if self.exiting:
            self.alpha = max(0.0, self.alpha - 0.10)
            if self.alpha <= 0.02:
                self.done = True

    def exit(self):
        self.exiting = True
        self.ty      = float(self.H) * 1.5

    @property
    def settled(self):
        return (abs(self.ty - self.y)                            < 0.6 and
                abs(self.tx - self.x)                            < 0.6 and
                abs((0.0 if self.exiting else 1.0) - self.scale) < 0.02)


class CapsuleLetterDisplay(tk.Canvas):
    PILL_W  = 40
    PILL_H  = 40
    GAP     = 6
    PAD_X   = 14
    CAP_H   = 76

    BG       = "#F2F2F2"
    OUTLINE  = "#CCCCCC"
    C_NORM   = "#E0E0E0"
    C_LATEST = "#1A1A1A"
    T_NORM   = "#333333"
    T_LATEST = "#FFFFFF"

    def __init__(self, parent, max_visible=14, **kw):
        self._letters    = []
        self._pills      = []
        self._exiting    = []
        self._running    = False
        self._max_visible = max_visible
        w = self._W(max_visible)
        super().__init__(parent, width=w, height=self.CAP_H,
                         bg=parent["bg"], highlightthickness=0, **kw)
        self._redraw()

    # ── public ──────────────────────────────────────────────────────────────

    def set_max_visible(self, n):
        """Change how many pills are visible. Excess letters scroll off left."""
        if n == self._max_visible:
            return
        self._max_visible = max(1, n)
        # Resize canvas width
        self.config(width=self._W(self._max_visible))
        # Re-sync pills for the new visible window
        self.set_letters(self._letters)

    def set_letters(self, letters: list):
        new     = list(letters)
        visible = new[-self._max_visible:]   # always show the LATEST n letters

        if not new:
            for p in self._pills:
                p.exit()
            self._exiting.extend(self._pills)
            self._pills = []
        else:
            # Pills that no longer fit scroll off to the left (exit)
            drop = len(self._pills) - len(visible)
            if drop > 0:
                for p in self._pills[:drop]:
                    p.exit()
                self._exiting.extend(self._pills[:drop])
                self._pills = self._pills[drop:]

            # Re-target x for remaining pills (they may need to shift)
            for i, p in enumerate(self._pills):
                p.tx = float(self._X(i))

            # Spawn new pills for letters that appeared
            while len(self._pills) < len(visible):
                idx = len(self._pills)
                self._pills.append(Pill(visible[idx], self._X(idx)))

        self._letters = new
        self._kick()

    # ── internals ───────────────────────────────────────────────────────────

    def _W(self, n):
        return 2*self.PAD_X + n*self.PILL_W + max(n-1, 0)*self.GAP

    def _X(self, i):
        return self.PAD_X + i * (self.PILL_W + self.GAP)

    def _kick(self):
        if not self._running:
            self._running = True
            self._tick()

    def _tick(self):
        for p in self._pills:
            p.tick()
        for p in self._exiting:
            p.tick()
        self._exiting = [p for p in self._exiting if not p.done]

        self._redraw()

        moving = (any(not p.settled or p.exiting for p in self._pills)
                  or bool(self._exiting))
        if moving:
            self.after(14, self._tick)
        else:
            self._running = False

    def _redraw(self):
        self.delete("all")
        w  = self._W(self._max_visible)
        h  = self.CAP_H
        cy = h // 2
        n  = len(self._pills)

        self.config(width=w)
        self._rrect(0, 4, w, h-4, 26, fill=self.BG, outline=self.OUTLINE, width=1)

        for i, pill in enumerate(self._pills):
            self._dpill(pill, cy, is_latest=(i == n-1))
        for pill in self._exiting:
            self._dpill(pill, cy, is_latest=False)

    def _dpill(self, pill, cy, is_latest):
        s = max(0.0, pill.scale)
        a = pill.alpha
        if s < 0.03 or a < 0.03:
            return

        pw  = max(4, int(self.PILL_W * s))
        ph  = max(4, int(self.PILL_H * s))
        pr  = ph // 2
        cx  = int(pill.x + self.PILL_W / 2)
        cy2 = int(cy + pill.y)

        fill = self.C_LATEST if is_latest else self.C_NORM
        tcol = self.T_LATEST if is_latest else self.T_NORM

        fc = self._blend(self._rgb(fill), self._rgb(self.BG), a)
        tc = self._blend(self._rgb(tcol), self._rgb(self.BG), a)

        self._rrect(cx-pw//2, cy2-ph//2, cx+pw//2, cy2+ph//2,
                    pr, fill=self._hex(fc), outline="")

        if s > 0.45:
            self.create_text(cx, cy2, text=pill.letter.upper(),
                             fill=self._hex(tc),
                             font=("Helvetica", max(9, int(15*s)), "bold"))

        # "now" indicator under latest pill
        if is_latest and s > 0.7:
            dot_y   = cy2 + ph//2 + 6
            dot_col = self._blend(self._rgb('#1A1A1A'), self._rgb(self.BG), a * 0.7)
            lbl_col = self._blend(self._rgb('#888888'), self._rgb(self.BG), a * 0.8)
            self.create_oval(cx-2, dot_y-2, cx+2, dot_y+2,
                             fill=self._hex(dot_col), outline="")
            self.create_text(cx, dot_y + 10, text="now",
                             fill=self._hex(lbl_col), font=("Helvetica", 8))

    def _rrect(self, x1, y1, x2, y2, r, **kw):
        r   = max(1, r)
        pts = [x1+r,y1, x2-r,y1, x2,y1, x2,y1+r,
               x2,y2-r, x2,y2, x2-r,y2, x1+r,y2,
               x1,y2, x1,y2-r, x1,y1+r, x1,y1]
        return self.create_polygon(pts, smooth=True, **kw)

    @staticmethod
    def _rgb(h):
        h = h.lstrip("#")
        return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))

    @staticmethod
    def _blend(fg, bg, a):
        return tuple(int(fg[i]*a + bg[i]*(1-a)) for i in range(3))

    @staticmethod
    def _hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(*rgb)


# ------------------- Logo Animation Player -------------------
LOGO_NORMAL_FILE = "link_normal.mov"
LOGO_RECOG_FILE  = "link_recog.mov"
LOGO_SIZE        = 140
LOGO_PAD         = 18

class LogoPlayer:
    def __init__(self, root, size=LOGO_SIZE, pad=LOGO_PAD):
        self.root       = root
        self.size       = size
        self.pad        = pad
        self._mode      = 'normal'
        self._frame_idx = 0
        self._job       = None
        self._recog_pending = False

        self.canvas = tk.Canvas(root, width=size, height=size,
                                highlightthickness=0, bg='white', cursor='arrow')
        self.canvas.place(x=pad, y=pad)
        self._img_item = self.canvas.create_image(0, 0, anchor='nw')

        self._fps_normal, pil_normal = self._load_video(LOGO_NORMAL_FILE)
        self._fps_recog,  pil_recog  = self._load_video(LOGO_RECOG_FILE)
        # Keep PIL frames for rescaling, PhotoImages for display
        self._pil_normal = pil_normal
        self._pil_recog  = pil_recog
        self._frames_normal = [ImageTk.PhotoImage(f) for f in pil_normal]
        self._frames_recog  = [ImageTk.PhotoImage(f) for f in pil_recog]
        self._scaled_cache  = {}   # (list_id, idx, size) → PhotoImage

        if not self._frames_normal:
            print(f"Warning: could not load {LOGO_NORMAL_FILE} — logo disabled")
            self.canvas.place_forget()
            return
        self._start_normal()

    def trigger_recog(self):
        if not self._frames_recog:
            return
        self._recog_pending = True
        if self._mode == 'normal':
            self._start_recog()

    def set_size(self, new_size):
        """Resize the logo canvas smoothly without reloading all frames."""
        new_size = max(40, int(new_size))
        if abs(new_size - self.size) < 2:
            return
        self.size = new_size
        self.canvas.config(width=new_size, height=new_size)
        # Invalidate cached scaled frames so they rebuild at new size
        self._scaled_cache = {}

    def _get_frame(self, pil_frames, idx):
        """Return a PhotoImage for the given frame index, scaled to current size."""
        key = (id(pil_frames), idx, self.size)
        if key not in self._scaled_cache:
            pil = pil_frames[idx % len(pil_frames)]
            if pil.size[0] != self.size:
                pil = pil.resize((self.size, self.size), Image.LANCZOS)
            self._scaled_cache[key] = ImageTk.PhotoImage(pil)
        return self._scaled_cache[key]

    def destroy(self):
        if self._job:
            self.root.after_cancel(self._job)
        self.canvas.destroy()

    def _start_normal(self):
        self._mode = 'normal'; self._frame_idx = 0; self._tick()

    def _start_recog(self):
        self._mode = 'recog'; self._frame_idx = 0; self._recog_pending = False; self._tick()

    def _tick(self):
        pil_frames = self._pil_normal if self._mode == 'normal' else self._pil_recog
        fps        = self._fps_normal if self._mode == 'normal' else self._fps_recog
        if not pil_frames:
            return
        delay = max(16, int(1000 / fps))
        idx   = self._frame_idx % len(pil_frames)
        photo = self._get_frame(pil_frames, idx)
        self.canvas.itemconfig(self._img_item, image=photo)
        self._frame_idx += 1
        if self._mode == 'normal':
            self._frame_idx = self._frame_idx % len(pil_frames)
            self._job = self.root.after(delay, self._tick)
        else:
            if self._frame_idx >= len(pil_frames):
                self._start_normal()
            else:
                self._job = self.root.after(delay, self._tick)

    def _load_video(self, filename):
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, filename)
        if not os.path.exists(path):
            print(f"Logo: file not found — {path}"); return 30, []
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Logo: cannot open — {path}"); return 30, []
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 240: fps = 30
        fps = float(fps)
        size = self.size
        mask = Image.new('L', (size, size), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, size-1, size-1), fill=255)
        pil_frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            fh, fw = frame.shape[:2]
            side = min(fh, fw)
            y0 = (fh-side)//2; x0 = (fw-side)//2
            frame = frame[y0:y0+side, x0:x0+side]
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame).convert('RGBA')
            pil.putalpha(mask)
            bg = Image.new('RGBA', (size, size), (255,255,255,255))
            bg.paste(pil, (0,0), pil)
            pil_frames.append(bg.convert('RGB'))
        cap.release()
        print(f"Logo: loaded {len(pil_frames)} frames from {filename} @ {fps:.1f} fps")
        return fps, pil_frames


# ------------------- Hand View Panel -------------------
class HandViewPanel:
    """
    Rounded panel overlaying the window from the right edge.
    Position and size update whenever the window resizes.
    """
    def __init__(self, root):
        self.root        = root
        self._visible    = False
        self._anim       = None
        self._photo_crop = None
        self._photo_lm   = None
        self._win_w      = 1200
        self._win_h      = 700

        self._canvas = tk.Canvas(root, width=PANEL_W, height=700,
                                 bg='white', highlightthickness=0)
        # Hidden off right edge initially
        self._canvas.place(x=1200, y=0)

        self._draw_bg(700)

        PAD = 14
        self._frame = tk.Frame(self._canvas, bg='#F5F5F5')
        self._frame_win = self._canvas.create_window(
            PANEL_W//2, 700//2,
            window=self._frame,
            width=PANEL_W - PAD*2,
            height=700 - PAD*2)
        self._frame.pack_propagate(False)

        tk.Label(self._frame, text="Hand View", bg='#F5F5F5', fg='#444444',
                 font=tkfont.Font(family="Helvetica", size=13, weight="bold")
                 ).pack(pady=(14, 6))

        self.crop_label = tk.Label(self._frame, bg='#E0E0E0')
        self.crop_label.pack(padx=10, pady=(0, 4), fill=tk.X)

        tk.Label(self._frame, text="camera crop", bg='#F5F5F5', fg='#BBBBBB',
                 font=tkfont.Font(family="Helvetica", size=9)).pack()

        tk.Frame(self._frame, bg='#DDDDDD', height=1).pack(fill=tk.X, padx=14, pady=6)

        self.lm_label = tk.Label(self._frame, bg='#EBEBEB')
        self.lm_label.pack(padx=10, pady=(0, 4), fill=tk.X)

        tk.Label(self._frame, text="rotation-adjusted landmarks",
                 bg='#F5F5F5', fg='#BBBBBB',
                 font=tkfont.Font(family="Helvetica", size=9)).pack()

    def on_resize(self, win_w, win_h):
        """Called whenever window resizes. Repositions panel accordingly."""
        self._win_w = win_w
        self._win_h = win_h
        panel_h = int(win_h * PANEL_H_RATIO)
        panel_y = (win_h - panel_h) // 2

        self._canvas.config(width=PANEL_W, height=panel_h)
        self._draw_bg(panel_h)
        PAD = 14
        self._canvas.coords(self._frame_win, PANEL_W//2, panel_h//2)
        self._canvas.itemconfig(self._frame_win,
                                width=PANEL_W - PAD*2,
                                height=panel_h - PAD*2)

        if self._visible:
            panel_x = win_w - PANEL_W - PANEL_PAD
            self._canvas.place(x=panel_x, y=panel_y)
        else:
            self._canvas.place(x=win_w, y=panel_y)   # keep off-screen to the right

    def _draw_bg(self, h):
        self._canvas.delete("bg")
        w, r = PANEL_W, PANEL_R
        for i, shade in enumerate(['#E0E0E0', '#D8D8D8', '#CECECE']):
            self._rrect(i+2, i+2, w-2+i, h-2+i, r, fill=shade, outline='', tag='bg')
        self._rrect(0, 0, w-6, h-6, r, fill='#F6F6F6', outline='#E2E2E2', width=1, tag='bg')

    def _rrect(self, x1, y1, x2, y2, r, tag='', **kw):
        pts = [x1+r,y1, x2-r,y1, x2,y1, x2,y1+r,
               x2,y2-r, x2,y2, x2-r,y2, x1+r,y2,
               x1,y2, x1,y2-r, x1,y1+r, x1,y1]
        return self._canvas.create_polygon(pts, smooth=True, tags=tag, **kw)

    def show(self):
        if self._visible: return
        self._visible = True
        win_h   = self._win_h
        panel_h = int(win_h * PANEL_H_RATIO)
        panel_y = (win_h - panel_h) // 2
        self._animate(start=self._win_w, end=self._win_w - PANEL_W - PANEL_PAD,
                      panel_y=panel_y)

    def hide(self):
        if not self._visible: return
        self._visible = False
        win_h   = self._win_h
        panel_h = int(win_h * PANEL_H_RATIO)
        panel_y = (win_h - panel_h) // 2
        self._animate(start=self._win_w - PANEL_W - PANEL_PAD, end=self._win_w,
                      panel_y=panel_y)

    def update(self, crop_bgr, landmark_bgr):
        if not self._visible: return
        dw = PANEL_W - 60
        if crop_bgr is not None and crop_bgr.size > 0:
            try:
                rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                pil    = Image.fromarray(rgb)
                ch, cw = crop_bgr.shape[:2]
                asp    = ch / max(cw, 1)
                nw     = dw
                nh     = min(int(nw * asp), 180)
                nw     = int(nh / asp) if nh == 180 else nw
                pil    = pil.resize((nw, nh), Image.LANCZOS)
                self._photo_crop = ImageTk.PhotoImage(pil)
                self.crop_label.config(image=self._photo_crop, width=nw, height=nh)
            except Exception: pass

        if landmark_bgr is not None:
            try:
                rgb  = cv2.cvtColor(landmark_bgr, cv2.COLOR_BGR2RGB)
                lm_s = PANEL_W - 80
                pil  = Image.fromarray(rgb).resize((lm_s, lm_s), Image.LANCZOS)
                self._photo_lm = ImageTk.PhotoImage(pil)
                self.lm_label.config(image=self._photo_lm, width=lm_s, height=lm_s)
            except Exception: pass

    @property
    def visible(self): return self._visible

    def _animate(self, start, end, panel_y, steps=12):
        if self._anim: self.root.after_cancel(self._anim)
        def _tick(i):
            t = i / steps
            t = 1 - (1 - t) ** 3
            x = int(start + (end - start) * t)
            self._canvas.place(x=x, y=panel_y)
            if i < steps:
                self._anim = self.root.after(14, lambda: _tick(i+1))
            else:
                self._anim = None
        _tick(0)


# ------------------- Circular Hand Toggle Button -------------------
class CircularHandButton(tk.Canvas):
    SIZE = 54
    PAD  = 16

    def __init__(self, root, on_toggle):
        super().__init__(root, width=self.SIZE, height=self.SIZE,
                         bg='white', highlightthickness=0, cursor='hand2')
        self._on_toggle = on_toggle
        self._active    = False
        self._hovered   = False
        self._pressed   = False
        self._win_w     = 1200
        self._reposition(1200)
        self._draw()
        self.bind('<ButtonPress-1>',   self._press)
        self.bind('<ButtonRelease-1>', self._release)
        self.bind('<Enter>',  lambda e: self._set_hover(True))
        self.bind('<Leave>',  lambda e: (self._set_hover(False),
                                          setattr(self, '_pressed', False),
                                          self._draw()))

    def on_resize(self, win_w, win_h):
        self._win_w = win_w
        self._reposition(win_w)

    def _reposition(self, win_w):
        self.place(x=win_w - self.SIZE - self.PAD, y=self.PAD)

    def _draw(self):
        self.delete("all")
        s = self.SIZE
        if self._pressed:
            bg, ol, fg = '#CCCCCC', '#AAAAAA', '#222222'
        elif self._active:
            bg, ol, fg = '#1A1A1A', '#1A1A1A', '#FFFFFF'
        elif self._hovered:
            bg, ol, fg = '#F0F0F0', '#CCCCCC', '#222222'
        else:
            bg, ol, fg = '#FFFFFF', '#DDDDDD', '#333333'
        self.create_oval(2, 2, s-2, s-2, fill=bg, outline=ol, width=1.5)
        self.create_text(s//2, s//2, text="✋", font=("Helvetica", 22), fill=fg)

    def _press(self, _):
        self._pressed = True; self._draw()

    def _release(self, _):
        self._pressed = False
        self._active  = not self._active
        self._draw()
        self._on_toggle(self._active)

    def _set_hover(self, val):
        self._hovered = val; self._draw()


# ------------------- UI -------------------
class ASLTranslationUI:
    _GLOW    = ["#00EE55","#00DD44","#00CC3A","#00AA2D",
                "#008820","#006616","#00440F","#000000"]
    _GLOW_MS = 30
    PILL_W   = 40
    PILL_GAP = 6
    PILL_PAD = 14

    def __init__(self, root):
        self.root        = root
        self.root.title("ASL Live Translation")
        self.root.geometry("1200x700")
        self.root.resizable(True, True)
        self.root.configure(bg='white')
        self._panel_open  = False
        self._win_w       = 1200
        self._win_h       = 700
        self._shift_job   = None
        self._cur_shift   = 0    # current right-padding applied to content

        self.top_frame = tk.Frame(root, bg='white')
        self.top_frame.pack(fill=tk.BOTH, expand=True)

        self.word_label = tk.Label(
            self.top_frame, text="", bg='white', fg='black',
            font=tkfont.Font(family="Helvetica", size=48, weight="bold"),
            wraplength=750
        )
        self.word_label.pack(pady=(50, 4))

        self._lf = tk.Frame(self.top_frame, bg='white')
        self._lf.pack(pady=(0, 2))

        self.big_letter = tk.Label(
            self._lf, text="", bg='white', fg='black',
            font=tkfont.Font(family="Helvetica", size=80, weight="bold"),
            width=2, anchor='center'
        )
        self.big_letter.pack()

        self.confirmed_label = tk.Label(
            self._lf, text="", bg='white', fg='#999999',
            font=tkfont.Font(family="Helvetica", size=13), anchor='center'
        )
        self.confirmed_label.pack()

        self._cf = tk.Frame(self.top_frame, bg='white')
        self._cf.pack(pady=(8, 2))
        self.capsule = CapsuleLetterDisplay(self._cf, max_visible=14)
        self.capsule.pack()

        self.bottom_frame = tk.Frame(root, bg='#808080', height=180)
        self.bottom_frame.pack(fill=tk.BOTH, expand=True)
        self.bottom_frame.pack_propagate(False)

        self.sentence_label = tk.Label(
            self.bottom_frame, text="", bg='#808080', fg='white',
            font=tkfont.Font(family="Helvetica", size=20),
            wraplength=750, justify=tk.LEFT
        )
        self.sentence_label.pack(pady=40, padx=20)

        self._prev_big  = ""
        self._prev_conf = ""
        self._glow_job  = None

        self.logo       = LogoPlayer(root)
        self.hand_panel = HandViewPanel(root)
        self.hand_btn   = CircularHandButton(root, self.toggle_panel)

        root.bind('<Configure>', self._on_configure)

        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ── Resize handler ────────────────────────────────────────────────────────

    def _on_configure(self, event):
        if event.widget != self.root:
            return
        w, h = event.width, event.height
        if w == self._win_w and h == self._win_h:
            return
        self._win_w = w
        self._win_h = h
        self._relayout(w, h, self._cur_shift)

    def _relayout(self, w, h, shift):
        """Recalculate all proportional layout for new window size + current shift."""
        avail = max(200, w - shift)

        wl = int(avail * 0.62)
        self.word_label.config(wraplength=wl)
        self.sentence_label.config(wraplength=wl)

        max_pills = max(1, (avail - 2*self.PILL_PAD + self.PILL_GAP) //
                           (self.PILL_W + self.PILL_GAP))
        max_pills = min(max_pills, 14)
        self.capsule.set_max_visible(max_pills)

        self.hand_btn.on_resize(w, h)
        self.hand_panel.on_resize(w, h)

    # ── Shift animation ───────────────────────────────────────────────────────

    def _animate_shift(self, target, steps=14):
        """Smoothly push content left (positive target) or restore (0)."""
        if self._shift_job:
            self.root.after_cancel(self._shift_job)
        start = self._cur_shift

        def _tick(i):
            t   = i / steps
            t   = 1 - (1 - t) ** 3           # ease-out cubic
            pad = int(start + (target - start) * t)
            self._cur_shift = pad
            # Apply right padding to all content elements → pushes them left
            self.word_label.pack_configure(padx=(0, pad))
            self._lf.pack_configure(padx=(0, pad))
            self._cf.pack_configure(padx=(0, pad))
            self.sentence_label.pack_configure(padx=(20, max(20, pad)))
            # Recalculate wraplength and capsule width for this shift amount
            self._relayout(self._win_w, self._win_h, pad)
            # Scale logo proportionally
            logo_size = max(60, int(LOGO_SIZE * (1.0 - 0.35 * (pad / (PANEL_W + PANEL_PAD*2)))))
            self.logo.set_size(logo_size)
            if i < steps:
                self._shift_job = self.root.after(14, lambda: _tick(i + 1))
            else:
                self._shift_job = None
        _tick(0)

    # ── Panel toggle ─────────────────────────────────────────────────────────

    def toggle_panel(self, open_panel: bool):
        self._panel_open = open_panel
        if open_panel:
            self.hand_panel.show()
            self._animate_shift(PANEL_W + PANEL_PAD * 2)
        else:
            self.hand_panel.hide()
            self._animate_shift(0)

    # ── Hand panel update ────────────────────────────────────────────────────

    def update_hand_panel(self, crop_bgr, landmark_bgr):
        if self._panel_open:
            self.hand_panel.update(crop_bgr, landmark_bgr)

    # ── Green glow ───────────────────────────────────────────────────────────

    def _do_glow(self, step=0):
        if step < len(self._GLOW):
            self.big_letter.config(fg=self._GLOW[step])
            self._glow_job = self.root.after(
                self._GLOW_MS, lambda: self._do_glow(step + 1))
        else:
            self._glow_job = None

    def _trigger_glow(self):
        if self._glow_job:
            self.root.after_cancel(self._glow_job)
        self._do_glow(0)

    # ── Main update ──────────────────────────────────────────────────────────

    def update_display(self, last_word, last_letter, full_sentence,
                       raw_letters_list, new_letter_confirmed=False):
        self.word_label.config(text=last_word)
        self.sentence_label.config(text=full_sentence)

        if last_letter != self._prev_big:
            self._prev_big = last_letter
            self.big_letter.config(text=last_letter.upper() if last_letter else "")

        if new_letter_confirmed:
            self._trigger_glow()

        conf = f"confirmed: {last_letter.upper()}" if last_letter else ""
        if conf != self._prev_conf:
            self._prev_conf = conf
            self.confirmed_label.config(text=conf)

        self.capsule.set_letters(raw_letters_list)

    def trigger_recog(self):
        self.logo.trigger_recog()

    def on_closing(self):
        self.running = False
        self.logo.destroy()
        self.root.destroy()


# ------------------- Main loop -------------------
def main():
    global cap, current_letter, sentence_text, all_raw_letters, current_letters
    global is_recording, last_confident_time, last_letter_time, prediction_history
    cap = None

    root = tk.Tk()
    ui   = ASLTranslationUI(root)

    last_word            = ""
    current_landmark_img = None

    root.deiconify()
    root.attributes('-topmost', False)

    cap = cv2.VideoCapture(0)
    print("Live ASL Translation — close window or press Q to quit")

    def update_frame():
        global current_letter, sentence_text, all_raw_letters, current_letters
        global is_recording, last_confident_time, last_letter_time, prediction_history

        nonlocal last_word, current_landmark_img

        if not ui.running:
            cap.release()
            cv2.destroyAllWindows()
            pygame.mixer.quit()
            return

        ret, frame = cap.read()
        if not ret:
            root.after(10, update_frame)
            return

        frame                = cv2.flip(frame, 1)
        hand_landmarks       = get_hand_landmarks(frame)
        landmark_image       = calculate_slope_and_adjust(hand_landmarks, frame.shape) \
                               if hand_landmarks else None
        current_landmark_img = landmark_image
        current_time         = time.time()
        new_confirmed        = False

        if ui._panel_open:
            hand_crop = get_hand_crop_with_landmarks(frame, hand_landmarks)
            ui.update_hand_panel(hand_crop, current_landmark_img)

        if landmark_image is not None:
            processed = preprocess_image(landmark_image)
            if processed is not None:
                pred       = tflite_predict(processed)
                pred_idx   = int(np.argmax(pred))
                confidence = float(np.max(pred))
                if confidence > CONFIDENCE_THRESHOLD:
                    prediction_history.append(
                        class_names[pred_idx] if pred_idx < len(class_names) else "")
                    last_confident_time = current_time
        else:
            prediction_history.clear()

        committed = process_letter(prediction_history, current_time)
        if committed:
            new_confirmed = True

        if last_confident_time and (current_time - last_confident_time > WORD_TIMEOUT):
            if is_recording:
                is_recording = False
                combined     = ''.join(all_raw_letters)
                if combined:
                    print(f"Word: {combined} → querying LLM…")
                    ui.trigger_recog()
                    snap = combined
                    def qt(s=snap):
                        global sentence_text
                        nonlocal last_word
                        result = query_hf_llm(s)
                        if result:
                            sentence_text = result
                            last_word     = result.split()[-1] if result.split() else ""
                            print("LLM →", sentence_text)
                    threading.Thread(target=qt, daemon=True).start()
                    def _clear():
                        global all_raw_letters, current_letters
                        all_raw_letters.clear(); current_letters.clear()
                    root.after(int(CAPSULE_LINGER * 1000), _clear)
                last_letter_time = None

        if last_confident_time and (current_time - last_confident_time > SENTENCE_TIMEOUT):
            combined = ''.join(all_raw_letters)
            if combined:
                print(f"Sentence timeout: {combined} → querying LLM…")
                ui.trigger_recog()
                snap = combined
                def qts(s=snap):
                    global sentence_text
                    nonlocal last_word
                    result = query_hf_llm(s)
                    if result:
                        sentence_text = result
                        last_word     = result.split()[-1] if result.split() else ""
                        print("LLM (sentence) →", sentence_text)
                threading.Thread(target=qts, daemon=True).start()
                all_raw_letters.clear(); current_letters.clear()

        if last_confident_time and (current_time - last_confident_time > CLEAR_TIMEOUT):
            all_raw_letters.clear(); current_letters.clear()
            sentence_text = ""; current_letter = ""; last_word = ""
            is_recording = False; last_confident_time = None
            prediction_history.clear()
            print("State cleared (inactivity)")

        ui.update_display(
            last_word, current_letter, sentence_text,
            list(all_raw_letters), new_letter_confirmed=new_confirmed
        )

        root.after(10, update_frame)

    update_frame()
    root.mainloop()

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()


if __name__ == "__main__":
    main()
