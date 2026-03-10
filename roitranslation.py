# ASL Live Translation with Hugging Face Inference API + UI

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
IMG_SIZE                = 200
HISTORY_LENGTH          = 16
FRAMES_THRESHOLD        = 4
CONFIDENCE_THRESHOLD    = 0.70
TOLERANCE_THRESHOLD     = 0.50
WORD_TIMEOUT            = 0.35
SENTENCE_TIMEOUT        = 5.0
CLEAR_TIMEOUT           = 9.0
DOUBLE_LETTER_TIME      = 0.7
DOUBLE_LETTER_STABILITY = 0.92
HF_TIMEOUT              = 30
CAPSULE_LINGER          = 2.0   # seconds pills linger after word clears

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
    model = tf.keras.models.load_model('v1exhaustiveroiasl_landmark_model.keras')
    with open('class_names.txt', 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    print("Model and class names loaded. Classes:", class_names)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# ------------------- Startup Animation (commented out for testing) -------------------
# def play_startup_animation(root, on_finish, video_filename="Startup copy.mp4", fade_frames=15):
#     base_dir   = os.path.dirname(os.path.abspath(__file__))
#     video_path = os.path.join(base_dir, video_filename)
#     if not os.path.exists(video_path):
#         on_finish(); return
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         on_finish(); return
#     fps = cap.get(cv2.CAP_PROP_FPS) or 60
#     frame_delay = int(1000 / fps)
#     orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     root.deiconify(); root.update()
#     ww, wh = 1200, 700
#     ar = orig_w / orig_h
#     if ww / wh > ar: dh, dw = wh, int(wh * ar)
#     else:            dw, dh = ww, int(ww / ar)
#     canvas = tk.Canvas(root, width=ww, height=wh, highlightthickness=0, bg='white')
#     canvas.place(x=0, y=0, relwidth=1, relheight=1)
#     img_c = canvas.create_image(ww//2, wh//2, anchor=tk.CENTER)
#     frames = []
#     while True:
#         ret, frm = cap.read()
#         if not ret: break
#         frm = cv2.resize(frm, (dw, dh), interpolation=cv2.INTER_AREA)
#         frames.append(Image.fromarray(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)))
#     cap.release()
#     if not frames: canvas.destroy(); on_finish(); return
#     photos = [ImageTk.PhotoImage(f) for f in frames]
#     canvas._refs = photos; canvas._frms = frames
#     def show(i=0):
#         if i < len(photos):
#             canvas.itemconfig(img_c, image=photos[i])
#             root.after(frame_delay, lambda: show(i+1))
#         else: fade(0)
#     def fade(s):
#         if s >= fade_frames:
#             def cleanup(): canvas.destroy(); canvas._refs=None; on_finish()
#             root.after(50, cleanup); return
#         a = s / fade_frames
#         last = np.array(frames[-1]).astype(np.float32)
#         blended = (last*(1-a) + np.ones_like(last)*255*a).astype(np.uint8)
#         img = ImageTk.PhotoImage(Image.fromarray(blended))
#         canvas.itemconfig(img_c, image=img); canvas._fade=img
#         root.after(25, lambda: fade(s+1))
#     show()

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
def get_hand_landmarks(image):
    rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

# ------------------- Original sequence processing (restored exactly) -------------------
prediction_history  = deque(maxlen=HISTORY_LENGTH)
current_letters     = []
all_raw_letters     = []
is_recording        = False
last_confident_time = None
last_letter_time    = None
current_letter      = ""
sentence_text       = ""

def process_letter(prediction_history, current_time):
    """
    Original letter-commit logic restored verbatim.
    Returns the newly committed letter string, or None if nothing committed.
    """
    global current_letters, all_raw_letters, is_recording
    global last_confident_time, last_letter_time, current_letter

    if not prediction_history or len(prediction_history) < FRAMES_THRESHOLD:
        return None

    most_common = max(set(prediction_history), key=prediction_history.count)
    confidence_ratio = prediction_history.count(most_common) / len(prediction_history)
    unique_preds = len(set(prediction_history))
    is_chaotic = confidence_ratio < 0.5 or (unique_preds / HISTORY_LENGTH > 0.5)

    if confidence_ratio >= TOLERANCE_THRESHOLD and most_common in class_names:
        if not is_recording:
            is_recording = True
            current_letters = []

        last_letter = current_letters[-1] if current_letters else None

        if last_letter != most_common:
            current_letters.append(most_common)
            all_raw_letters.append(most_common)
            current_letter   = most_common
            last_letter_time = current_time
            try: boop_sound.play()
            except: pass
            last_confident_time = current_time
            return most_common

        else:
            if last_letter_time and (current_time - last_letter_time >= DOUBLE_LETTER_TIME):
                stability = prediction_history.count(most_common) / len(prediction_history)
                if not is_chaotic and stability >= DOUBLE_LETTER_STABILITY:
                    current_letters.append(most_common)
                    all_raw_letters.append(most_common)
                    current_letter   = most_common
                    last_letter_time = current_time
                    try: boop_sound.play()
                    except: pass
                    last_confident_time = current_time
                    return most_common

        last_confident_time = current_time

    return None

# ------------------- Capsule spring physics -------------------
def _spring(pos, vel, target, k, d):
    f   = (target - pos) * k
    vel = vel * d + f
    return pos + vel, vel


class Pill:
    """
    One committed letter pill.
    Enters by springing in from above (scale 0→1, y above→0).
    Exits by shrinking and dropping. Alpha always 1.0 while alive.
    """
    W = 40
    H = 40

    def __init__(self, letter, tx):
        self.letter  = letter
        self.tx      = float(tx)
        self.x       = float(tx)
        self.y       = -float(self.H) * 1.8   # above capsule
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
        return (abs(self.ty - self.y) < 0.6 and
                abs(self.tx - self.x) < 0.6 and
                abs((0.0 if self.exiting else 1.0) - self.scale) < 0.02)


class CapsuleLetterDisplay(tk.Canvas):
    PILL_W      = 40
    PILL_H      = 40
    GAP         = 6
    PAD_X       = 14
    MAX_VISIBLE = 14
    CAP_H       = 60

    BG       = "#F2F2F2"
    OUTLINE  = "#CCCCCC"
    C_NORM   = "#E0E0E0"
    C_LATEST = "#1A1A1A"
    T_NORM   = "#333333"
    T_LATEST = "#FFFFFF"

    def __init__(self, parent, **kw):
        self._letters = []
        self._pills   = []
        self._exiting = []
        self._running = False
        w = self._W()
        super().__init__(parent, width=w, height=self.CAP_H,
                         bg=parent["bg"], highlightthickness=0, **kw)
        self._redraw()

    # ── public ──────────────────────────────────────────────────────────────

    def set_letters(self, letters: list):
        new = list(letters)
        if new == self._letters:
            return

        visible = new[-self.MAX_VISIBLE:]

        if not new:
            for p in self._pills:
                p.exit()
            self._exiting.extend(self._pills)
            self._pills = []
        else:
            # retire pills that scrolled off the left
            drop = len(self._pills) - len(visible)
            if drop > 0:
                for p in self._pills[:drop]:
                    p.exit()
                self._exiting.extend(self._pills[:drop])
                self._pills = self._pills[drop:]

            # slide remaining pills to their new x targets
            for i, p in enumerate(self._pills):
                p.tx = float(self._X(i))

            # spawn new pills
            while len(self._pills) < len(visible):
                idx = len(self._pills)
                self._pills.append(Pill(visible[idx], self._X(idx)))

        self._letters = new
        self._kick()

    # ── internals ───────────────────────────────────────────────────────────

    def _W(self):
        return 2*self.PAD_X + self.MAX_VISIBLE*self.PILL_W + (self.MAX_VISIBLE-1)*self.GAP

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
        w, h = self._W(), self.CAP_H
        cy   = h // 2
        n    = len(self._pills)

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
LOGO_NORMAL_FILE = "link_normal.mov"   # loops continuously
LOGO_RECOG_FILE  = "link_recog.mov"    # plays once on query, then returns to normal
LOGO_SIZE        = 140                 # diameter of the circle crop (px)
LOGO_PAD         = 18                  # distance from top-left corner of window


class LogoPlayer:
    """
    Circular logo animation widget.
    - Plays link_normal.mp4 on a seamless loop when idle.
    - On trigger_recog(), plays link_recog.mp4 once, then resumes normal loop.
    - Rendered as a circle-cropped Canvas placed absolutely in the top-left.
    - Runs its own frame-tick via root.after so it never blocks the main loop.
    """

    def __init__(self, root, size=LOGO_SIZE, pad=LOGO_PAD):
        self.root   = root
        self.size   = size
        self.pad    = pad
        self._mode  = 'normal'      # 'normal' | 'recog'
        self._frame_idx = 0
        self._job   = None
        self._recog_pending = False

        # Canvas placed absolutely over the window
        self.canvas = tk.Canvas(root, width=size, height=size,
                                highlightthickness=0, bg='white',
                                cursor='arrow')
        self.canvas.place(x=pad, y=pad)

        # Create circle clip mask item (filled white outside)
        self._img_item = self.canvas.create_image(0, 0, anchor='nw')

        # Pre-load both video sets as lists of circular PhotoImage frames
        self._fps_normal,  pil_normal = self._load_video(LOGO_NORMAL_FILE)
        self._fps_recog,   pil_recog  = self._load_video(LOGO_RECOG_FILE)

        # Convert PIL → PhotoImage and keep hard references (prevents GC)
        self._frames_normal = [ImageTk.PhotoImage(f) for f in pil_normal]
        self._frames_recog  = [ImageTk.PhotoImage(f) for f in pil_recog]

        if not self._frames_normal:
            print(f"Warning: could not load {LOGO_NORMAL_FILE} — logo disabled")
            self.canvas.place_forget()
            return

        self._start_normal()

    # ── public ──────────────────────────────────────────────────────────────

    def trigger_recog(self):
        """Call when an LLM query fires — switches to recog animation."""
        if not self._frames_recog:
            return
        self._recog_pending = True
        if self._mode == 'normal':
            self._start_recog()

    def destroy(self):
        if self._job:
            self.root.after_cancel(self._job)
        self.canvas.destroy()

    # ── internals ───────────────────────────────────────────────────────────

    def _start_normal(self):
        self._mode      = 'normal'
        self._frame_idx = 0
        self._tick()

    def _start_recog(self):
        self._mode           = 'recog'
        self._frame_idx      = 0
        self._recog_pending  = False
        self._tick()

    def _tick(self):
        if self._mode == 'normal':
            frames = self._frames_normal
            fps    = self._fps_normal
        else:
            frames = self._frames_recog
            fps    = self._fps_recog

        if not frames:
            return

        delay = max(16, int(1000 / fps))

        # Show current frame
        frame = frames[self._frame_idx % len(frames)]
        self.canvas.itemconfig(self._img_item, image=frame)

        self._frame_idx += 1

        if self._mode == 'normal':
            # Seamless loop
            self._frame_idx = self._frame_idx % len(frames)
            self._job = self.root.after(delay, self._tick)

        else:  # recog
            if self._frame_idx >= len(frames):
                # Recog clip finished → back to normal
                self._start_normal()
            else:
                self._job = self.root.after(delay, self._tick)

    def _load_video(self, filename):
        """
        Load all frames from a video file, square-crop and circle-mask them.
        Returns (fps, [PIL.Image, ...]) or (30, []) on failure.
        Stores PIL Images — PhotoImage conversion happens in _tick to keep
        references alive and avoid garbage collection issues.
        """
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, filename)
        if not os.path.exists(path):
            print(f"Logo: file not found — {path}")
            return 30, []

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Logo: cannot open — {path}")
            return 30, []

        # Read FPS directly here while cap is open
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 240:
            fps = 30
        fps = float(fps)

        size = self.size

        # Build circular alpha mask once
        from PIL import ImageDraw
        mask = Image.new('L', (size, size), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, size - 1, size - 1), fill=255)

        pil_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fh, fw = frame.shape[:2]
            # Square-crop from center
            side = min(fh, fw)
            y0 = (fh - side) // 2
            x0 = (fw - side) // 2
            frame = frame[y0:y0+side, x0:x0+side]
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pil = Image.fromarray(frame).convert('RGBA')
            pil.putalpha(mask)
            bg = Image.new('RGBA', (size, size), (255, 255, 255, 255))
            bg.paste(pil, (0, 0), pil)
            pil_frames.append(bg.convert('RGB'))

        cap.release()
        print(f"Logo: loaded {len(pil_frames)} frames from {filename} @ {fps:.1f} fps")
        return fps, pil_frames


# ------------------- UI -------------------
class ASLTranslationUI:
    # Green glow: bright flash → decay back to black
    _GLOW = ["#00EE55","#00DD44","#00CC3A","#00AA2D",
             "#008820","#006616","#00440F","#000000"]
    _GLOW_MS = 30   # ms per step

    def __init__(self, root):
        self.root = root
        self.root.title("ASL Live Translation")
        self.root.geometry("1200x700")
        self.root.configure(bg='white')

        self.top_frame = tk.Frame(root, bg='white')
        self.top_frame.pack(fill=tk.BOTH, expand=True)

        # Translated word
        self.word_label = tk.Label(
            self.top_frame, text="", bg='white', fg='black',
            font=tkfont.Font(family="Helvetica", size=48, weight="bold"),
            wraplength=750
        )
        self.word_label.pack(pady=(50, 4))

        # Big live letter
        lf = tk.Frame(self.top_frame, bg='white')
        lf.pack(pady=(0, 2))

        self.big_letter = tk.Label(
            lf, text="", bg='white', fg='black',
            font=tkfont.Font(family="Helvetica", size=80, weight="bold"),
            width=2, anchor='center'
        )
        self.big_letter.pack()

        # Small "confirmed" subtitle
        self.confirmed_label = tk.Label(
            lf, text="", bg='white', fg='#999999',
            font=tkfont.Font(family="Helvetica", size=13),
            anchor='center'
        )
        self.confirmed_label.pack()

        # Capsule
        cf = tk.Frame(self.top_frame, bg='white')
        cf.pack(pady=8)
        self.capsule = CapsuleLetterDisplay(cf)
        self.capsule.pack()

        # Grey sentence bar
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

        # Logo animation (placed absolutely, top-left)
        self.logo = LogoPlayer(root)

        self.running = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

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

        # Big letter tracks current_letter (last confirmed)
        if last_letter != self._prev_big:
            self._prev_big = last_letter
            self.big_letter.config(
                text=last_letter.upper() if last_letter else "")

        # Glow fires exactly when a new letter is confirmed this frame
        if new_letter_confirmed:
            self._trigger_glow()

        # Small subtitle
        conf = f"confirmed: {last_letter.upper()}" if last_letter else ""
        if conf != self._prev_conf:
            self._prev_conf = conf
            self.confirmed_label.config(text=conf)

        self.capsule.set_letters(raw_letters_list)

    def trigger_recog(self):
        """Fire the recog logo animation when an LLM query is sent."""
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

    # Startup animation skipped for testing — uncomment to re-enable:
    # play_startup_animation(root, on_finish=start_main_app,
    #                        video_filename="Startup copy.mp4")

    root.deiconify()
    root.attributes('-topmost', False)

    cap = cv2.VideoCapture(0)
    print("Live ASL Translation — close window or press Q to quit")

    def update_frame():
        global current_letter, sentence_text, all_raw_letters, current_letters
        global is_recording, last_confident_time, last_letter_time, prediction_history

        nonlocal last_word, current_landmark_img

        if not ui.running:
            if cap: cap.release()
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

        if landmark_image is not None:
            processed = preprocess_image(landmark_image)
            if processed is not None:
                pred       = model.predict(processed, verbose=0)
                pred_idx   = int(np.argmax(pred))
                confidence = float(np.max(pred))
                if confidence > CONFIDENCE_THRESHOLD:
                    prediction_history.append(
                        class_names[pred_idx] if pred_idx < len(class_names) else "")
                    last_confident_time = current_time
        else:
            prediction_history.clear()

        # ── Original letter-commit logic ──────────────────────────────────────
        committed = process_letter(prediction_history, current_time)
        if committed:
            new_confirmed = True

        # ── Word timeout ──────────────────────────────────────────────────────
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
                    # let pills linger before clearing
                    def _clear():
                        global all_raw_letters, current_letters
                        all_raw_letters.clear()
                        current_letters.clear()
                    root.after(int(CAPSULE_LINGER * 1000), _clear)
                last_letter_time = None

        # ── Sentence timeout ──────────────────────────────────────────────────
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
                all_raw_letters.clear()
                current_letters.clear()

        # ── Full clear ────────────────────────────────────────────────────────
        if last_confident_time and (current_time - last_confident_time > CLEAR_TIMEOUT):
            all_raw_letters.clear()
            current_letters.clear()
            sentence_text       = ""
            current_letter      = ""
            last_word           = ""
            is_recording        = False
            last_confident_time = None
            prediction_history.clear()
            print("State cleared (inactivity)")

        ui.update_display(
            last_word,
            current_letter,
            sentence_text,
            list(all_raw_letters),
            new_letter_confirmed=new_confirmed
        )

        # Landmark debug window
        disp = current_landmark_img if current_landmark_img is not None \
               else np.ones((500, 500, 3), dtype=np.uint8) * 255
        cv2.imshow('Landmark Image (Rotation Adjusted)',
                   cv2.resize(disp, (500, 500), interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            ui.on_closing()
            return

        root.after(10, update_frame)

    update_frame()
    root.mainloop()

    if cap: cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()


if __name__ == "__main__":
    main()
