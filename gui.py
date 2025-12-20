import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# --- Asset Loading ---
try:
    ASSET_PATH = 'roiux'
    live_button_img = cv2.imread(os.path.join(ASSET_PATH, 'live.png'), cv2.IMREAD_UNCHANGED)
    standby_button_img = cv2.imread(os.path.join(ASSET_PATH, 'standby.png'), cv2.IMREAD_UNCHANGED)
    settings_icon_img = cv2.imread(os.path.join(ASSET_PATH, 'settings.png'), cv2.IMREAD_UNCHANGED)
    back_button_img = cv2.imread(os.path.join(ASSET_PATH, 'preceeding.png'), cv2.IMREAD_UNCHANGED)
    assert all(img is not None for img in [live_button_img, standby_button_img, settings_icon_img, back_button_img])
    print("UI assets loaded successfully from 'roiux' folder.")
except (cv2.error, AssertionError):
    print("FATAL ERROR: Could not load UI assets from the 'roiux' folder.")
    print("Please ensure all .png assets exist in a subfolder named 'roiux'.")
    exit()

# --- Font Setup ---
FONT_PATH = None
try:
    font_path_mac = "/System/Library/Fonts/Helvetica.ttc"
    font_path_local = "Helvetica.ttf"
    if os.path.exists(font_path_mac):
        FONT_PATH = font_path_mac
    elif os.path.exists(font_path_local):
        FONT_PATH = font_path_local
    else:
        raise FileNotFoundError
    print(f"Using font: {FONT_PATH}")
except Exception:
    print("WARNING: Helvetica font not found. Falling back to OpenCV.")
    FONT_PATH = None

# --- Configuration & Styling ---
SCALE_FACTOR = 4
BASE_WIDTH = 450
ASPECT_RATIO = 656 / 524
WIDTH, HEIGHT = BASE_WIDTH, int(BASE_WIDTH * ASPECT_RATIO)
S_WIDTH, S_HEIGHT = WIDTH * SCALE_FACTOR, HEIGHT * SCALE_FACTOR

# Colors
CV_COLOR_DISPLAY_BACKGROUND = (255, 255, 255)
CV_COLOR_UI_BACKGROUND = (90, 90, 90)
CV_COLOR_WHITE = (245, 245, 245)
CV_COLOR_GREY = (149, 149, 149)
CV_COLOR_SHADOW = (20, 20, 20)
PIL_COLOR_TEXT_DARK = (20, 20, 20)
PIL_COLOR_TEXT_LIGHT = (235, 235, 235)
PIL_COLOR_TEXT_SUBTLE = (100, 100, 100)

# UI Geometry
FRAME_RADIUS, FRAME_PADDING = 80, 12.5
WINDOW_RADIUS = FRAME_RADIUS - FRAME_PADDING
WORD_MARGIN, SUBTITLE_RATIO = 9, 0.20

# --- State Variables ---
app_state = 'STANDBY'
animation_speed = 0.8 # Snappier animation

is_play_stop_pressed = False
is_settings_pressed = False
is_back_pressed = False

white_window_y, white_window_height, white_window_x_padding = 0, 0, 0
grey_window_y, grey_window_height, grey_window_x_padding = 0, 0, 0
settings_opacity, settings_target_opacity = 0.0, 0.0
button_size_mult, target_button_size_mult = 1.0, 1.0
button_opacity, target_button_opacity = 1.0, 1.0
settings_icon_opacity, target_settings_icon_opacity = 1.0, 1.0
back_button_size_mult, target_back_button_size_mult = 1.0, 1.0
back_button_opacity, target_back_button_opacity = 1.0, 1.0

# --- Helper Functions ---

def draw_rounded_rectangle(image, top_left, bottom_right, color, radius, thickness=-1):
    x1, y1 = top_left
    x2, y2 = bottom_right
    h = y2 - y1
    radius = min(radius, h // 2) if h > 0 else 0
    cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
    cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, thickness)

def lerp(start, end, factor):
    return start + factor * (end - start)

def get_font(size, font_path=None):
    try:
        if font_path:
            return ImageFont.truetype(font_path, size)
        else:
            return ImageFont.load_default()
    except IOError:
        return ImageFont.load_default()

def get_dynamic_title_sizes(main_text, sub_text, sub_ratio, target_w, target_h, font_path):
    main_size = 400
    while main_size > 10:
        sub_size = int(main_size * sub_ratio)
        main_font = get_font(main_size, font_path)
        sub_font = get_font(sub_size, font_path)

        main_bbox = main_font.getbbox(main_text)
        sub_bbox = sub_font.getbbox(sub_text)

        main_w = main_bbox[2] - main_bbox[0]
        main_h = main_bbox[3] - main_bbox[1]
        sub_w = sub_bbox[2] - sub_bbox[0]
        sub_h = sub_bbox[3] - sub_bbox[1]

        total_h = main_h + sub_h + 10  # 10px spacing between main and subtitle

        if main_w <= target_w and sub_w <= target_w and total_h <= target_h:
            return main_size, sub_size

        main_size -= 1

    return 12, 6

def draw_text_with_pil(cv_image, text, position, font_path, font_size, color, weight='normal', alpha=255):
    pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)).convert("RGBA")
    txt_layer = Image.new("RGBA", pil_image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_layer)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
    text_color = color + (alpha,)
    if weight == 'bold':
        offsets = [(-1,-1),(-1,1),(1,-1),(1,1),(-1,0),(1,0),(0,-1),(0,1),(-2,0),(2,0),(0,-2),(0,2)]
        for x_off, y_off in offsets:
            draw.text((position[0] + x_off, position[1] + y_off), text, font=font, fill=text_color)
    draw.text(position, text, font=font, fill=text_color)
    combined_image = Image.alpha_composite(pil_image, txt_layer)
    return cv2.cvtColor(np.array(combined_image), cv2.COLOR_RGBA2BGR)

def overlay_png(background, overlay, position, opacity=1.0):
    x, y = position
    h, w = overlay.shape[:2]
    if x + w > background.shape[1] or y + h > background.shape[0]:
        return
    alpha = (overlay[:, :, 3] / 255.0) * opacity
    overlay_bgr = overlay[:, :, :3]
    roi = background[y:y+h, x:x+w]
    bg_roi = (roi * (1.0 - alpha[:, :, np.newaxis])).astype(np.uint8)
    overlay_roi = (overlay_bgr * alpha[:, :, np.newaxis]).astype(np.uint8)
    background[y:y+h, x:x+w] = cv2.add(bg_roi, overlay_roi)

# Mouse click handler
def handle_mouse_click(event, x, y, flags, param):
    global app_state, is_play_stop_pressed, is_settings_pressed, is_back_pressed

    button_y_pos = white_window_y + white_window_height * 0.58
    play_stop_hitbox = ((175, button_y_pos - 50), (275, button_y_pos + 50))
    settings_hitbox = ((0, grey_window_y), (WIDTH, HEIGHT))
    back_hitbox = ((FRAME_PADDING, white_window_y), (120, white_window_y + white_window_height))

    if event == cv2.EVENT_LBUTTONDOWN:
        if app_state in ['STANDBY', 'LIVE'] and play_stop_hitbox[0][0] < x < play_stop_hitbox[1][0] and play_stop_hitbox[0][1] < y < play_stop_hitbox[1][1]:
            is_play_stop_pressed = True
        if app_state == 'STANDBY' and settings_hitbox[0][1] < y < settings_hitbox[1][1]:
            is_settings_pressed = True
        elif app_state == 'SETTINGS' and back_hitbox[0][0] < x < back_hitbox[1][0] and back_hitbox[0][1] < y < back_hitbox[1][1]:
            is_back_pressed = True

    elif event == cv2.EVENT_LBUTTONUP:
        if is_play_stop_pressed and play_stop_hitbox[0][0] < x < play_stop_hitbox[1][0] and play_stop_hitbox[0][1] < y < play_stop_hitbox[1][1]:
            app_state = 'LIVE' if app_state == 'STANDBY' else 'STANDBY'
        elif is_settings_pressed and settings_hitbox[0][1] < y < settings_hitbox[1][1]:
            app_state = 'SETTINGS'
        elif is_back_pressed and back_hitbox[0][0] < x < back_hitbox[1][0] and back_hitbox[0][1] < y < back_hitbox[1][1]:
            app_state = 'STANDBY'

        is_play_stop_pressed = False
        is_settings_pressed = False
        is_back_pressed = False

# --- Main Loop ---
cv2.namedWindow("ASL Recognition UI")
cv2.setMouseCallback("ASL Recognition UI", handle_mouse_click)

# Initialize positions
frame_inner_height = HEIGHT - (2 * FRAME_PADDING)
white_window_y = white_window_target_y = FRAME_PADDING
white_window_height = white_window_target_height = frame_inner_height * 0.90
white_window_x_padding = white_window_target_x_padding = FRAME_PADDING
grey_window_y = grey_window_target_y = white_window_y + white_window_height - 20
grey_window_height = grey_window_target_height = (HEIGHT - FRAME_PADDING) - grey_window_y
grey_window_x_padding = grey_window_target_x_padding = 40

while True:
    ui_canvas = np.full((S_HEIGHT, S_WIDTH, 3), CV_COLOR_UI_BACKGROUND, dtype=np.uint8)
    frame_inner_height = HEIGHT - (2 * FRAME_PADDING)

    # State Logic & Animation Targets
    if app_state == 'STANDBY':
        white_window_target_y = FRAME_PADDING; white_window_target_height = frame_inner_height * 0.90
        white_window_target_x_padding = FRAME_PADDING
        grey_window_target_x_padding = 40
        grey_window_target_y = white_window_target_y + white_window_target_height - 20
        grey_window_target_height = (HEIGHT - FRAME_PADDING) - grey_window_target_y
        settings_target_opacity = 0.0
    elif app_state == 'LIVE':
        white_window_target_y = FRAME_PADDING; white_window_target_height = frame_inner_height * 0.80
        white_window_target_x_padding = FRAME_PADDING
        grey_window_target_x_padding = 20
        grey_window_target_y = white_window_target_y + (white_window_target_height * 0.9)
        grey_window_target_height = (HEIGHT - FRAME_PADDING) - grey_window_target_y
        settings_target_opacity = 0.0
    elif app_state == 'SETTINGS':
        white_window_target_y = FRAME_PADDING; white_window_target_height = 80
        white_window_target_x_padding = 40
        grey_window_target_x_padding = FRAME_PADDING
        grey_window_target_y = white_window_target_y + white_window_target_height + 10
        grey_window_target_height = HEIGHT - grey_window_target_y - FRAME_PADDING
        settings_target_opacity = 1.0

    # Independent Button Animation Targets
    target_play_stop_size = 0.9 if is_play_stop_pressed else 1.0
    target_play_stop_opacity = 0.7 if is_play_stop_pressed else 1.0
    target_settings_opacity = 0.7 if is_settings_pressed else 1.0
    target_back_size = 0.9 if is_back_pressed else 1.0
    target_back_opacity = 0.7 if is_back_pressed else settings_opacity

    # Animate UI Elements
    white_window_y = lerp(white_window_y, white_window_target_y, animation_speed)
    white_window_height = lerp(white_window_height, white_window_target_height, animation_speed)
    white_window_x_padding = lerp(white_window_x_padding, white_window_target_x_padding, animation_speed)
    grey_window_y = lerp(grey_window_y, grey_window_target_y, animation_speed)
    grey_window_height = lerp(grey_window_height, grey_window_target_height, animation_speed)
    grey_window_x_padding = lerp(grey_window_x_padding, grey_window_target_x_padding, animation_speed)
    settings_opacity_anim = lerp(settings_opacity, settings_target_opacity, animation_speed)
    settings_opacity = settings_opacity_anim
    button_size_mult = lerp(button_size_mult, target_play_stop_size, animation_speed * 1)
    button_opacity = lerp(button_opacity, target_play_stop_opacity, animation_speed * 1)
    settings_icon_opacity = lerp(settings_icon_opacity, target_settings_opacity, animation_speed * 1.5)
    back_button_size_mult = lerp(back_button_size_mult, target_back_size, animation_speed * 1.5)
    back_button_opacity = lerp(back_button_opacity, target_back_opacity, animation_speed * 1.5)

    # Calculate scaled coordinates, draw shadows and windows
    s_pad = int(FRAME_PADDING * SCALE_FACTOR)
    s_wx_pad = int(white_window_x_padding * SCALE_FACTOR)
    s_gx_pad = int(grey_window_x_padding * SCALE_FACTOR)
    s_wy, s_wh = int(white_window_y * SCALE_FACTOR), int(white_window_height * SCALE_FACTOR)
    s_gy, s_gh = int(grey_window_y * SCALE_FACTOR), int(grey_window_height * SCALE_FACTOR)
    s_window_radius = int(WINDOW_RADIUS * SCALE_FACTOR)

    shadow_offset = 7 * SCALE_FACTOR
    draw_rounded_rectangle(ui_canvas, (s_gx_pad, s_gy + shadow_offset), (S_WIDTH - s_gx_pad, s_gy + s_gh + shadow_offset), CV_COLOR_SHADOW, radius=s_window_radius)
    draw_rounded_rectangle(ui_canvas, (s_wx_pad, s_wy + shadow_offset), (S_WIDTH - s_wx_pad, s_wy + s_wh + shadow_offset), CV_COLOR_SHADOW, radius=s_window_radius)
    ui_canvas = cv2.GaussianBlur(ui_canvas, (65, 65), 30)

    if app_state == 'LIVE':
        draw_rounded_rectangle(ui_canvas, (s_pad, s_wy), (S_WIDTH - s_pad, s_wy + s_wh), CV_COLOR_WHITE, radius=s_window_radius)
        draw_rounded_rectangle(ui_canvas, (s_gx_pad, s_gy), (S_WIDTH - s_gx_pad, s_gy + s_gh), CV_COLOR_GREY, radius=s_window_radius)
    else:
        draw_rounded_rectangle(ui_canvas, (s_gx_pad, s_gy), (S_WIDTH - s_gx_pad, s_gy + s_gh), CV_COLOR_GREY, radius=s_window_radius)
        white_radius = int(s_wh / 2) if app_state == 'SETTINGS' or abs(white_window_target_height - 80) < 5 else s_window_radius
        draw_rounded_rectangle(ui_canvas, (s_wx_pad, s_wy), (S_WIDTH - s_wx_pad, s_wy + s_wh), CV_COLOR_WHITE, radius=white_radius)

    # DYNAMIC WORD RENDERING
    if app_state in ['STANDBY', 'LIVE']:
        main_word = "Hello" if app_state == 'STANDBY' else "Latte"
        subtitle = "I'm Tickle Mcbunties" if app_state == 'STANDBY' else "Current Letter: e"
        target_text_width = (WIDTH - 2 * FRAME_PADDING - 2 * WORD_MARGIN) * SCALE_FACTOR
        target_text_height = (HEIGHT * 0.30) * SCALE_FACTOR
        if FONT_PATH:
            main_size, sub_size = get_dynamic_title_sizes(main_word, subtitle, SUBTITLE_RATIO, target_text_width, target_text_height, FONT_PATH)
            block_center_y = int(HEIGHT * 0.25 * SCALE_FACTOR)
            # Calculate vertical start so text block is centered vertically on block_center_y
            main_font = ImageFont.truetype(FONT_PATH, main_size)
            main_bbox = main_font.getbbox(main_word)
            main_h = main_bbox[3] - main_bbox[1]
            sub_font = ImageFont.truetype(FONT_PATH, sub_size)
            sub_bbox = sub_font.getbbox(subtitle)
            sub_h = sub_bbox[3] - sub_bbox[1]
            total_h = main_h + sub_h + 10

            block_top_y = block_center_y - (total_h // 2)

            main_w = main_bbox[2] - main_bbox[0]
            main_x = (S_WIDTH - main_w) // 2
            ui_canvas = draw_text_with_pil(ui_canvas, main_word, (main_x, block_top_y), FONT_PATH, main_size, PIL_COLOR_TEXT_DARK, weight='bold')

            sub_w = sub_bbox[2] - sub_bbox[0]
            sub_x = (S_WIDTH - sub_w) // 2
            sub_y = block_top_y + main_h + 10
            ui_canvas = draw_text_with_pil(ui_canvas, subtitle, (sub_x, sub_y), FONT_PATH, sub_size, PIL_COLOR_TEXT_SUBTLE)

    # RENDER BUTTONS AND ICONS
    button_y_pos = int((white_window_y + white_window_height * 0.58) * SCALE_FACTOR)
    base_button_size = 100 * SCALE_FACTOR

    if app_state == 'STANDBY':
        current_button_size = int(base_button_size * button_size_mult)
        h, w = live_button_img.shape[:2]
        target_h = current_button_size
        target_w = int(w * target_h / h)
        resized_live = cv2.resize(live_button_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        overlay_png(ui_canvas, resized_live, ((S_WIDTH - target_w) // 2, button_y_pos - (target_h // 2)), opacity=button_opacity)

        settings_icon_size = 45 * SCALE_FACTOR
        h, w = settings_icon_img.shape[:2]
        target_h = settings_icon_size
        target_w = int(w * target_h / h)
        resized_settings = cv2.resize(settings_icon_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        settings_y_pos = int((grey_window_y + grey_window_height * 0.6) * SCALE_FACTOR) - (target_h // 2)
        overlay_png(ui_canvas, resized_settings, ((S_WIDTH - target_w) // 2, settings_y_pos), opacity=settings_icon_opacity)

    elif app_state == 'LIVE':
        current_button_size = int(base_button_size * button_size_mult)
        h, w = standby_button_img.shape[:2]
        target_h = current_button_size
        target_w = int(w * target_h / h)
        resized_standby = cv2.resize(standby_button_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        overlay_png(ui_canvas, resized_standby, ((S_WIDTH - target_w) // 2, button_y_pos - (target_h // 2)), opacity=button_opacity)

        font_sent = ImageFont.truetype(FONT_PATH, 50)
        left_sent, _, right_sent, _ = font_sent.getbbox("can I get a latte . .")
        text_w = right_sent - left_sent
        ui_canvas = draw_text_with_pil(ui_canvas, "can I get a latte . .", ((S_WIDTH - text_w)//2, int((grey_window_y + 40)*SCALE_FACTOR)), FONT_PATH, 50, PIL_COLOR_TEXT_LIGHT)

    elif app_state == 'SETTINGS':
        header_center_y = s_wy + (s_wh // 2)

        current_back_button_size = int(50 * SCALE_FACTOR * back_button_size_mult)
        h, w = back_button_img.shape[:2]
        target_h = current_back_button_size
        target_w = int(w*target_h/h)
        resized_back = cv2.resize(back_button_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        overlay_png(ui_canvas, resized_back, (40*SCALE_FACTOR, header_center_y - target_h//2), opacity=back_button_opacity)

        icon_size = 45 * SCALE_FACTOR
        h, w = settings_icon_img.shape[:2]
        target_h = icon_size
        target_w = int(w*target_h/h)
        resized_icon = cv2.resize(settings_icon_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        overlay_png(ui_canvas, resized_icon, ((S_WIDTH - target_w) // 2, header_center_y - target_h//2), opacity=settings_opacity_anim)

    # Downscale for Anti-Aliasing and Masking
    ui_canvas_smooth = cv2.resize(ui_canvas, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    mask = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    draw_rounded_rectangle(mask, (0, 0), (WIDTH, HEIGHT), (255, 255, 255), radius=FRAME_RADIUS)
    final_frame = np.where(mask > 0, ui_canvas_smooth, CV_COLOR_DISPLAY_BACKGROUND).astype(np.uint8)

    cv2.imshow("ASL Recognition UI", final_frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
