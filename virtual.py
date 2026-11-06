"""
AI Virtual Mouse - Hand Gesture Controlled Mouse
=================================================
Controls your computer's cursor, clicks, dragging, scrolling, and window closing
using hand gestures captured from your webcam.

Unique Gestures:
  - 👋 BYE / Wave Gesture (Open 5 fingers waving or held 1.5s):
       CLOSES the window / application cleanly!
  - ☝️ Move Cursor: Point Index finger UP (smooth jitter-free tracking)
  - 🤏 Left Click: Quick tap/pinch of Index finger tip and Thumb (or Index folded click)
  - 🤏 Drag & Drop: Hold Thumb + Index pinch (> 0.25s) to drag; release to drop
  - 🤏🤏 Double Click: Double pinch within 0.35s
  - ✌️ Middle + Thumb Tap (or Peace Sign): Right Click
  - ✌️ Smooth Scroll: Index + Middle fingers UP together; move hand up/down
  - ✊ Pause / Rest: Closed fist freezes cursor

Controls:
  - 'q' or 'ESC': Quit application
  - 'r': Reset cursor to center of screen
"""

import math
import os
import sys
import time
import urllib.request
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from pynput.mouse import Button, Controller as MouseController
import ctypes

# pynput mouse controller (works from any process on Windows)
_mouse = MouseController()

def _get_screen_size():
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

def _mouse_move(x, y):
    _mouse.position = (int(x), int(y))

def _mouse_click():
    _mouse.click(Button.left)

def _mouse_double_click():
    _mouse.click(Button.left, 2)

def _mouse_right_click():
    _mouse.click(Button.right)

def _mouse_down():
    _mouse.press(Button.left)

def _mouse_up():
    _mouse.release(Button.left)

def _mouse_scroll(dy):
    _mouse.scroll(0, dy)

def _mouse_position():
    return _mouse.position

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]


class Config:
    CAM_WIDTH: int = 640
    CAM_HEIGHT: int = 480

    # Interaction bounding box margins (px)
    FRAME_MARGIN_X: int = 80
    FRAME_MARGIN_Y: int = 60

    # Mouse smoothing (adaptive: low value when moving fast, higher when holding)
    SMOOTHING: float = 3.5

    # Pinch distance thresholds (px)
    PINCH_CLICK_DIST: int = 40        # Generous threshold for easy clicking
    PINCH_RIGHT_DIST: int = 42

    # Timing parameters (seconds)
    CLICK_COOLDOWN: float = 0.22      # Delay between clicks
    DRAG_HOLD_DELAY: float = 0.25     # Pinch longer than this triggers Drag & Drop
    DOUBLE_CLICK_GAP: float = 0.38    # Time gap to trigger double click
    SCROLL_DEADZONE: int = 3          # Pixel delta to trigger scroll
    SCROLL_MULTIPLIER: float = 4.0

    # Bye gesture hold time (seconds) to close window
    BYE_HOLD_DURATION: float = 2.0


def ensure_model_exists():
    """Downloads the MediaPipe hand landmarker model if missing."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading hand landmarker model to {MODEL_PATH}...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Model downloaded successfully!")
        except Exception as e:
            print(f"[Error] Failed to download model: {e}")
            sys.exit(1)


class VirtualMouse:
    def __init__(self):
        ensure_model_exists()

        self.screen_w, self.screen_h = _get_screen_size()

        # State tracking
        self.prev_x, self.prev_y = _mouse_position()
        self.is_dragging: bool = False
        self.pinch_start_time: Optional[float] = None
        self.last_click_time: float = 0.0
        self.prev_scroll_y: Optional[float] = None
        self.last_right_click_time: float = 0.0

        # Bye / Close gesture tracking
        self.bye_start_time: Optional[float] = None
        self.bye_progress: float = 0.0

        # Performance metric
        self.prev_frame_time: float = time.time()
        self.fps: float = 0.0

        # Initialize MediaPipe HandLandmarker with high sensitivity
        try:
            base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.45,  # lowered for instant detection
                min_tracking_confidence=0.45,
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            print(f"[Error] Failed to initialize HandLandmarker: {e}")
            sys.exit(1)

        # Open webcam (try DirectShow on Windows, fallback to default)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            print("[Error] Could not open webcam. Ensure camera is connected.")
            sys.exit(1)

    @staticmethod
    def get_extended_fingers(lm) -> List[bool]:
        """
        Determines which fingers are extended.
        Returns: [thumb, index, middle, ring, pinky]
        """
        tip_ids = [8, 12, 16, 20]
        pip_ids = [6, 10, 14, 18]

        extended = []

        # Thumb extension: tip distance to pinky base (lm 17) vs IP joint (lm 3)
        pinky_base = lm[17]
        thumb_tip = lm[4]
        thumb_ip = lm[3]
        d_tip = math.hypot(thumb_tip.x - pinky_base.x, thumb_tip.y - pinky_base.y)
        d_ip = math.hypot(thumb_ip.x - pinky_base.x, thumb_ip.y - pinky_base.y)
        extended.append(d_tip > d_ip * 1.08)

        # 4 Fingers: extended if tip y is noticeably above PIP joint y
        for tip, pip in zip(tip_ids, pip_ids):
            extended.append(lm[tip].y < lm[pip].y)

        return extended

    def move_cursor(self, tip_x: float, tip_y: float):
        """Map camera coordinates to screen coordinates with adaptive smoothing."""
        target_x = np.interp(
            tip_x,
            (Config.FRAME_MARGIN_X, Config.CAM_WIDTH - Config.FRAME_MARGIN_X),
            (0, self.screen_w),
        )
        target_y = np.interp(
            tip_y,
            (Config.FRAME_MARGIN_Y, Config.CAM_HEIGHT - Config.FRAME_MARGIN_Y),
            (0, self.screen_h),
        )

        target_x = min(max(target_x, 1), self.screen_w - 1)
        target_y = min(max(target_y, 1), self.screen_h - 1)

        dx = target_x - self.prev_x
        dy = target_y - self.prev_y
        dist = math.hypot(dx, dy)

        smoothing = max(Config.SMOOTHING * 0.5, Config.SMOOTHING - (dist / 100.0))

        cur_x = self.prev_x + dx / smoothing
        cur_y = self.prev_y + dy / smoothing

        cur_x = min(max(cur_x, 1), self.screen_w - 1)
        cur_y = min(max(cur_y, 1), self.screen_h - 1)

        _mouse_move(cur_x, cur_y)
        self.prev_x, self.prev_y = cur_x, cur_y

    def handle_left_pinch(self, pinch_dist: float, thumb_pt: Tuple[int, int], index_pt: Tuple[int, int], frame):
        """Handles single click, double click, and drag-and-drop."""
        now = time.time()
        mid_pt = ((thumb_pt[0] + index_pt[0]) // 2, (thumb_pt[1] + index_pt[1]) // 2)

        is_pinched = pinch_dist < Config.PINCH_CLICK_DIST

        if is_pinched:
            cv2.line(frame, thumb_pt, index_pt, (0, 0, 255), 3)
            cv2.circle(frame, mid_pt, 12, (0, 0, 255), cv2.FILLED)

            if self.pinch_start_time is None:
                self.pinch_start_time = now

            hold_duration = now - self.pinch_start_time
            if hold_duration >= Config.DRAG_HOLD_DELAY and not self.is_dragging:
                _mouse_down()
                self.is_dragging = True

            return "DRAGGING" if self.is_dragging else "PINCHING"

        else:
            if self.is_dragging:
                _mouse_up()
                self.is_dragging = False
                self.pinch_start_time = None
                return "DROPPED"

            elif self.pinch_start_time is not None:
                duration = now - self.pinch_start_time
                self.pinch_start_time = None

                if duration < Config.DRAG_HOLD_DELAY and (now - self.last_click_time) > Config.CLICK_COOLDOWN:
                    if (now - self.last_click_time) < Config.DOUBLE_CLICK_GAP:
                        _mouse_double_click()
                        self.last_click_time = now
                        cv2.circle(frame, mid_pt, 22, (255, 255, 0), cv2.FILLED)
                        return "DOUBLE CLICK"
                    else:
                        _mouse_click()
                        self.last_click_time = now
                        cv2.circle(frame, mid_pt, 18, (0, 255, 0), cv2.FILLED)
                        return "CLICK"

        return None

    def draw_skeleton(self, frame, lm):
        """Draws hand skeleton and joints clearly."""
        h, w, _ = frame.shape
        points = [(int(pt.x * w), int(pt.y * h)) for pt in lm]

        for p1, p2 in HAND_CONNECTIONS:
            cv2.line(frame, points[p1], points[p2], (255, 140, 0), 2, cv2.LINE_AA)

        for pt in points:
            cv2.circle(frame, pt, 4, (0, 255, 200), cv2.FILLED)

    def draw_hud(self, frame, mode: str, hand_present: bool):
        """Renders overlay with boundary box, mode, FPS, bye progress, and tips."""
        h, w, _ = frame.shape

        # Active boundary box
        cv2.rectangle(
            frame,
            (Config.FRAME_MARGIN_X, Config.FRAME_MARGIN_Y),
            (w - Config.FRAME_MARGIN_X, h - Config.FRAME_MARGIN_Y),
            (255, 120, 0),
            2,
            cv2.LINE_AA,
        )

        # Header Bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), (15, 15, 15), -1)
        cv2.rectangle(overlay, (0, h - 35), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Status text & color
        status_color = (0, 255, 128) if hand_present else (80, 80, 240)
        cv2.putText(
            frame,
            f"MODE: {mode.upper()}",
            (15, 36),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            status_color,
            2,
            cv2.LINE_AA,
        )

        # FPS indicator
        cv2.putText(
            frame,
            f"FPS: {int(self.fps)}",
            (w - 120, 36),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Bye gesture closing progress bar
        if self.bye_progress > 0.0:
            bar_w = int(w * 0.6)
            bx = (w - bar_w) // 2
            by = 70
            cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 18), (50, 50, 50), -1)
            fill_w = int(bar_w * min(self.bye_progress, 1.0))
            cv2.rectangle(frame, (bx, by), (bx + fill_w, by + 18), (0, 0, 255), -1)
            cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 18), (255, 255, 255), 2)
            cv2.putText(
                frame,
                f"WAVING BYE... CLOSING ({int(self.bye_progress * 100)}%)",
                (bx + 15, by + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Footer guide
        tips = "Point 1: Move | Pinch: Click/Drag | Peace: Scroll | Wave 5: BYE (Close) | 'q': Quit"
        cv2.putText(
            frame,
            tips,
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

    def run(self):
        """Main loop capturing webcam and translating gestures to mouse actions."""
        print("\n" + "=" * 65)
        print("  AI Virtual Mouse is ACTIVE!")
        print("    [BYE] Wave 5 fingers:       Close application")
        print("    [MOVE] Index finger up:     Move cursor")
        print("    [CLICK] Thumb + Index:      Left Click / Hold to Drag")
        print("    [SCROLL] Index + Middle:    Scroll (move hand up/down)")
        print("    [RIGHT CLICK] 3 fingers:    Right Click")
        print("    [PAUSE] Fist:               Pause / Rest cursor")
        print("    [QUIT] Press 'q' or 'ESC':  Quit")
        print("=" * 65 + "\n")

        cv2.namedWindow("AI Virtual Mouse (q to quit)", cv2.WINDOW_AUTOSIZE)

        while True:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_image)

            now = time.time()
            self.fps = 1.0 / (now - self.prev_frame_time) if now != self.prev_frame_time else 0.0
            self.prev_frame_time = now

            mode = "No Hand Detected"
            hand_present = False

            if result.hand_landmarks:
                hand_present = True
                lm = result.hand_landmarks[0]

                # Draw skeleton
                self.draw_skeleton(frame, lm)

                # Landmark coordinates
                tx, ty = int(lm[4].x * w), int(lm[4].y * h)    # Thumb tip
                ix, iy = int(lm[8].x * w), int(lm[8].y * h)    # Index tip
                mx, my = int(lm[12].x * w), int(lm[12].y * h)  # Middle tip

                # Detect extended fingers
                thumb_up, index_up, middle_up, ring_up, pinky_up = self.get_extended_fingers(lm)
                pinch_index_dist = math.hypot(tx - ix, ty - iy)
                pinch_middle_dist = math.hypot(tx - mx, ty - my)

                # All 5 fingers extended and spread
                all_five_up = thumb_up and index_up and middle_up and ring_up and pinky_up

                # =========================================================
                # 1. UNIQUE BYE GESTURE: Open 5 Fingers held for 2 seconds
                # =========================================================
                if all_five_up:
                    mode = "Waving Bye (Closing...)"
                    if self.bye_start_time is None:
                        self.bye_start_time = now

                    elapsed = now - self.bye_start_time
                    self.bye_progress = min(elapsed / Config.BYE_HOLD_DURATION, 1.0)

                    # When held for full duration, close cleanly!
                    if elapsed >= Config.BYE_HOLD_DURATION:
                        print("\n[BYE] Gesture detected! Closing Virtual Mouse window...\n")
                        break
                else:
                    self.bye_start_time = None
                    self.bye_progress = 0.0

                    # =====================================================
                    # 2. SCROLL MODE: Index and Middle UP (Peace sign)
                    # =====================================================
                    if index_up and middle_up and not ring_up and not pinky_up and pinch_index_dist > Config.PINCH_CLICK_DIST:
                        mode = "Scrolling"
                        cv2.circle(frame, (ix, iy), 8, (255, 0, 255), cv2.FILLED)
                        cv2.circle(frame, (mx, my), 8, (255, 0, 255), cv2.FILLED)

                        avg_y = (iy + my) / 2.0
                        if self.prev_scroll_y is not None:
                            dy = self.prev_scroll_y - avg_y
                            if abs(dy) >= Config.SCROLL_DEADZONE:
                                scroll_val = int(dy * Config.SCROLL_MULTIPLIER)
                                pyautogui.scroll(scroll_val)
                        self.prev_scroll_y = avg_y

                    # =====================================================
                    # 3. RIGHT CLICK: 3 fingers up, OR Thumb + Middle pinch
                    # =====================================================
                    elif (index_up and middle_up and ring_up and not pinky_up) or (pinch_middle_dist < Config.PINCH_RIGHT_DIST and not index_up):
                        mode = "Right Click"
                        self.prev_scroll_y = None
                        if now - self.last_right_click_time > Config.CLICK_COOLDOWN * 2.0:
                            pyautogui.rightClick()
                            self.last_right_click_time = now
                            cv2.circle(frame, ((tx + mx) // 2, (ty + my) // 2), 16, (0, 165, 255), cv2.FILLED)

                    # =====================================================
                    # 4. MOVE & PINCH/CLICK/DRAG: Index finger extended
                    # =====================================================
                    elif index_up and not ring_up and not pinky_up:
                        self.prev_scroll_y = None

                        # Smooth cursor movement
                        self.move_cursor(ix, iy)
                        cv2.circle(frame, (ix, iy), 10, (0, 255, 0), cv2.FILLED)

                        # Handle pinch for click / drag & drop
                        pinch_action = self.handle_left_pinch(pinch_index_dist, (tx, ty), (ix, iy), frame)
                        if pinch_action:
                            mode = pinch_action
                        else:
                            mode = "Moving Cursor"

                    # =====================================================
                    # 5. PAUSE / IDLE: Closed fist
                    # =====================================================
                    else:
                        self.prev_scroll_y = None
                        if self.is_dragging:
                            pyautogui.mouseUp()
                            self.is_dragging = False
                            self.pinch_start_time = None
                        mode = "Idle / Paused"

            else:
                self.prev_scroll_y = None
                self.bye_start_time = None
                self.bye_progress = 0.0
                if self.is_dragging:
                    pyautogui.mouseUp()
                    self.is_dragging = False
                    self.pinch_start_time = None

            # Render HUD
            self.draw_hud(frame, mode, hand_present)

            # Display window
            cv2.imshow("AI Virtual Mouse (q to quit)", frame)

            # Keyboard shortcuts
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or ESC
                break
            elif key == ord("r"):      # Reset cursor to center
                pyautogui.moveTo(self.screen_w // 2, self.screen_h // 2)

        # Cleanup
        if self.is_dragging:
            pyautogui.mouseUp()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    mouse = VirtualMouse()
    mouse.run()