"""
Webcam & Gesture Mouse Web Streamer (Flask + OpenCV)
====================================================
Streams the live webcam feed with MediaPipe gesture tracking
directly into a web page you can view right inside Antigravity!
"""

import math
import os
import sys
import time
from typing import List, Optional, Tuple

import cv2
from flask import Flask, Response, render_template_string
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyautogui

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.0

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

app = Flask(__name__)

class GestureMouseEngine:
    def __init__(self):
        self.screen_w, self.screen_h = pyautogui.size()
        self.prev_x, self.prev_y = pyautogui.position()
        self.is_dragging = False
        self.pinch_start_time = None
        self.last_click_time = 0.0
        self.prev_scroll_y = None
        self.last_right_click_time = 0.0
        self.bye_start_time = None
        self.bye_progress = 0.0
        self.fps = 0.0
        self.prev_frame_time = time.time()

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def get_extended_fingers(self, lm):
        tip_ids = [8, 12, 16, 20]
        pip_ids = [6, 10, 14, 18]
        extended = []
        d_tip = math.hypot(lm[4].x - lm[17].x, lm[4].y - lm[17].y)
        d_ip = math.hypot(lm[3].x - lm[17].x, lm[3].y - lm[17].y)
        extended.append(d_tip > d_ip * 1.08)
        for tip, pip in zip(tip_ids, pip_ids):
            extended.append(lm[tip].y < lm[pip].y)
        return extended

    def move_cursor(self, ix, iy):
        target_x = np.interp(ix, (80, 560), (0, self.screen_w))
        target_y = np.interp(iy, (60, 420), (0, self.screen_h))
        # Clamp away from corners to avoid pyautogui failsafe (5px margin)
        target_x = min(max(target_x, 5), self.screen_w - 5)
        target_y = min(max(target_y, 5), self.screen_h - 5)

        dx = target_x - self.prev_x
        dy = target_y - self.prev_y
        dist = math.hypot(dx, dy)
        smoothing = max(1.8, 3.5 - (dist / 100.0))

        cur_x = self.prev_x + dx / smoothing
        cur_y = self.prev_y + dy / smoothing
        cur_x = min(max(cur_x, 5), self.screen_w - 5)
        cur_y = min(max(cur_y, 5), self.screen_h - 5)
        try:
            pyautogui.moveTo(cur_x, cur_y)
        except pyautogui.FailSafeException:
            pyautogui.moveTo(self.screen_w // 2, self.screen_h // 2)
        self.prev_x, self.prev_y = cur_x, cur_y

    def process_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return None

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

            points = [(int(pt.x * w), int(pt.y * h)) for pt in lm]
            for p1, p2 in HAND_CONNECTIONS:
                cv2.line(frame, points[p1], points[p2], (255, 140, 0), 2, cv2.LINE_AA)
            for pt in points:
                cv2.circle(frame, pt, 4, (0, 255, 200), cv2.FILLED)

            tx, ty = points[4]
            ix, iy = points[8]
            mx, my = points[12]

            thumb_up, index_up, middle_up, ring_up, pinky_up = self.get_extended_fingers(lm)
            pinch_dist = math.hypot(tx - ix, ty - iy)
            pinch_mid_dist = math.hypot(tx - mx, ty - my)
            all_five_up = thumb_up and index_up and middle_up and ring_up and pinky_up

            if all_five_up:
                mode = "👋 Waving Bye..."
                if self.bye_start_time is None:
                    self.bye_start_time = now
                elapsed = now - self.bye_start_time
                self.bye_progress = min(elapsed / 2.0, 1.0)
            else:
                self.bye_start_time = None
                self.bye_progress = 0.0

                if index_up and middle_up and not ring_up and not pinky_up and pinch_dist > 40:
                    mode = "Scrolling"
                    cv2.circle(frame, (ix, iy), 8, (255, 0, 255), cv2.FILLED)
                    cv2.circle(frame, (mx, my), 8, (255, 0, 255), cv2.FILLED)
                    avg_y = (iy + my) / 2.0
                    if self.prev_scroll_y is not None:
                        dy = self.prev_scroll_y - avg_y
                        if abs(dy) >= 3:
                            pyautogui.scroll(int(dy * 4.0))
                    self.prev_scroll_y = avg_y

                elif (index_up and middle_up and ring_up and not pinky_up) or (pinch_mid_dist < 42 and not index_up):
                    mode = "Right Click"
                    self.prev_scroll_y = None
                    if now - self.last_right_click_time > 0.44:
                        pyautogui.rightClick()
                        self.last_right_click_time = now
                        cv2.circle(frame, ((tx + mx) // 2, (ty + my) // 2), 16, (0, 165, 255), cv2.FILLED)

                elif index_up and not ring_up and not pinky_up:
                    self.prev_scroll_y = None
                    self.move_cursor(ix, iy)
                    cv2.circle(frame, (ix, iy), 10, (0, 255, 0), cv2.FILLED)

                    is_pinched = pinch_dist < 40
                    mid_pt = ((tx + ix) // 2, (ty + iy) // 2)

                    if is_pinched:
                        cv2.line(frame, (tx, ty), (ix, iy), (0, 0, 255), 3)
                        cv2.circle(frame, mid_pt, 12, (0, 0, 255), cv2.FILLED)
                        if self.pinch_start_time is None:
                            self.pinch_start_time = now
                        if (now - self.pinch_start_time) >= 0.25 and not self.is_dragging:
                            pyautogui.mouseDown()
                            self.is_dragging = True
                        mode = "DRAGGING" if self.is_dragging else "PINCHING"
                    else:
                        if self.is_dragging:
                            pyautogui.mouseUp()
                            self.is_dragging = False
                            self.pinch_start_time = None
                            mode = "DROPPED"
                        elif self.pinch_start_time is not None:
                            duration = now - self.pinch_start_time
                            self.pinch_start_time = None
                            if duration < 0.25 and (now - self.last_click_time) > 0.22:
                                pyautogui.click()
                                self.last_click_time = now
                                cv2.circle(frame, mid_pt, 18, (0, 255, 0), cv2.FILLED)
                                mode = "CLICK"
                            else:
                                mode = "Moving Cursor"
                        else:
                            mode = "Moving Cursor"
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

        # HUD
        cv2.rectangle(frame, (80, 60), (w - 80, h - 60), (255, 120, 0), 2)
        status_color = (0, 255, 128) if hand_present else (80, 80, 240)
        cv2.putText(frame, f"MODE: {mode.upper()}", (15, 36), cv2.FONT_HERSHEY_DUPLEX, 0.75, status_color, 2)
        cv2.putText(frame, f"FPS: {int(self.fps)}", (w - 120, 36), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)

        if self.bye_progress > 0.0:
            bar_w = int(w * 0.6)
            bx = (w - bar_w) // 2
            by = 70
            cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 18), (50, 50, 50), -1)
            fill_w = int(bar_w * min(self.bye_progress, 1.0))
            cv2.rectangle(frame, (bx, by), (bx + fill_w, by + 18), (0, 0, 255), -1)
            cv2.putText(frame, f"BYE GESTURE ({int(self.bye_progress * 100)}%)", (bx + 15, by + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes() if ret else None

engine = None

def get_engine():
    global engine
    if engine is None:
        engine = GestureMouseEngine()
    return engine

def generate_frames():
    eng = get_engine()
    while True:
        frame_bytes = eng.process_frame()
        if frame_bytes is None:
            time.sleep(0.02)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI Virtual Mouse - Live Stream</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: #0f172a; color: #f8fafc; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; padding: 20px; }
        .container { background: #1e293b; border-radius: 16px; padding: 24px; box-shadow: 0 20px 40px rgba(0,0,0,0.5); border: 1px solid #334155; max-width: 760px; width: 100%; text-align: center; }
        h1 { font-size: 1.6rem; margin-bottom: 8px; color: #38bdf8; display: flex; align-items: center; justify-content: center; gap: 8px; }
        p.subtitle { color: #94a3b8; font-size: 0.95rem; margin-bottom: 20px; }
        .video-box { position: relative; border-radius: 12px; overflow: hidden; border: 2px solid #38bdf8; background: #000; display: inline-block; box-shadow: 0 8px 24px rgba(56, 189, 248, 0.2); }
        .video-box img { display: block; max-width: 100%; height: auto; }
        .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-top: 24px; text-align: left; }
        .card { background: #0f172a; padding: 12px 14px; border-radius: 10px; border: 1px solid #334155; }
        .card b { display: block; color: #38bdf8; font-size: 0.85rem; margin-bottom: 4px; }
        .card span { font-size: 0.8rem; color: #cbd5e1; }
    </style>
</head>
<body>
    <div class="container">
        <h1>✨ AI Virtual Mouse (Live Inside Antigravity)</h1>
        <p class="subtitle">Real-time webcam stream & hand gesture tracking</p>
        
        <div class="video-box">
            <img src="/video_feed" alt="Webcam Live Stream" />
        </div>

        <div class="cards">
            <div class="card"><b>☝️ Move Cursor</b><span>Point Index Finger UP</span></div>
            <div class="card"><b>🤏 Left Click</b><span>Thumb + Index Quick Pinch</span></div>
            <div class="card"><b>🤏 Drag & Drop</b><span>Hold Pinch > 0.25s & Move</span></div>
            <div class="card"><b>🤟 Right Click</b><span>3 Fingers UP</span></div>
            <div class="card"><b>✌️ Smooth Scroll</b><span>Peace Sign (Move Hand)</span></div>
            <div class="card"><b>👋 Bye Gesture</b><span>Wave 5 Fingers</span></div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Starting Antigravity Webcam Server on http://127.0.0.1:5000 ...")
    app.run(host='127.0.0.1', port=5000, threaded=True)
