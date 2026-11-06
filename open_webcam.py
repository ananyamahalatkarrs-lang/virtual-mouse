"""
Simple Webcam Viewer
====================
A minimal script to test and open your webcam.
Press 'q' or 'ESC' to close the window.
"""

import cv2
import sys

def main():
    print("Opening webcam...")
    
    # Try DirectShow first on Windows for fast opening, fallback to default
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[Error] Could not open webcam. Make sure your camera is connected.")
        sys.exit(1)

    print("Webcam successfully opened! Press 'q' or 'ESC' in the window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Failed to grab frame.")
            break

        # Flip horizontally for natural mirror effect
        frame = cv2.flip(frame, 1)

        # Overlay instructions
        cv2.putText(
            frame,
            "Webcam Live - Press 'q' to Quit",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Display the frame
        cv2.imshow("Webcam Live", frame)

        # Exit on 'q' or ESC (key code 27)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")

if __name__ == "__main__":
    main()
