"""
Video Recorder for Isolated Sign Data Collection
------------------------------------------------------------
Records short video clips of one sign at a time and automatically saves
them into: raw_videos/<signer_name>/<sign_name>/take_N.mp4

This folder structure is what extract_features.py expects later, so you
don't need to make any labels.csv by hand.

CONTROLS (while camera window is open):
    r  -> start recording one take (records for RECORD_SECONDS seconds)
    q  -> quit and close camera

RUN (one session per sign):
    python record_videos.py
    -> it will ask for signer name and sign name
    -> press 'r' repeatedly to record multiple takes of that same sign
    -> press 'q' when done, then run again for the next sign
"""

import cv2
import os
import time

RAW_VIDEOS_DIR = "raw_videos"
RECORD_SECONDS = 2.5   # length of each take — increase if your signs are longer
FPS = 30


def get_next_take_number(folder):
    existing = [f for f in os.listdir(folder) if f.startswith("take_") and f.endswith(".mp4")]
    return len(existing) + 1


def main():
    signer_name = input("Signer name (e.g. signer1): ").strip()
    sign_name = input("Sign name (e.g. hello): ").strip()

    save_dir = os.path.join(RAW_VIDEOS_DIR, signer_name, sign_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera open nahi hui. Camera connected hai, ya kisi aur app mein busy to nahi, check karo.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("\nReady! 'r' dabao recording start karne ke liye, 'q' dabao quit karne ke liye.\n")

    recording = False
    writer = None
    record_start_time = None
    take_path = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()

        if recording:
            elapsed = time.time() - record_start_time
            cv2.putText(display_frame, f"REC {elapsed:.1f}s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            writer.write(frame)
            if elapsed >= RECORD_SECONDS:
                writer.release()
                recording = False
                print(f"✅ Saved: {take_path}")
        else:
            cv2.putText(display_frame, "Press 'r' to record | 'q' to quit", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Record Sign Videos", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r') and not recording:
            take_num = get_next_take_number(save_dir)
            take_path = os.path.join(save_dir, f"take_{take_num}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(take_path, fourcc, FPS, (width, height))
            recording = True
            record_start_time = time.time()

        elif key == ord('q'):
            break

    cap.release()
    if writer is not None and recording:
        writer.release()
    cv2.destroyAllWindows()
    print("\nDone recording session.")


if __name__ == "__main__":
    main()
