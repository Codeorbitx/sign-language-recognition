"""
Video Recorder for Isolated Sign Data Collection
------------------------------------------------------------
Records short video clips of one sign at a time and automatically saves
them into: raw_videos/<signer_name>/<sign_name>/take_N.mp4
"""

import cv2
import os
import re
import time

RAW_VIDEOS_DIR = "raw_videos"
RECORD_SECONDS = 2.5
FPS = 30


def get_next_take_number(folder):
    """Sabse bada existing take number dhoondo, count nahi — taaki
    gaps (failed/missing takes) hone par bhi files overwrite na hon."""
    max_num = 0
    pattern = re.compile(r"take_(\d+)\.mp4$")
    for f in os.listdir(folder):
        match = pattern.match(f)
        if match:
            max_num = max(max_num, int(match.group(1)))
    return max_num + 1


def main():
    signer_name = input("Signer name (e.g. signer1): ").strip()
    sign_name = input("Sign name (e.g. hello): ").strip()

    save_dir = os.path.join(RAW_VIDEOS_DIR, signer_name, sign_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera open nahi hui.")
        return

    # ZAROORI FIX: pehle ek frame padho taaki camera warm up ho aur
    # sahi width/height mile — warna VideoWriter chup-chaap fail hota hai.
    ret, warm_frame = cap.read()
    if not ret:
        print("❌ Camera se frame nahi mil raha.")
        cap.release()
        return
    height, width = warm_frame.shape[:2]

    print(f"\nCamera resolution: {width}x{height}")
    print("Ready! 'r' dabao recording start karne ke liye, 'q' dabao quit karne ke liye.\n")

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

                # ZAROORI FIX: check karo file actually bani aur khali nahi
                if os.path.exists(take_path) and os.path.getsize(take_path) > 1024:
                    print(f"✅ Saved: {take_path} ({os.path.getsize(take_path)} bytes)")
                else:
                    print(f"❌ FAILED (file empty/missing): {take_path} — dobara try karo (press r)")
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

            if not writer.isOpened():
                print(f"❌ VideoWriter khul nahi saka take_{take_num} ke liye — 'r' dobara dabao")
                writer = None
                continue

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