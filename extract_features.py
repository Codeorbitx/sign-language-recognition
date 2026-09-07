"""
Feature Extraction for Isolated Sign Recognition
------------------------------------------------------------
Walks raw_videos/<signer>/<sign_name>/take_N.mp4 (created by record_videos.py),
runs MediaPipe Holistic on every frame, and pools each video into ONE fixed-length
feature row (mean + std of landmarks across all frames). Saves everything into
features.csv — one row per video, ready for train_classifier.py.

RUN:
    pip install -r requirements.txt
    python extract_features.py
"""

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

mp_holistic = mp.solutions.holistic

RAW_VIDEOS_DIR = "raw_videos"
OUTPUT_CSV = "features.csv"


def extract_landmarks(results):
    def get_coords(landmark_list, n_points):
        if landmark_list is None:
            return np.zeros((n_points, 3))
        return np.array([[lm.x, lm.y, lm.z] for lm in landmark_list.landmark])

    def get_arm_coords(pose_landmarks):
        if pose_landmarks is None:
            return np.zeros((6, 3))
        idxs = [11, 12, 13, 14, 15, 16]
        pts = [pose_landmarks.landmark[i] for i in idxs]
        return np.array([[p.x, p.y, p.z] for p in pts])

    left_hand = get_coords(results.left_hand_landmarks, 21)
    right_hand = get_coords(results.right_hand_landmarks, 21)
    arms = get_arm_coords(results.pose_landmarks)

    # Normalize relative to shoulder midpoint + shoulder width, so the signer's
    # distance/position from the camera doesn't change the feature values.
    if results.pose_landmarks is not None:
        shoulder_l = np.array([arms[0][0], arms[0][1], arms[0][2]])
        shoulder_r = np.array([arms[1][0], arms[1][1], arms[1][2]])
        center = (shoulder_l + shoulder_r) / 2
        scale = np.linalg.norm(shoulder_l - shoulder_r)
        scale = scale if scale > 1e-6 else 1.0
    else:
        center = np.zeros(3)
        scale = 1.0

    left_hand = (left_hand - center) / scale
    right_hand = (right_hand - center) / scale
    arms = (arms - center) / scale

    return np.concatenate([left_hand.flatten(), right_hand.flatten(), arms.flatten()])


def pool_video(video_path, holistic):
    cap = cv2.VideoCapture(video_path)
    frame_feats = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        frame_feats.append(extract_landmarks(results))
    cap.release()

    if len(frame_feats) == 0:
        return None

    frame_feats = np.array(frame_feats)          # (num_frames, num_features)
    return frame_feats.mean(axis=0)  # single fixed-length vector per video (mean only)


def main():
    if not os.path.isdir(RAW_VIDEOS_DIR):
        print(f"❌ '{RAW_VIDEOS_DIR}' folder not found.")
        return

    rows = []
    with mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for signer in sorted(os.listdir(RAW_VIDEOS_DIR)):
            signer_path = os.path.join(RAW_VIDEOS_DIR, signer)
            if not os.path.isdir(signer_path):
                continue

            for sign_name in sorted(os.listdir(signer_path)):
                sign_path = os.path.join(signer_path, sign_name)
                if not os.path.isdir(sign_path):
                    continue

                for video_file in sorted(os.listdir(sign_path)):
                    if not video_file.lower().endswith((".mp4", ".avi", ".mov")):
                        continue

                    video_path = os.path.join(sign_path, video_file)
                    print(f"Processing: {video_path}")
                    feats = pool_video(video_path, holistic)

                    if feats is None:
                        print(f"⚠️ No frames extracted, skipping: {video_path}")
                        continue

                    row = {"signer": signer, "label": sign_name}
                    for i, val in enumerate(feats):
                        row[f"f{i}"] = val
                    rows.append(row)

    if not rows:
        print("❌ No videos processed. Check your raw_videos folder structure.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Done. {len(df)} videos processed, {df['label'].nunique()} signs found.")
    print(f"Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
