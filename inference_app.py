"""
Live Demo (UI) for Isolated Sign Recognition — Push-to-Sign version
------------------------------------------------------------
Instead of guessing continuously in the background (which didn't match how
the training videos were recorded), this version has a "Capture Sign" button.
When pressed, it records for exactly RECORD_SECONDS (same length as
record_videos.py) and then classifies that clip — matching the training data
much more closely, which gives far more reliable predictions.

RUN:
    pip install -r requirements.txt
    streamlit run inference_app.py
"""

import streamlit as st
import cv2
import numpy as np
import pickle
import time
import os
import glob
import mediapipe as mp
import pyttsx3

MODEL_PATH = "sign_classifier.pkl"
RAW_VIDEOS_DIR = "raw_videos"
RECORD_SECONDS = 2.5     # MUST match RECORD_SECONDS in record_videos.py
CONFIDENCE_THRESHOLD = 0.35

mp_holistic = mp.solutions.holistic


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


def pool_clip(frame_feats):
    frame_feats = np.array(frame_feats)
    return frame_feats.mean(axis=0)


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["model"]


def list_demo_videos():
    """Find recorded videos under raw_videos/<signer>/<sign>/take_N.mp4"""
    videos = {}
    if not os.path.isdir(RAW_VIDEOS_DIR):
        return videos
    for signer in sorted(os.listdir(RAW_VIDEOS_DIR)):
        signer_path = os.path.join(RAW_VIDEOS_DIR, signer)
        if not os.path.isdir(signer_path):
            continue
        for sign in sorted(os.listdir(signer_path)):
            sign_path = os.path.join(signer_path, sign)
            if not os.path.isdir(sign_path):
                continue
            files = sorted(glob.glob(os.path.join(sign_path, "*.mp4")))
            if files:
                videos.setdefault(sign, []).extend(files)
    return videos


def classify_video_file(video_path, model):
    cap = cv2.VideoCapture(video_path)
    frame_feats = []
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_feats.append(extract_landmarks(results))
    cap.release()

    if len(frame_feats) < 5:
        return None, 0.0

    pooled = pool_clip(frame_feats).reshape(1, -1)
    probs = model.predict_proba(pooled)[0]
    best_idx = np.argmax(probs)
    return model.classes_[best_idx], probs[best_idx]


def main():
    st.set_page_config(page_title="PSL Sign Recognition", layout="wide")
    st.title("🤟 PSL Sign Language Recognition")

    model = load_model()

    if "sentence_tokens" not in st.session_state:
        st.session_state.sentence_tokens = []

    tab_live, tab_demo = st.tabs(["📷 Live Camera", "🎬 Demo Mode (pre-recorded)"])

    with tab_live:
        run_live_tab(model)

    with tab_demo:
        run_demo_tab(model)


def run_live_tab(model):
    st.caption("Position yourself in frame, then press the button and perform ONE sign.")

    col1, col2 = st.columns([2, 1])
    frame_placeholder = col1.empty()
    status_placeholder = col1.empty()
    text_placeholder = col2.empty()

    text_placeholder.markdown(f"### {' '.join(st.session_state.sentence_tokens)}")

    capture = col2.button("📸 Capture Sign (2.5s)", type="primary", key="live_capture")
    c1, c2 = col2.columns(2)
    clear = c1.button("Clear Sentence", key="live_clear")
    speak = c2.button("🔊 Speak", key="live_speak")

    if clear:
        st.session_state.sentence_tokens = []
        text_placeholder.markdown("### ")

    if speak:
        engine = pyttsx3.init()
        engine.say(" ".join(st.session_state.sentence_tokens))
        engine.runAndWait()

    if capture:
        cap = cv2.VideoCapture(0)
        raw_frames = []
        start_time = time.time()

        while time.time() - start_time < RECORD_SECONDS:
            ret, frame = cap.read()
            if not ret:
                break
            raw_frames.append(frame)
            elapsed = time.time() - start_time
            display_frame = frame.copy()
            cv2.putText(display_frame, f"REC {elapsed:.1f}s / {RECORD_SECONDS}s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frame_placeholder.image(display_frame, channels="BGR")

        cap.release()
        status_placeholder.info("Processing...")

        frame_feats = []
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for frame in raw_frames:
                results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_feats.append(extract_landmarks(results))

        if len(frame_feats) < 5:
            status_placeholder.error("Not enough frames captured from the camera. Please try again.")
        else:
            pooled = pool_clip(frame_feats).reshape(1, -1)
            probs = model.predict_proba(pooled)[0]
            best_idx = np.argmax(probs)
            pred_label = model.classes_[best_idx]
            confidence = probs[best_idx]

            if confidence >= CONFIDENCE_THRESHOLD:
                st.session_state.sentence_tokens.append(pred_label)
                status_placeholder.success(f"Detected: {pred_label} ({confidence:.0%})")
            else:
                status_placeholder.warning(f"Not confident enough: {pred_label} ({confidence:.0%}) — sign not added.")

            text_placeholder.markdown(f"### {' '.join(st.session_state.sentence_tokens)}")


def run_demo_tab(model):
    st.caption("Reliable backup: run an already-recorded clip through the same pipeline.")

    videos = list_demo_videos()
    if not videos:
        st.error(f"No videos found in the '{RAW_VIDEOS_DIR}' folder.")
        return

    sign_choice = st.selectbox("Choose a sign", sorted(videos.keys()), key="demo_sign")
    file_choice = st.selectbox("Choose a take", videos[sign_choice], key="demo_file")

    if st.button("▶️ Run Demo", type="primary"):
        st.video(file_choice)
        with st.spinner("Processing..."):
            pred_label, confidence = classify_video_file(file_choice, model)

        if pred_label is None:
            st.error("Could not extract frames from this video.")
        else:
            st.success(f"Detected: {pred_label} ({confidence:.0%})")
            if st.button("➕ Add to sentence", key="demo_add"):
                st.session_state.sentence_tokens.append(pred_label)

    st.markdown(f"### Sentence: {' '.join(st.session_state.sentence_tokens)}")
    d1, d2 = st.columns(2)
    if d1.button("Clear Sentence", key="demo_clear"):
        st.session_state.sentence_tokens = []
    if d2.button("🔊 Speak", key="demo_speak"):
        engine = pyttsx3.init()
        engine.say(" ".join(st.session_state.sentence_tokens))
        engine.runAndWait()


if __name__ == "__main__":
    main()
