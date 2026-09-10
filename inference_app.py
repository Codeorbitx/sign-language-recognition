"""
Live Demo (UI) for Isolated Sign Recognition — Push-to-Sign version
------------------------------------------------------------
CLOUD-COMPATIBLE VERSION
------------------------------------------------------------
This version replaces two things that only work on your own computer with
versions that work when this app is deployed to a cloud server:

1. Camera: cv2.VideoCapture(0) -> streamlit-webrtc
   streamlit-webrtc streams video from the VIEWER'S browser (their own
   webcam) to this script, instead of trying to open a camera that is
   physically attached to the server (which doesn't exist).

2. Speech: pyttsx3 -> gTTS + st.audio
   pyttsx3 speaks through the SERVER's speakers (which don't exist on a
   cloud machine). gTTS instead generates an mp3 file, which is sent to
   the browser and played through the VIEWER's own speakers.

Everything else (your model, feature extraction, mediapipe logic) is
UNCHANGED.

RUN LOCALLY:
    pip install -r requirements.txt
    streamlit run inference_app.py

DEPLOY:
    Make sure requirements.txt includes (add these if missing):
        streamlit-webrtc
        av
        gTTS
        opencv-python-headless   (instead of opencv-python, for servers)
"""

import io
import threading
import time
from collections import deque

import streamlit as st
import cv2
import numpy as np
import pickle
import os
import glob
import mediapipe as mp
from gtts import gTTS
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

MODEL_PATH = "sign_classifier.pkl"
RAW_VIDEOS_DIR = "raw_videos"
RECORD_SECONDS = 2.5     # MUST match RECORD_SECONDS in record_videos.py
CONFIDENCE_THRESHOLD = 0.35
ASSUMED_FPS = 24         # buffer sizing only; actual frames used = whatever arrives

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
    # Unchanged: this reads an already-recorded FILE, not a live camera,
    # so cv2.VideoCapture(path) is fine on a server.
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


def speak_text(text):
    """Generate speech audio in the cloud and play it through the VIEWER's
    browser speakers (instead of pyttsx3, which needs a local sound device)."""
    text = text.strip()
    if not text:
        st.warning("Nothing to speak yet — capture a sign first.")
        return
    try:
        tts = gTTS(text=text, lang="en")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        st.audio(buf, format="audio/mp3")
    except Exception as e:
        st.error(f"Could not generate speech: {e}")


class CameraBuffer(VideoProcessorBase):
    """Continuously receives frames from the viewer's browser webcam.
    While `recording` is True, frames are copied into a buffer so the
    main thread can grab them after RECORD_SECONDS."""

    def __init__(self):
        self.frames = deque(maxlen=int(RECORD_SECONDS * ASSUMED_FPS) + 30)
        self.recording = False
        self.lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if self.recording:
            with self.lock:
                self.frames.append(img.copy())
        return av.VideoFrame.from_ndarray(img, format="bgr24")


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
    st.caption(
        "Click 'START' below to allow camera access in your browser, "
        "position yourself in frame, then press Capture and perform ONE sign."
    )

    col1, col2 = st.columns([2, 1])
    text_placeholder = col2.empty()
    text_placeholder.markdown(f"### {' '.join(st.session_state.sentence_tokens)}")

    with col1:
        ctx = webrtc_streamer(
            key="live-sign",
            video_processor_factory=CameraBuffer,
            media_stream_constraints={"video": True, "audio": False},
        )
        status_placeholder = st.empty()

    capture = col2.button("📸 Capture Sign (2.5s)", type="primary", key="live_capture")
    c1, c2 = col2.columns(2)
    clear = c1.button("Clear Sentence", key="live_clear")
    speak = c2.button("🔊 Speak", key="live_speak")

    if clear:
        st.session_state.sentence_tokens = []
        text_placeholder.markdown("### ")

    if speak:
        speak_text(" ".join(st.session_state.sentence_tokens))

    if capture:
        if ctx.video_processor is None:
            status_placeholder.warning("Please click START above to enable your camera first.")
            return

        status_placeholder.info(f"Recording for {RECORD_SECONDS}s... perform your sign now.")

        with ctx.video_processor.lock:
            ctx.video_processor.frames.clear()
        ctx.video_processor.recording = True
        time.sleep(RECORD_SECONDS)
        ctx.video_processor.recording = False

        with ctx.video_processor.lock:
            raw_frames = list(ctx.video_processor.frames)

        status_placeholder.info("Processing...")

        frame_feats = []
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for frame in raw_frames:
                results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_feats.append(extract_landmarks(results))

        if len(frame_feats) < 5:
            status_placeholder.error(
                "Not enough frames captured from the camera. "
                "Make sure your camera is started (green 'LIVE' indicator) and try again."
            )
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
                status_placeholder.warning(
                    f"Not confident enough: {pred_label} ({confidence:.0%}) — sign not added."
                )

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
        speak_text(" ".join(st.session_state.sentence_tokens))


if __name__ == "__main__":
    main()
