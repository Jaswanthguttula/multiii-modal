"""
Real-Time Multimodal Surveillance Anomaly Detection System
Uses webcam (video) + microphone (audio) for live anomaly detection.
Combines Isolation Forest (audio) + Conv Autoencoder (video) via weighted fusion.
"""

import os
import sys
import json
import time
import threading
import queue
import numpy as np
import cv2
import librosa
import sounddevice as sd
import joblib
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = r'd:\multimodel-surveillence'
IMG_SIZE = 128
AUDIO_DURATION = 3        # seconds of audio per analysis
AUDIO_SAMPLE_RATE = 22050
AUDIO_CHANNELS = 1

# Load fusion config
config_path = os.path.join(BASE_DIR, 'fusion_config.json')
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
    AUDIO_WEIGHT = config.get('audio_weight', 0.4)
    VIDEO_WEIGHT = config.get('video_weight', 0.6)
    THRESHOLD = config.get('threshold', 0.5)
else:
    AUDIO_WEIGHT = 0.4
    VIDEO_WEIGHT = 0.6
    THRESHOLD = 0.5

# ============================================================
# LOAD MODELS
# ============================================================
print("Loading models...")
audio_model = joblib.load(os.path.join(BASE_DIR, 'audio_isolation_forest.pkl'))
audio_scaler = joblib.load(os.path.join(BASE_DIR, 'audio_scaler.pkl'))
video_model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'video_autoencoder.h5'), compile=False)
video_model.compile(optimizer='adam', loss='mse')
print("✅ All models loaded!")

# ============================================================
# FEATURE EXTRACTION
# ============================================================
def extract_audio_features_from_signal(y, sr=AUDIO_SAMPLE_RATE, n_mfcc=40):
    """Extract features from raw audio signal (same pipeline as training)."""
    if len(y) < sr * 0.5:  # Less than 0.5 seconds
        return None
    feat = []
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feat.extend(np.mean(mfccs.T, axis=0)); feat.extend(np.std(mfccs.T, axis=0))
    sc = librosa.feature.spectral_contrast(y=y, sr=sr)
    feat.extend(np.mean(sc.T, axis=0)); feat.extend(np.std(sc.T, axis=0))
    ch = librosa.feature.chroma_stft(y=y, sr=sr)
    feat.extend(np.mean(ch.T, axis=0)); feat.extend(np.std(ch.T, axis=0))
    for fn_name in ['zero_crossing_rate', 'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'rms']:
        fn = getattr(librosa.feature, fn_name)
        r = fn(y=y) if fn_name in ['zero_crossing_rate', 'rms'] else fn(y=y, sr=sr)
        feat.append(np.mean(r)); feat.append(np.std(r))
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)
    feat.extend(np.mean(mel.T, axis=0)); feat.extend(np.std(mel.T, axis=0))
    return np.array(feat)

def get_audio_score(y, sr=AUDIO_SAMPLE_RATE):
    """Get anomaly score from audio signal."""
    feat = extract_audio_features_from_signal(y, sr)
    if feat is None:
        return 0.0
    feat_scaled = audio_scaler.transform(feat.reshape(1, -1))
    raw_score = audio_model.decision_function(feat_scaled)[0]
    score = np.clip(-raw_score, -1, 1)
    return (score + 1) / 2

def get_video_score(frame_gray):
    """Get anomaly score from a grayscale frame."""
    resized = cv2.resize(frame_gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype('float32') / 255.0
    inp = normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    recon = video_model.predict(inp, verbose=0)
    error = np.mean(np.square(inp - recon))
    return np.clip(error / 0.1, 0, 1), error

# ============================================================
# AUDIO CAPTURE THREAD
# ============================================================
class AudioCapture:
    def __init__(self, duration=AUDIO_DURATION, sr=AUDIO_SAMPLE_RATE):
        self.duration = duration
        self.sr = sr
        self.buffer = queue.Queue()
        self.running = False
        self.current_score = 0.0
        self.thread = None

    def _capture_loop(self):
        while self.running:
            try:
                audio = sd.rec(int(self.duration * self.sr), samplerate=self.sr,
                              channels=1, dtype='float32')
                sd.wait()
                y = audio.flatten()
                self.current_score = get_audio_score(y, self.sr)
            except Exception as e:
                print(f"Audio error: {e}")
                self.current_score = 0.0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("🎤 Audio capture started")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("🎤 Audio capture stopped")

# ============================================================
# MAIN REAL-TIME DETECTION LOOP
# ============================================================
def draw_dashboard(frame, video_score, audio_score, fused_score, is_anomaly, fps):
    """Draw the HUD overlay on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent overlay panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 220), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Title
    cv2.putText(frame, "SURVEILLANCE MONITOR", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Status indicator
    status_color = (0, 0, 255) if is_anomaly else (0, 255, 0)
    status_text = "!! ANOMALY DETECTED !!" if is_anomaly else "NORMAL"
    cv2.putText(frame, status_text, (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # Score bars
    bar_x, bar_w = 20, 200
    # Video score bar
    cv2.putText(frame, f"Video: {video_score:.2f}", (bar_x, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.rectangle(frame, (bar_x + 120, 82), (bar_x + 120 + bar_w, 98), (50, 50, 50), -1)
    v_color = (0, 255, 0) if video_score < 0.5 else (0, 165, 255) if video_score < 0.7 else (0, 0, 255)
    cv2.rectangle(frame, (bar_x + 120, 82), (bar_x + 120 + int(bar_w * video_score), 98), v_color, -1)

    # Audio score bar
    cv2.putText(frame, f"Audio: {audio_score:.2f}", (bar_x, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.rectangle(frame, (bar_x + 120, 112), (bar_x + 120 + bar_w, 128), (50, 50, 50), -1)
    a_color = (0, 255, 0) if audio_score < 0.5 else (0, 165, 255) if audio_score < 0.7 else (0, 0, 255)
    cv2.rectangle(frame, (bar_x + 120, 112), (bar_x + 120 + int(bar_w * audio_score), 128), a_color, -1)

    # Fused score bar
    cv2.putText(frame, f"Fused: {fused_score:.2f}", (bar_x, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(frame, (bar_x + 120, 142), (bar_x + 120 + bar_w, 158), (50, 50, 50), -1)
    f_color = (0, 255, 0) if fused_score < THRESHOLD else (0, 0, 255)
    cv2.rectangle(frame, (bar_x + 120, 142), (bar_x + 120 + int(bar_w * min(fused_score, 1.0)), 158), f_color, -1)
    # Threshold marker
    th_x = bar_x + 120 + int(bar_w * THRESHOLD)
    cv2.line(frame, (th_x, 140), (th_x, 160), (255, 255, 255), 2)

    # FPS & info
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    cv2.putText(frame, f"Weights: A={AUDIO_WEIGHT} V={VIDEO_WEIGHT} Th={THRESHOLD}",
                (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    # Border flash on anomaly
    if is_anomaly:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)

    # Bottom bar
    cv2.putText(frame, "Press 'q' to quit | 's' to screenshot | 'r' to record",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    return frame


def run_realtime():
    """Main real-time detection loop."""
    print("\n" + "=" * 60)
    print("  MULTIMODAL REAL-TIME SURVEILLANCE SYSTEM")
    print("=" * 60)
    print(f"  Audio weight: {AUDIO_WEIGHT}")
    print(f"  Video weight: {VIDEO_WEIGHT}")
    print(f"  Threshold:    {THRESHOLD}")
    print("=" * 60)

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam! Exiting.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("📷 Webcam started")

    # Start audio capture
    audio_cap = AudioCapture()
    audio_cap.start()

    prev_time = time.time()
    recording = False
    video_writer = None
    screenshot_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            # Get video anomaly score
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            video_score, video_error = get_video_score(gray)

            # Get audio anomaly score (from background thread)
            audio_score = audio_cap.current_score

            # Fuse scores
            fused_score = AUDIO_WEIGHT * audio_score + VIDEO_WEIGHT * video_score
            is_anomaly = fused_score > THRESHOLD

            # Draw dashboard
            display = draw_dashboard(frame, video_score, audio_score, fused_score, is_anomaly, fps)

            # Record if recording
            if recording and video_writer:
                video_writer.write(display)

            cv2.imshow('Multimodal Surveillance', display)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_count += 1
                fname = os.path.join(BASE_DIR, f'screenshot_{screenshot_count}.png')
                cv2.imwrite(fname, display)
                print(f"📸 Screenshot saved: {fname}")
            elif key == ord('r'):
                if not recording:
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    fname = os.path.join(BASE_DIR, f'recording_{int(time.time())}.avi')
                    video_writer = cv2.VideoWriter(fname, fourcc, 20.0, (640, 480))
                    recording = True
                    print(f"🔴 Recording started: {fname}")
                else:
                    recording = False
                    if video_writer:
                        video_writer.release()
                    print("⏹ Recording stopped")

    finally:
        audio_cap.stop()
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print("System shut down.")


if __name__ == '__main__':
    run_realtime()
