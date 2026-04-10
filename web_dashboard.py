"""
Web Dashboard for Multimodal Surveillance System
Flask + SocketIO backend with live camera feed, anomaly detection, and alert system.
Supports EfficientNetV2 + Focal Loss advanced multiclass model.
"""

import os
import sys
import json
import time
import base64
import threading
import logging
import traceback
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import librosa
import sounddevice as sd
import joblib
import tensorflow as tf
import winsound

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ============================================================
# FOCAL LOSS - Custom object for loading advanced model
# ============================================================
@tf.keras.utils.register_keras_serializable()
class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss for handling class imbalance."""

    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        num_classes = tf.shape(y_pred)[-1]
        y_true_one_hot = tf.one_hot(tf.squeeze(y_true), depth=num_classes)

        if self.label_smoothing > 0:
            num_classes_float = tf.cast(num_classes, tf.float32)
            y_true_one_hot = (
                y_true_one_hot * (1 - self.label_smoothing)
                + self.label_smoothing / num_classes_float
            )

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        focal_weight = self.alpha * tf.pow(1 - y_pred, self.gamma)
        focal_loss = focal_weight * cross_entropy

        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "gamma": self.gamma,
                "alpha": self.alpha,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config


# ============================================================
# APP SETUP
# ============================================================
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "surveillance-secret-key"
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    logger=True,
    engineio_logger=True,
)

# ============================================================
# LOGGING
# ============================================================
BASE_DIR = r"d:\multimodel-surveillence"
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "dashboard.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("dashboard")

# ============================================================
# CONFIGURATION
# ============================================================
AUDIO_DURATION = 3
AUDIO_SAMPLE_RATE = 22050

config_path = os.path.join(BASE_DIR, "fusion_config.json")
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
    AUDIO_WEIGHT = config.get("audio_weight", 0.4)
    VIDEO_WEIGHT = config.get("video_weight", 0.6)
    THRESHOLD = config.get("threshold", 0.5)
    VIDEO_MODEL_TYPE = config.get("video_model_type", "transfer_mobilenetv2")
    IMG_SIZE_TRANSFER = config.get("img_size_transfer", 96)
    IMG_SIZE_AUTOENC = config.get("img_size", 128)
    IMG_SIZE_MULTICLASS = config.get(
        "img_size_multiclass", 128
    )  # For multi-class model
else:
    AUDIO_WEIGHT, VIDEO_WEIGHT, THRESHOLD = 0.4, 0.6, 0.5
    VIDEO_MODEL_TYPE = "transfer_mobilenetv2"
    IMG_SIZE_TRANSFER = 96
    IMG_SIZE_MULTICLASS = 128
    IMG_SIZE_AUTOENC = 128

# ============================================================
# LOAD MODELS
# ============================================================
logger.info("Initializing models...")

audio_model = None
audio_scaler = None
try:
    audio_path = os.path.join(BASE_DIR, "audio_isolation_forest.pkl")
    scaler_path = os.path.join(BASE_DIR, "audio_scaler.pkl")
    if os.path.exists(audio_path) and os.path.exists(scaler_path):
        audio_model = joblib.load(audio_path)
        audio_scaler = joblib.load(scaler_path)
        logger.info("[OK] Audio models loaded")
except Exception as e:
    logger.error(f"[ERROR] Audio model error: {e}")

video_model = None
video_model_type = VIDEO_MODEL_TYPE
img_size = IMG_SIZE_TRANSFER

# Class names for multi-class model
CLASS_NAMES = config.get(
    "class_names", ["Normal", "Accident", "Fire", "Robbery", "Violence"]
)

# Try to load multi-class model first (check both .keras and .h5 formats)
multiclass_name = config.get("video_multiclass_model", "video_multiclass_model.keras")
multiclass_path = os.path.join(BASE_DIR, multiclass_name)

# Also check for .keras extension if .h5 not found
if not os.path.exists(multiclass_path):
    alt_name = multiclass_name.replace(".h5", ".keras")
    alt_path = os.path.join(BASE_DIR, alt_name)
    if os.path.exists(alt_path):
        multiclass_path = alt_path
        multiclass_name = alt_name

# Get model architecture from config
MODEL_ARCHITECTURE = config.get("model_architecture", "mobilenetv2")

if os.path.exists(multiclass_path):
    try:
        # Load with custom objects for Focal Loss
        custom_objects = {"FocalLoss": FocalLoss}
        video_model = tf.keras.models.load_model(
            multiclass_path, compile=False, custom_objects=custom_objects
        )
        video_model_type = "multiclass"
        img_size = IMG_SIZE_MULTICLASS  # Use correct size for multiclass model
        logger.info(
            f"[OK] Multi-class Video Model loaded: {multiclass_name} (size: {img_size}, arch: {MODEL_ARCHITECTURE})"
        )
    except Exception as e:
        logger.error(f"[ERROR] Multi-class model load failed: {e}")
        traceback.print_exc()

# Fallback to binary transfer model
if video_model is None:
    transfer_name = config.get("video_model_transfer", "video_transfer_model.h5")
    transfer_path = os.path.join(BASE_DIR, transfer_name)
    if os.path.exists(transfer_path):
        try:
            video_model = tf.keras.models.load_model(transfer_path, compile=False)
            video_model_type = "transfer_mobilenetv2"
            img_size = IMG_SIZE_TRANSFER
            logger.info(f"[OK] Video Transfer Model loaded: {transfer_name}")
        except Exception as e:
            logger.error(f"[ERROR] Transfer model load failed: {e}")

# Fallback to autoencoder
if video_model is None:
    autoenc_name = config.get("video_model", "video_autoencoder.h5")
    autoenc_path = os.path.join(BASE_DIR, autoenc_name)
    if os.path.exists(autoenc_path):
        try:
            video_model = tf.keras.models.load_model(autoenc_path, compile=False)
            video_model_type = "autoencoder"
            img_size = IMG_SIZE_AUTOENC
            logger.info(f"[OK] Fallback to Autoencoder: {autoenc_name}")
        except Exception as e:
            logger.error(f"[ERROR] All video models failed: {e}")

# ============================================================
# VARIABLES
# ============================================================
ALERTS_DIR = os.path.join(BASE_DIR, "alerts")
os.makedirs(ALERTS_DIR, exist_ok=True)
alert_log = deque(maxlen=100)
alert_cooldown = 0
ALERT_COOLDOWN_SECS = 5
sound_enabled = True
logging_enabled = True

# Monitoring states
camera_active = False
audio_active = False
current_audio_score = 0.0
current_video_score = 0.0
current_fused_score = 0.0
score_history = deque(maxlen=60)
system_diagnostics = {"camera_ok": False, "audio_ok": False, "last_error": None}


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def trigger_alert(
    fused_score, audio_score, video_score, frame=None, anomaly_type="Anomaly"
):
    global alert_cooldown
    now = time.time()
    if now - alert_cooldown < ALERT_COOLDOWN_SECS:
        return
    alert_cooldown = now

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Overriding to generic "Anomaly" per user request
    display_anomaly_type = "Anomaly"

    alert_entry = {
        "time": timestamp,
        "fused": round(fused_score, 4),
        "audio": round(audio_score, 4),
        "video": round(video_score, 4),
        "anomaly_type": display_anomaly_type,
        "screenshot": None,
    }

    if frame is not None:
        fname = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(os.path.join(ALERTS_DIR, fname), frame)
        alert_entry["screenshot"] = fname

    alert_log.appendleft(alert_entry)
    socketio.emit("new_alert", alert_entry)
    logger.info(
        f"🚨 ALERT: {display_anomaly_type} detected! (Original prediction: {anomaly_type}) Score: {fused_score:.2f}"
    )

    if sound_enabled:
        try:
            threading.Thread(
                target=lambda: winsound.Beep(1000, 300), daemon=True
            ).start()
        except:
            pass


def get_audio_score(y, sr=AUDIO_SAMPLE_RATE):
    """
    Audio anomaly scoring using Isolation Forest.
    MUST match training feature extraction exactly (40 MFCC means only).
    """
    if audio_model is None:
        return 0.0
    try:
        # Extract 40 MFCCs - EXACTLY matching train_audio.py
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        # Take mean across time - results in 40 features
        feat = np.mean(mfccs.T, axis=0)

        # Reshape for scaler (trained on 40 features)
        feat_scaled = audio_scaler.transform(feat.reshape(1, -1))

        # Get anomaly score from Isolation Forest
        raw = audio_model.decision_function(feat_scaled)[0]

        # Convert to 0-1 range (higher = more anomalous)
        # Isolation Forest: negative = anomaly, positive = normal
        # We invert and normalize to [0, 1]
        return float((np.clip(-raw, -1, 1) + 1) / 2)
    except Exception as e:
        logger.error(f"Audio scoring error: {e}")
        return 0.0


def get_video_score(frame):
    """
    Get video anomaly score and classification.
    Returns: (score, raw_value, anomaly_type)
    - For multiclass: score=max prob of anomaly classes, anomaly_type=class name
    - For binary: score=anomaly prob, anomaly_type='Anomaly' or 'Normal'
    """
    if video_model is None:
        return 0.0, 0.0, "Unknown"
    try:
        if video_model_type == "multiclass":
            # Multi-class classification
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (img_size, img_size))

            # Use correct preprocessing based on model architecture
            if MODEL_ARCHITECTURE == "efficientnetv2":
                norm = tf.keras.applications.efficientnet_v2.preprocess_input(
                    resized.astype("float32")
                )
            else:
                norm = tf.keras.applications.mobilenet_v2.preprocess_input(
                    resized.astype("float32")
                )

            inp = np.expand_dims(norm, axis=0)
            pred = video_model.predict(inp, verbose=0)[0]

            # Get predicted class
            class_idx = int(np.argmax(pred))
            confidence = float(pred[class_idx])
            anomaly_type = (
                CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else "Unknown"
            )

            # Score is 0 for normal, or the confidence of anomaly class
            if class_idx == 0:  # Normal
                score = 1.0 - confidence  # Low score for normal
            else:
                score = confidence  # High score for anomaly

            return score, confidence, anomaly_type

        elif video_model_type == "transfer_mobilenetv2":
            # Binary classification
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (img_size, img_size))
            norm = tf.keras.applications.mobilenet_v2.preprocess_input(
                resized.astype("float32")
            )
            inp = np.expand_dims(norm, axis=0)
            pred = video_model.predict(inp, verbose=0)
            score = float(pred[0][0]) if pred.shape[-1] == 1 else float(1 - pred[0][0])
            anomaly_type = "Anomaly" if score > THRESHOLD else "Normal"
            return score, score, anomaly_type
        else:
            # Autoencoder
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (img_size, img_size))
            norm = resized.astype("float32") / 255.0
            inp = norm.reshape(1, img_size, img_size, 1)
            recon = video_model.predict(inp, verbose=0)
            error = float(np.mean(np.square(inp - recon)))
            score = np.clip(error / 0.1, 0, 1)
            anomaly_type = "Anomaly" if score > THRESHOLD else "Normal"
            return score, error, anomaly_type

    except Exception as e:
        logger.error(f"Video score error: {e}")
        return 0.0, 0.0, "Error"


# ============================================================
# FILE ANALYSIS FUNCTIONS
# ============================================================
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "bmp",
    "gif",
    "mp4",
    "avi",
    "mov",
    "mkv",
    "wav",
    "mp3",
    "flac",
    "ogg",
    "m4a",
}


def extract_audio_features_full(y, sr=AUDIO_SAMPLE_RATE, n_mfcc=40):
    """Full feature extraction matching training pipeline."""
    if len(y) < sr * 0.5:
        return None
    feat = []
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feat.extend(np.mean(mfccs.T, axis=0))
    feat.extend(np.std(mfccs.T, axis=0))
    sc = librosa.feature.spectral_contrast(y=y, sr=sr)
    feat.extend(np.mean(sc.T, axis=0))
    feat.extend(np.std(sc.T, axis=0))
    ch = librosa.feature.chroma_stft(y=y, sr=sr)
    feat.extend(np.mean(ch.T, axis=0))
    feat.extend(np.std(ch.T, axis=0))
    for fn_name in [
        "zero_crossing_rate",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
        "rms",
    ]:
        fn = getattr(librosa.feature, fn_name)
        r = fn(y=y) if fn_name in ["zero_crossing_rate", "rms"] else fn(y=y, sr=sr)
        feat.append(np.mean(r))
        feat.append(np.std(r))
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max
    )
    feat.extend(np.mean(mel.T, axis=0))
    feat.extend(np.std(mel.T, axis=0))
    return np.array(feat)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_type(filename):
    ext = filename.rsplit(".", 1)[1].lower()
    if ext in {"png", "jpg", "jpeg", "bmp", "gif"}:
        return "image"
    elif ext in {"mp4", "avi", "mov", "mkv"}:
        return "video"
    elif ext in {"wav", "mp3", "flac", "ogg", "m4a"}:
        return "audio"
    return None


def analyze_image_file(filepath):
    """Analyze a single image for anomalies."""
    frame = cv2.imread(filepath)
    if frame is None:
        return {"error": "Could not read image"}
    score, raw, anomaly_type = get_video_score(frame)
    is_anomaly = score > THRESHOLD
    thumb = cv2.resize(frame, (320, 240))
    _, buffer = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
    preview_b64 = base64.b64encode(buffer).decode("utf-8")
    return {
        "file_type": "image",
        "score": round(float(score), 4),
        "is_anomaly": bool(is_anomaly),
        "anomaly_type": anomaly_type,
        "threshold": THRESHOLD,
        "model_type": video_model_type,
        "preview": preview_b64,
        "details": {"raw_score": round(float(raw), 6)},
    }


def analyze_video_upload(filepath):
    """Analyze a video file frame-by-frame for anomalies."""
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        return {"error": "Could not open video"}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = total_frames / fps
    max_samples = min(30, total_frames)
    indices = np.linspace(0, max(total_frames - 1, 0), max_samples, dtype=int)
    scores = []
    preview_b64 = None
    anomaly_types_found = {}
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        score, raw, anomaly_type = get_video_score(frame)
        scores.append(
            {
                "frame": int(idx),
                "time": round(int(idx) / fps, 2),
                "score": round(float(score), 4),
                "is_anomaly": bool(score > THRESHOLD),
                "anomaly_type": anomaly_type,
            }
        )
        if score > THRESHOLD and anomaly_type != "Normal":
            anomaly_types_found[anomaly_type] = (
                anomaly_types_found.get(anomaly_type, 0) + 1
            )
        if preview_b64 is None:
            thumb = cv2.resize(frame, (320, 240))
            _, buffer = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
            preview_b64 = base64.b64encode(buffer).decode("utf-8")
    cap.release()
    avg_score = float(np.mean([s["score"] for s in scores])) if scores else 0
    max_score = float(max([s["score"] for s in scores])) if scores else 0
    anomaly_count = sum(1 for s in scores if s["is_anomaly"])
    # Determine dominant anomaly type
    dominant_anomaly = (
        max(anomaly_types_found, key=anomaly_types_found.get)
        if anomaly_types_found
        else "Normal"
    )
    return {
        "file_type": "video",
        "avg_score": round(avg_score, 4),
        "max_score": round(max_score, 4),
        "is_anomaly": bool(max_score > THRESHOLD),
        "anomaly_type": dominant_anomaly,
        "anomaly_types_found": anomaly_types_found,
        "threshold": THRESHOLD,
        "model_type": video_model_type,
        "preview": preview_b64,
        "details": {
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "duration": round(duration, 2),
            "frames_analyzed": len(scores),
            "anomaly_frames": anomaly_count,
            "frame_scores": scores,
        },
    }


def analyze_audio_upload(filepath):
    """Analyze an audio file segment-by-segment for anomalies."""
    if audio_model is None:
        return {"error": "Audio model not loaded"}
    try:
        y, sr = librosa.load(filepath, sr=AUDIO_SAMPLE_RATE)
    except Exception as e:
        return {"error": f"Could not load audio: {str(e)}"}
    duration = len(y) / sr
    segment_length = int(AUDIO_DURATION * sr)
    segments = []
    for start in range(0, len(y), segment_length):
        end = min(start + segment_length, len(y))
        segment = y[start:end]
        if len(segment) < sr * 0.5:
            continue
        feat = extract_audio_features_full(segment, sr)
        if feat is None:
            continue
        feat_scaled = audio_scaler.transform(feat.reshape(1, -1))
        raw = audio_model.decision_function(feat_scaled)[0]
        score = float((np.clip(-raw, -1, 1) + 1) / 2)
        segments.append(
            {
                "start": round(start / sr, 2),
                "end": round(end / sr, 2),
                "score": round(score, 4),
                "is_anomaly": bool(score > THRESHOLD),
            }
        )
    avg_score = float(np.mean([s["score"] for s in segments])) if segments else 0
    max_score = float(max([s["score"] for s in segments])) if segments else 0
    anomaly_count = sum(1 for s in segments if s["is_anomaly"])
    return {
        "file_type": "audio",
        "avg_score": round(avg_score, 4),
        "max_score": round(max_score, 4),
        "is_anomaly": bool(max_score > THRESHOLD),
        "threshold": THRESHOLD,
        "details": {
            "duration": round(duration, 2),
            "segments_analyzed": len(segments),
            "anomaly_segments": anomaly_count,
            "segment_scores": segments,
        },
    }


# ============================================================
# LOOPS
# ============================================================
def audio_loop():
    global current_audio_score, audio_active
    logger.info("🎤 Audio capture thread started")
    audio_active = True
    while audio_active:
        try:
            audio = sd.rec(
                int(AUDIO_DURATION * AUDIO_SAMPLE_RATE),
                samplerate=AUDIO_SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            current_audio_score = get_audio_score(audio.flatten())
            system_diagnostics["audio_ok"] = True
        except Exception as e:
            logger.error(f"Audio Error: {e}")
            system_diagnostics["audio_ok"] = False
            time.sleep(2)
    logger.info("🎤 Audio capture thread stopped")


def camera_loop():
    global camera_active, current_video_score, current_fused_score
    logger.info("📷 Camera capture thread started")

    # Consecutive frame detection for robust alerts
    consecutive_anomaly_frames = 0
    REQUIRED_CONSECUTIVE_FRAMES = 3  # Require 3 consecutive anomaly frames
    last_anomaly_type = None

    cap = None
    # We found CAP_DSHOW works best on this Windows system
    for cam_idx in [0, 1]:
        for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
            logger.info(f"Opening camera {cam_idx} with backend {backend}...")
            cap = cv2.VideoCapture(cam_idx, backend)
            if cap.isOpened():
                ret, test = cap.read()
                if ret and test is not None:
                    logger.info(f"[OK] Camera {cam_idx} opened successfully")
                    system_diagnostics["camera_ok"] = True
                    break
            if cap:
                cap.release()
            cap = None
        if cap:
            break

    if cap is None:
        logger.error("[ERROR] Failed to open any camera")
        system_diagnostics["camera_ok"] = False
        socketio.emit("camera_error", {"message": "Could not open camera"})
        camera_active = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera_active = True

    # Motion detection to prevent false positives when nothing is happening
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=25, detectShadows=False
    )

    # Simple frame differencing as fallback
    ret, prev_frame = cap.read()
    if ret:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    while camera_active:
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.warning("Failed to read frame")
            time.sleep(0.1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Calculate absolute difference between current frame and previous frame
        # If prev_gray is not initialized properly, default to some motion
        try:
            frame_diff = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)

            motion_ratio = np.count_nonzero(thresh) / (frame.shape[0] * frame.shape[1])
            prev_gray = gray
        except:
            motion_ratio = 1.0  # Force inference if diff fails

        if (
            motion_ratio < 0.05
        ):  # increased to 5% of pixels changed to filter out breathing/minor movements
            current_video_score, raw_confidence, anomaly_type = 0.0, 1.0, "Normal"
        else:
            # Get video score AND anomaly type
            current_video_score, raw_confidence, anomaly_type = get_video_score(frame)
        fused = float(
            AUDIO_WEIGHT * current_audio_score + VIDEO_WEIGHT * current_video_score
        )
        current_fused_score = fused

        # IMPROVED ANOMALY DETECTION:
        # Variable thresholds based on the anomaly type
        # Fire/Violence are distinct, so they need high confidence (to stop false alarms)
        # Accident/Robbery are subtle in still frames, so we lower their thresholds.
        anomaly_thresholds = {
            "Fire": {"conf": 0.85, "vid": 0.70},
            "Violence": {"conf": 0.85, "vid": 0.70},
            "Accident": {"conf": 0.50, "vid": 0.45},
            "Robbery": {"conf": 0.50, "vid": 0.45},
        }

        # Get thresholds for current detected type, default to strict if unknown
        th_val = anomaly_thresholds.get(anomaly_type, {"conf": 0.85, "vid": 0.70})

        frame_is_anomaly = (
            fused > THRESHOLD
            and current_video_score >= th_val["vid"]
            and anomaly_type != "Normal"
            and raw_confidence >= th_val["conf"]
        )

        # Track consecutive anomaly frames for robust detection
        if frame_is_anomaly and anomaly_type == last_anomaly_type:
            consecutive_anomaly_frames += 1
        elif frame_is_anomaly:
            # New anomaly type, reset counter
            consecutive_anomaly_frames = 1
            last_anomaly_type = anomaly_type
        else:
            # Not an anomaly, reset
            consecutive_anomaly_frames = 0
            last_anomaly_type = None

        # Only trigger alert if anomaly persists for multiple frames
        is_anomaly = (
            frame_is_anomaly
            and consecutive_anomaly_frames >= REQUIRED_CONSECUTIVE_FRAMES
        )

        score_history.append(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "audio": round(current_audio_score, 4),
                "video": round(current_video_score, 4),
                "fused": round(fused, 4),
                "anomaly_type": anomaly_type,
            }
        )

        # Trigger alert only for real anomalies (Accident, Fire, Robbery, Violence)
        if is_anomaly:
            trigger_alert(
                fused, current_audio_score, current_video_score, frame, anomaly_type
            )
            # Reset counter after alert to avoid duplicate alerts
            consecutive_anomaly_frames = 0

        # Encode and emit
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_b64 = base64.b64encode(buffer).decode("utf-8")

        socketio.emit(
            "video_frame",
            {
                "image": frame_b64,
                "audio_score": round(current_audio_score, 4),
                "video_score": round(current_video_score, 4),
                "fused_score": round(fused, 4),
                "is_anomaly": frame_is_anomaly,  # Show current frame status
                "anomaly_type": anomaly_type,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            },
        )
        socketio.sleep(0.05)

    cap.release()
    logger.info("📷 Camera thread stopped")


# ============================================================
# SERVER
# ============================================================
@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/config", methods=["GET", "POST"])
def handle_config():
    global AUDIO_WEIGHT, VIDEO_WEIGHT, THRESHOLD, sound_enabled, logging_enabled
    if request.method == "POST":
        data = request.json
        AUDIO_WEIGHT = float(data.get("audio_weight", AUDIO_WEIGHT))
        VIDEO_WEIGHT = float(data.get("video_weight", VIDEO_WEIGHT))
        THRESHOLD = float(data.get("threshold", THRESHOLD))
        sound_enabled = bool(data.get("sound_enabled", sound_enabled))
        logging_enabled = bool(data.get("logging_enabled", logging_enabled))
        return jsonify({"status": "ok"})
    return jsonify(
        {
            "audio_weight": AUDIO_WEIGHT,
            "video_weight": VIDEO_WEIGHT,
            "threshold": THRESHOLD,
            "sound_enabled": sound_enabled,
            "logging_enabled": logging_enabled,
            "video_model_type": video_model_type,
        }
    )


@app.route("/api/diagnostics")
def get_diagnostics():
    return jsonify(
        {
            "camera_ok": system_diagnostics["camera_ok"],
            "audio_ok": system_diagnostics["audio_ok"],
            "video_model_loaded": video_model is not None,
            "audio_model_loaded": audio_model is not None,
            "monitoring": camera_active,
            "video_model_type": video_model_type,
        }
    )


@app.route("/api/alerts/clear", methods=["POST"])
def clear_alerts():
    alert_log.clear()
    return jsonify({"status": "cleared"})


@app.route("/api/analyze", methods=["POST"])
def analyze_file():
    """Upload and analyze a file (image, video, or audio) for anomalies."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify(
            {"error": "Unsupported file type. Supported: images, videos, audio."}
        ), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_DIR, f"{int(time.time())}_{filename}")
    file.save(filepath)
    file_type = get_file_type(filename)
    try:
        if file_type == "image":
            result = analyze_image_file(filepath)
        elif file_type == "video":
            result = analyze_video_upload(filepath)
        elif file_type == "audio":
            result = analyze_audio_upload(filepath)
        else:
            result = {"error": "Unknown file type"}
        if "error" in result:
            return jsonify(result), 400
        logger.info(
            f"File analyzed: {filename} ({file_type}) - Anomaly: {result.get('is_anomaly')}"
        )
        return jsonify(result)
    except Exception as e:
        logger.error(f"Analysis error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(filepath)
        except:
            pass


@socketio.on("start_monitor")
def start_monitor():
    global camera_active, audio_active
    if not camera_active:
        socketio.start_background_task(camera_loop)
        socketio.start_background_task(audio_loop)
        emit("status", {"message": "Monitoring started"})
    else:
        emit("status", {"message": "Already monitoring"})


@socketio.on("stop_monitor")
def stop_monitor():
    global camera_active, audio_active
    camera_active = False
    audio_active = False
    emit("status", {"message": "Monitoring stopped"})


if __name__ == "__main__":
    logger.info("Starting Web Dashboard on http://localhost:5000")
    socketio.run(
        app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True
    )
