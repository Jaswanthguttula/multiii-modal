import os
import numpy as np
import librosa
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib


def extract_features(file_path):
    """
    Extract audio features matching inference pipeline.
    Returns 40 MFCC means.
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, duration=30)
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        # Taking the mean of MFCCs over time (40 features)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def train_audio_model(data_dir):
    features = []
    labels = []  # 1 for normal, -1 for anomaly

    normal_dir = os.path.join(data_dir, "normal")
    anomaly_dir = os.path.join(data_dir, "anomoly")

    print("Loading normal audio files...")
    if os.path.exists(normal_dir):
        for file in os.listdir(normal_dir):
            if file.endswith(".wav"):
                feat = extract_features(os.path.join(normal_dir, file))
                if feat is not None:
                    features.append(feat)
                    labels.append(1)

    print("Loading anomaly audio files...")
    if os.path.exists(anomaly_dir):
        for file in os.listdir(anomaly_dir):
            if file.endswith(".wav"):
                feat = extract_features(os.path.join(anomaly_dir, file))
                if feat is not None:
                    features.append(feat)
                    labels.append(-1)

    if not features:
        print("No audio data found.")
        return

    X = np.array(features)
    print(f"Feature shape: {X.shape} (should be (n_samples, 40))")

    # ===== IMPORTANT: Scale features before training =====
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Isolation Forest is typically trained on "normal" data only in a semi-supervised way,
    # but since we want to "test and train", we can use it to see if it catches the anomalies.
    # However, for a quality check, we can train on all and see the scores.

    print(f"Training Isolation Forest on {len(X)} samples...")
    # contamination is the proportion of outliers in the data set.
    contamination = labels.count(-1) / len(labels) if labels.count(-1) > 0 else 0.1
    if contamination == 0:
        contamination = 0.01

    print(f"Contamination rate: {contamination:.2%}")

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,  # More trees for better accuracy
        max_samples="auto",
        bootstrap=False,
    )
    model.fit(X_scaled)

    # Test on training data (sanity check)
    preds = model.predict(X_scaled)
    accuracy = np.mean(preds == labels)
    print(f"Audio Dataset Alignment (Accuracy): {accuracy * 100:.2f}%")

    # Show anomaly scores for debugging
    scores = model.decision_function(X_scaled)
    print(
        f"Anomaly scores - Normal mean: {np.mean(scores[np.array(labels) == 1]):.3f}, "
        f"Anomaly mean: {np.mean(scores[np.array(labels) == -1]):.3f}"
    )

    # Save the model AND scaler
    joblib.dump(model, "audio_isolation_forest.pkl")
    joblib.dump(scaler, "audio_scaler.pkl")
    print("✅ Audio model saved as audio_isolation_forest.pkl")
    print("✅ Audio scaler saved as audio_scaler.pkl")
    print(f"   Features: 40 MFCC means")


if __name__ == "__main__":
    audio_path = "d:\\multimodel-surveillence\\DATASET\\dataset.audio"
    train_audio_model(audio_path)
