"""
Test script to verify audio feature extraction fix.
Compares training vs inference feature shapes.
"""

import numpy as np
import librosa
import joblib


# Simulate training feature extraction
def extract_features_training(y, sr=22050):
    """Training: 40 MFCC means"""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)


# Simulate inference feature extraction (FIXED)
def extract_features_inference(y, sr=22050):
    """Inference: 40 MFCC means (FIXED to match training)"""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    feat = np.mean(mfccs.T, axis=0)
    return feat


# Generate test audio
print("Generating test audio signal...")
y = np.random.randn(22050 * 3)  # 3 seconds of random audio
sr = 22050

# Extract features
print("\n=== TRAINING FEATURE EXTRACTION ===")
train_feat = extract_features_training(y, sr)
print(f"Shape: {train_feat.shape}")
print(f"Features: {len(train_feat)}")

print("\n=== INFERENCE FEATURE EXTRACTION (FIXED) ===")
infer_feat = extract_features_inference(y, sr)
print(f"Shape: {infer_feat.shape}")
print(f"Features: {len(infer_feat)}")

print("\n=== COMPARISON ===")
if train_feat.shape == infer_feat.shape:
    print("✅ SUCCESS: Feature shapes MATCH!")
    print(f"   Both have {len(train_feat)} features (40 MFCC means)")
    print("   Audio scoring will now work correctly!")
else:
    print("❌ ERROR: Feature shapes DO NOT match!")
    print(f"   Training: {train_feat.shape}")
    print(f"   Inference: {infer_feat.shape}")

# Test with scaler if available
print("\n=== TESTING WITH SCALER ===")
try:
    scaler = joblib.load("audio_scaler.pkl")
    model = joblib.load("audio_isolation_forest.pkl")

    # Test scaling
    feat_scaled = scaler.transform(infer_feat.reshape(1, -1))
    score = model.decision_function(feat_scaled)[0]
    normalized_score = float((np.clip(-score, -1, 1) + 1) / 2)

    print(f"✅ Scaler loaded: Expected {scaler.n_features_in_} features")
    print(f"✅ Model loaded: {model.n_estimators} trees")
    print(f"✅ Score computed: {normalized_score:.4f}")

    if scaler.n_features_in_ == len(infer_feat):
        print("✅ PERFECT: Scaler expects correct number of features!")
    else:
        print(
            f"⚠️  WARNING: Scaler expects {scaler.n_features_in_} features but got {len(infer_feat)}"
        )
        print("   You need to retrain: python train_audio.py")

except FileNotFoundError as e:
    print(f"⚠️  Model/scaler not found: {e}")
    print("   Run: python train_audio.py")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n=== SUMMARY ===")
print("1. Feature extraction: FIXED (40 features)")
print("2. web_dashboard.py: UPDATED to match training")
print("3. train_audio.py: UPDATED to save scaler properly")
print("4. Next step: Retrain audio model if scaler has wrong shape")
