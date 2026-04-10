# Audio Feature Mismatch Fix

## Problem Identified

The audio anomaly detection was producing a constant score of ~0.55 because of a **feature extraction mismatch** between training and inference:

### Before Fix:

**Training** (`train_audio.py`):
```python
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
feat = np.mean(mfccs.T, axis=0)  # 40 features
# NO scaler was saved!
# Model trained on RAW 40 features
```

**Inference** (`web_dashboard.py`):
```python
# Extracted 40 MFCC means + 40 MFCC stds + 12 chroma means + 12 chroma stds
# = 104 features
# Then padded to 384 features with zeros
# Applied scaler trained on different dimension → WRONG!
```

**Result**: 
- Scaler expected 40 features (or unknown dimension)
- Inference provided 384 features with 280 zeros
- Model always produced ~0.55 score (meaningless)

---

## Solution Applied

### 1. Fixed `web_dashboard.py` - `get_audio_score()` function

**Changed from:**
```python
def get_audio_score(y, sr=AUDIO_SAMPLE_RATE):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    feat = []
    feat.extend(np.mean(mfccs.T, axis=0))  # 40 features
    feat.extend(np.std(mfccs.T, axis=0))   # 40 features
    ch = librosa.feature.chroma_stft(y=y, sr=sr)
    feat.extend(np.mean(ch.T, axis=0))     # 12 features
    feat.extend(np.std(ch.T, axis=0))      # 12 features
    # = 104 features
    while len(feat) < 384:
        feat.append(0)  # Pad to 384 with zeros
    feat_scaled = audio_scaler.transform(np.array(feat[:384]).reshape(1, -1))
    # WRONG DIMENSION!
```

**Changed to:**
```python
def get_audio_score(y, sr=AUDIO_SAMPLE_RATE):
    # Extract 40 MFCCs - EXACTLY matching train_audio.py
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # Take mean across time - results in 40 features
    feat = np.mean(mfccs.T, axis=0)
    
    # Reshape for scaler (trained on 40 features)
    feat_scaled = audio_scaler.transform(feat.reshape(1, -1))
    # NOW CORRECT!
```

### 2. Updated `train_audio.py` to include scaler

**Added:**
```python
from sklearn.preprocessing import StandardScaler

# Scale features before training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model on scaled features
model = IsolationForest(...)
model.fit(X_scaled)

# Save both model AND scaler
joblib.dump(model, "audio_isolation_forest.pkl")
joblib.dump(scaler, "audio_scaler.pkl")  # NEW!
```

---

## How to Apply the Fix

### Step 1: Retrain the audio model
```bash
cd d:\multimodel-surveillence
python train_audio.py
```

This will:
- Extract 40 MFCC features from audio files
- Scale features properly with StandardScaler
- Train Isolation Forest on scaled features
- Save both `audio_isolation_forest.pkl` and `audio_scaler.pkl`

### Step 2: Restart the dashboard
```bash
python web_dashboard.py
```

The dashboard will now:
- Load the new scaler and model
- Extract exactly 40 features during inference
- Apply scaler correctly (40 → 40 features)
- Produce meaningful audio scores (not stuck at 0.55)

### Step 3: Verify the fix
```bash
python test_audio_fix.py
```

This test script will:
- Compare training vs inference feature shapes
- Verify scaler expects correct number of features
- Test the full scoring pipeline

---

## Expected Results After Fix

### Before:
- Audio score: Always ~0.55 (meaningless)
- Audio model: Not contributing to detection

### After:
- Audio score: Dynamic range 0.0 - 1.0 based on audio content
- Lower scores (0.0-0.3): Normal audio
- Medium scores (0.3-0.6): Potentially anomalous
- High scores (0.6-1.0): Strong anomaly detected

---

## Technical Details

### Feature Extraction Pipeline:

1. **Load audio**: 3 seconds at 16 kHz
2. **Extract MFCCs**: 40 coefficients
3. **Aggregate**: Mean across time dimension
4. **Result**: 40 features (one per MFCC coefficient)

### Isolation Forest Scoring:

1. **Input**: 40 features (scaled)
2. **Model**: Isolation Forest with 200 trees
3. **Output**: Decision function score (negative = anomaly)
4. **Normalization**: Convert to [0, 1] range where 1 = high anomaly

### Files Modified:

- ✅ `web_dashboard.py` - Lines 281-306 (get_audio_score function)
- ✅ `train_audio.py` - Complete rewrite with proper scaling
- ✅ `test_audio_fix.py` - New test script to verify fix

### Files to Regenerate:

- ⚠️ `audio_isolation_forest.pkl` - Retrain with `python train_audio.py`
- ⚠️ `audio_scaler.pkl` - Retrain with `python train_audio.py`

---

## Troubleshooting

### If audio score is still stuck:

1. **Check scaler shape**:
   ```python
   import joblib
   scaler = joblib.load("audio_scaler.pkl")
   print(f"Scaler expects {scaler.n_features_in_} features")
   # Should print: "Scaler expects 40 features"
   ```

2. **Verify feature extraction**:
   ```bash
   python test_audio_fix.py
   ```
   Should show: "✅ SUCCESS: Feature shapes MATCH!"

3. **Check audio dataset**:
   - Ensure `DATASET/dataset.audio/normal/` has WAV files
   - Ensure `DATASET/dataset.audio/anomoly/` has WAV files
   - If dataset is empty, audio model won't be meaningful

### If retraining fails:

- **No audio data found**: Add audio samples to dataset
- **Import error**: Install sklearn: `pip install scikit-learn`
- **Librosa error**: Update librosa: `pip install librosa --upgrade`

---

## Summary

✅ **Fixed**: Feature extraction now matches between training and inference (40 features)
✅ **Added**: Proper StandardScaler saving in training pipeline  
✅ **Updated**: Dashboard inference to use exact same features as training  
⚠️ **Action Required**: Retrain audio model with `python train_audio.py`

After retraining, audio scoring will work correctly and contribute meaningfully to anomaly detection!
