import json
import os

notebook_path = r'd:\multimodel-surveillence\multimodal_fusion.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        source_str = "".join(source)
        
        # 1. Update Model Loading
        if "video_autoencoder.h5" in source_str:
            new_source = []
            new_source.append("BASE = r'd:\\multimodel-surveillence'\n")
            new_source.append("\n")
            new_source.append("# Audio model\n")
            new_source.append("audio_model = joblib.load(os.path.join(BASE, 'audio_isolation_forest.pkl'))\n")
            new_source.append("audio_scaler = joblib.load(os.path.join(BASE, 'audio_scaler.pkl'))\n")
            new_source.append("print('\\u2705 Audio Isolation Forest loaded')\n")
            new_source.append("\n")
            new_source.append("# Video model - Loading the NEW Transfer Learning model (MobileNetV2)\n")
            new_source.append("video_model = tf.keras.models.load_model(os.path.join(BASE, 'video_transfer_model.h5'), compile=False)\n")
            new_source.append("video_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\n")
            new_source.append("print('\\u2705 Video Transfer Learning (MobileNetV2) loaded')\n")
            new_source.append("video_model.summary()")
            cell['source'] = new_source
            
        # 2. Update IMG_SIZE and extract_video_frame
        if "IMG_SIZE = 128" in source_str:
            new_source = []
            new_source.append("IMG_SIZE = 96\n")
            new_source.append("\n")
            new_source.append("def extract_audio_features(file_path, n_mfcc=40):\n")
            new_source.append("    y, sr = librosa.load(file_path, duration=30)\n")
            new_source.append("    feat = []\n")
            new_source.append("    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)\n")
            new_source.append("    feat.extend(np.mean(mfccs.T, axis=0)); feat.extend(np.std(mfccs.T, axis=0))\n")
            new_source.append("    sc = librosa.feature.spectral_contrast(y=y, sr=sr)\n")
            new_source.append("    feat.extend(np.mean(sc.T, axis=0)); feat.extend(np.std(sc.T, axis=0))\n")
            new_source.append("    ch = librosa.feature.chroma_stft(y=y, sr=sr)\n")
            new_source.append("    feat.extend(np.mean(ch.T, axis=0)); feat.extend(np.std(ch.T, axis=0))\n")
            new_source.append("    for fn_name in ['zero_crossing_rate','spectral_centroid','spectral_bandwidth','spectral_rolloff','rms']:\n")
            new_source.append("        fn = getattr(librosa.feature, fn_name)\n")
            new_source.append("        r = fn(y=y) if fn_name in ['zero_crossing_rate','rms'] else fn(y=y, sr=sr)\n")
            new_source.append("        feat.append(np.mean(r)); feat.append(np.std(r))\n")
            new_source.append("    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128), ref=np.max)\n")
            new_source.append("    feat.extend(np.mean(mel.T, axis=0)); feat.extend(np.std(mel.T, axis=0))\n")
            new_source.append("    return np.array(feat)\n")
            new_source.append("\n")
            new_source.append("def extract_video_frame(file_path):\n")
            new_source.append("    img = cv2.imread(file_path)\n")
            new_source.append("    if img is None: return None\n")
            new_source.append("    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # MobileNetV2 uses RGB\n")
            new_source.append("    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))\n")
            new_source.append("    img_batch = np.expand_dims(img.astype('float32'), axis=0)\n")
            new_source.append("    return tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)\n")
            new_source.append("\n")
            new_source.append("print('Feature extraction functions defined for MobileNetV2.')")
            cell['source'] = new_source

        # 3. Update get_video_anomaly_score
        if "def get_video_anomaly_score(frame):" in source_str and "recon" in source_str:
            new_source = []
            new_source.append("def get_video_anomaly_score(preprocessed_frame):\n")
            new_source.append("    # preprocessed_frame is already (1, 96, 96, 3)\n")
            new_source.append("    score = video_model.predict(preprocessed_frame, verbose=0)[0][0]\n")
            new_source.append("    return score, score # Using classification score as the anomaly metric\n")
            new_source.append("\n")
            new_source.append("print('Scoring functions updated for MobileNetV2.')")
            cell['source'] = new_source

        # 4. Update Pie Chart Labels/Colors (optional but nice)
        if "pie([fusion.audio_weight" in source_str:
             cell['source'] = [s.replace('Conv Autoencoder', 'MobileNetV2') for s in cell['source']]

        # 5. Fix Model name strings in Architecture diagram cell
        if "Video (Conv Autoencoder)" in source_str:
             cell['source'] = [s.replace('Video (Conv Autoencoder)', 'Video (MobileNetV2 Transfer)') for s in cell['source']]

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Successfully updated multimodal_fusion.ipynb')
