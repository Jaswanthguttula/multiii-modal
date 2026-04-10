import os, cv2, numpy as np, warnings, json
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils import class_weight

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
VIDEO_DATASET = r'd:\multimodel-surveillence\DATASET\dataset.videos'
MODEL_SAVE_PATH = 'video_transfer_model.h5'
IMG_SIZE = 96
BATCH_SIZE = 16
EPOCHS_PHASE1 = 30
EPOCHS_PHASE2 = 40

def dataset_audit(acc, f1, total_samples):
    print("\n" + "="*50)
    print("          FINAL DATASET AUDIT REPORT          ")
    print("="*50)
    print(f"Total Image Samples: {total_samples}")
    print(f"Model accuracy:      {acc:.4f}")
    print(f"Model F1-Score:      {f1:.4f}")
    print("-" * 50)
    
    status = ""
    advice = ""
    
    if acc > 0.92 and f1 > 0.90:
        status = "EXCELLENT (READY)"
        advice = "The dataset is very robust. The model has learned highly distinct features for anomalies."
    elif acc > 0.85 and f1 > 0.80:
        status = "GOOD (STABLE)"
        advice = "Dataset is working well. To reach 'Excellent', add more edge cases (shadows/night) to 'Normal' category."
    elif total_samples < 350:
        status = "POOR (DATA STARVED)"
        advice = "Sample size is below the recommended threshold (500+). Deep learning may struggle with unexpected real-world scenes."
    else:
        status = "UNSTRICT (IMBALANCED)"
        advice = "High accuracy but low F1-score detected. Check if your anomaly classes are drowning out the normal samples or vice versa."
        
    print(f"DATASET HEALTH: {status}")
    print(f"EXPERT ADVICE:  {advice}")
    print("="*50 + "\n")

# --- 1. DATA LOADING ---
def load_data(data_root):
    frames, labels = [], []
    for cat_name, cat_label in [('normal', 0), ('anomoly', 1)]:
        cat_dir = os.path.join(data_root, cat_name)
        if not os.path.exists(cat_dir): continue
        count = 0
        for root, _, files in os.walk(cat_dir):
            for file in files:
                if not file.lower().endswith(('.png', '.jpg', '.jpeg')): continue
                img = cv2.imread(os.path.join(root, file))
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                frames.append(img.astype('float32'))
                labels.append(cat_label)
                count += 1
        print(f"  {cat_name}: {count} samples")
    return np.array(frames), np.array(labels)

print("Loading dataset...")
X, y = load_data(VIDEO_DATASET)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train = tf.keras.applications.mobilenet_v2.preprocess_input(X_train_raw)
X_test = tf.keras.applications.mobilenet_v2.preprocess_input(X_test_raw)

weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(weights))

# --- 2. MODEL BUILDING ---
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.1)
])

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# --- 3. TRAINING ---
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)

print("\nPhase 1: Training Classification Head...")
model.fit(X_train, y_train, epochs=EPOCHS_PHASE1, batch_size=BATCH_SIZE, validation_split=0.2, class_weight=class_weight_dict, callbacks=[early_stop, reduce_lr])

print("\nPhase 2: Fine-Tuning Top Layers...")
base_model.trainable = True
for layer in base_model.layers[:-20]: layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=EPOCHS_PHASE2, batch_size=BATCH_SIZE, validation_split=0.2, class_weight=class_weight_dict, callbacks=[early_stop, reduce_lr])

# --- 4. EVALUATION ---
print("\nEvaluating on Test Set...")
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(classification_report(y_test, y_pred))
dataset_audit(acc, f1, len(X))

model.save(MODEL_SAVE_PATH)
print(f"SUCCESS: Model saved to {MODEL_SAVE_PATH}")
