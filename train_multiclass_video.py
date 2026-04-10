"""
IMPROVED Multi-Class Video Anomaly Classification Training Script
Classifies: Normal, Accident, Fire, Robbery, Violence
Features: Data augmentation, class balancing, confusion matrix, better training.
"""

import os
import cv2
import json
import numpy as np
import warnings
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.utils import class_weight
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# --- CONFIGURATION ---
BASE_DIR = r"d:\multimodel-surveillence"
VIDEO_DATASET = os.path.join(BASE_DIR, "DATASET", "dataset.videos")
MODEL_SAVE_PATH = os.path.join(
    BASE_DIR, "video_multiclass_model.keras"
)  # Use .keras format
CONFIG_PATH = os.path.join(BASE_DIR, "fusion_config.json")

IMG_SIZE = 128  # Increased for better feature extraction
BATCH_SIZE = 8  # Smaller batch for better generalization
EPOCHS_PHASE1 = 50
EPOCHS_PHASE2 = 80

# Class mapping
CLASS_MAPPING = {
    "normal": 0,
    "accidents": 1,
    "fire explotion": 2,
    "robbery": 3,
    "violence": 4,
}

CLASS_NAMES = ["Normal", "Accident", "Fire", "Robbery", "Violence"]


def augment_image(img):
    """Apply random augmentations to a single image."""
    augmented = img.copy()

    # Random horizontal flip
    if np.random.random() > 0.5:
        augmented = np.fliplr(augmented)

    # Random rotation (-20 to +20 degrees)
    angle = np.random.uniform(-20, 20)
    h, w = augmented.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    augmented = cv2.warpAffine(
        augmented.astype(np.uint8), M, (w, h), borderMode=cv2.BORDER_REFLECT
    )

    # Random brightness
    brightness = np.random.uniform(0.6, 1.4)
    augmented = np.clip(augmented * brightness, 0, 255)

    # Random contrast
    if np.random.random() > 0.5:
        factor = np.random.uniform(0.8, 1.2)
        mean = np.mean(augmented, axis=(0, 1), keepdims=True)
        augmented = np.clip((augmented - mean) * factor + mean, 0, 255)

    # Random zoom/crop
    if np.random.random() > 0.5:
        zoom = np.random.uniform(0.8, 1.0)
        crop_size = int(IMG_SIZE * zoom)
        start = (IMG_SIZE - crop_size) // 2
        cropped = augmented[start : start + crop_size, start : start + crop_size]
        augmented = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))

    # Random noise
    if np.random.random() > 0.7:
        noise = np.random.normal(0, 8, augmented.shape)
        augmented = np.clip(augmented + noise, 0, 255)

    return augmented.astype("float32")


def load_and_augment_data(data_root, target_per_class=150):
    """Load dataset and augment to balance classes."""
    class_frames = {i: [] for i in range(len(CLASS_NAMES))}

    print("\n📁 Loading original images...")

    # Load Normal samples
    normal_dir = os.path.join(data_root, "normal")
    if os.path.exists(normal_dir):
        for root, _, files in os.walk(normal_dir):
            for file in files:
                if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                img = cv2.imread(os.path.join(root, file))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                class_frames[0].append(img.astype("float32"))

    # Load Anomaly samples
    anomaly_dir = os.path.join(data_root, "anomoly")
    if os.path.exists(anomaly_dir):
        for folder_name in os.listdir(anomaly_dir):
            folder_path = os.path.join(anomaly_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            class_idx = CLASS_MAPPING.get(folder_name.lower())
            if class_idx is None:
                continue

            for file in os.listdir(folder_path):
                if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                img = cv2.imread(os.path.join(folder_path, file))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                class_frames[class_idx].append(img.astype("float32"))

    # Print original distribution
    print("\n📊 Original Distribution:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {len(class_frames[i])} images")

    # Augment each class to target_per_class
    print(f"\n🔄 Augmenting each class to {target_per_class} samples...")

    frames, labels = [], []
    for class_idx in range(len(CLASS_NAMES)):
        original = class_frames[class_idx]
        if len(original) == 0:
            print(f"  ⚠️ No images for {CLASS_NAMES[class_idx]}")
            continue

        # Add all originals
        for img in original:
            frames.append(img)
            labels.append(class_idx)

        # Augment to reach target
        current = len(original)
        while current < target_per_class:
            # Pick random original image to augment
            idx = np.random.randint(0, len(original))
            aug_img = augment_image(original[idx])
            frames.append(aug_img)
            labels.append(class_idx)
            current += 1

        print(f"  {CLASS_NAMES[class_idx]}: {len(original)} -> {current}")

    print(f"\n✅ Total samples: {len(frames)}")
    return np.array(frames), np.array(labels)


def build_improved_model(num_classes=5):
    """Build improved model with deeper classification head."""

    # Strong data augmentation during training
    data_augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.25),
            layers.RandomZoom(0.25),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.2),
            layers.RandomTranslation(0.15, 0.15),
        ],
        name="augmentation",
    )

    # Base model
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    # Build model with deeper head
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    # Deeper classification head with BatchNorm
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs), base_model


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.title("Confusion Matrix - Anomaly Classification", fontsize=14)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"\n📊 Confusion matrix saved: {save_path}")

    # Print per-class accuracy
    print("\n📈 Per-Class Accuracy:")
    print("-" * 50)
    for i, name in enumerate(CLASS_NAMES):
        total = cm[i].sum()
        correct = cm[i][i]
        acc = correct / total if total > 0 else 0
        # Show what it gets confused with
        confused_with = []
        for j, conf_name in enumerate(CLASS_NAMES):
            if i != j and cm[i][j] > 0:
                confused_with.append(f"{conf_name}:{cm[i][j]}")
        conf_str = ", ".join(confused_with) if confused_with else "none"
        print(f"  {name}: {acc:.1%} ({correct}/{total}) | Confused with: {conf_str}")


def train_model():
    """Main improved training function."""
    print("=" * 65)
    print("  IMPROVED MULTI-CLASS ANOMALY CLASSIFICATION TRAINING")
    print("=" * 65)

    # Load and augment data
    X, y = load_and_augment_data(VIDEO_DATASET, target_per_class=150)

    if len(X) == 0:
        print("❌ No data found!")
        return

    # Split: 70% train, 15% val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.18, random_state=42, stratify=y_temp
    )

    print(f"\n📊 Data Split:")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Preprocess
    X_train = tf.keras.applications.mobilenet_v2.preprocess_input(X_train.copy())
    X_val = tf.keras.applications.mobilenet_v2.preprocess_input(X_val.copy())
    X_test = tf.keras.applications.mobilenet_v2.preprocess_input(X_test.copy())

    # Class weights
    weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(enumerate(weights))
    print(f"\n⚖️ Class weights: {class_weight_dict}")

    # Build model
    print("\n🏗️ Building improved model...")
    model, base_model = build_improved_model(num_classes=len(CLASS_NAMES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Simple callbacks without restore_best_weights (causes pickle error in Python 3.12)
    checkpoint = callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=7, min_lr=1e-7, verbose=1
    )

    # Phase 1: Train head
    print("\n" + "=" * 65)
    print("🚀 PHASE 1: Training Classification Head")
    print("=" * 65)

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS_PHASE1,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=[checkpoint, reduce_lr],
        verbose=1,
    )

    # Load best model from phase 1
    print("\n📥 Loading best model from Phase 1...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    # Phase 2: Fine-tune
    print("\n" + "=" * 65)
    print("🔧 PHASE 2: Fine-Tuning Base Model")
    print("=" * 65)

    # Get base model from loaded model and unfreeze top layers
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            layer.trainable = True
            for sublayer in layer.layers[:-30]:
                sublayer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS_PHASE2,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=[checkpoint, reduce_lr],
        verbose=1,
    )

    # Load best model
    print("\n📥 Loading best model...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    # Final evaluation
    print("\n" + "=" * 65)
    print("📈 FINAL EVALUATION ON TEST SET")
    print("=" * 65)

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3))

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\n✅ Test Accuracy: {accuracy:.2%}")
    print(f"✅ Weighted F1: {f1:.2%}")

    # Save confusion matrix
    cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    plot_confusion_matrix(y_test, y_pred, cm_path)

    # Update config
    update_config()

    # Health report
    print("\n" + "=" * 65)
    print("📋 TRAINING SUMMARY")
    print("=" * 65)

    if accuracy >= 0.85:
        print("✅ Model accuracy is GOOD")
    else:
        print("⚠️ Model may need more training data")
        print("\n💡 Tips to improve:")
        print("  • Add more diverse images to each category")
        print("  • Ensure Fire and Accident images are visually distinct")
        print("  • Add images with different lighting conditions")
        print("  • Remove any mislabeled images")

    print(f"\n💾 Model saved: {MODEL_SAVE_PATH}")
    print("=" * 65)


def update_config():
    """Update config file."""
    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)

    config["video_multiclass_model"] = "video_multiclass_model.keras"
    config["class_names"] = CLASS_NAMES
    config["video_model_type"] = "multiclass"
    config["img_size_multiclass"] = IMG_SIZE

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"📝 Config updated: {CONFIG_PATH}")


if __name__ == "__main__":
    train_model()
