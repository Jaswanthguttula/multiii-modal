"""
ADVANCED Multi-Class Video Anomaly Classification Training
Uses EfficientNetV2 + Focal Loss + Label Smoothing + Mixup
For significantly improved accuracy on imbalanced surveillance data.
"""

import os
import cv2
import json
import numpy as np
import warnings
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers, backend as K
from tensorflow.keras.applications import EfficientNetV2B2
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.utils import class_weight
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# --- CONFIGURATION ---
BASE_DIR = r"d:\multimodel-surveillence"
VIDEO_DATASET = os.path.join(BASE_DIR, "DATASET", "dataset.videos")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "video_multiclass_model.keras")
CONFIG_PATH = os.path.join(BASE_DIR, "fusion_config.json")

IMG_SIZE = 160  # EfficientNetV2B2 optimal size
BATCH_SIZE = 8
EPOCHS = 100
TARGET_PER_CLASS = 200  # More samples

CLASS_MAPPING = {
    "normal": 0,
    "accidents": 1,
    "fire explotion": 2,
    "robbery": 3,
    "violence": 4,
}

CLASS_NAMES = ["Normal", "Accident", "Fire", "Robbery", "Violence"]


# ============================================================
# FOCAL LOSS - Better for imbalanced classes
# ============================================================
class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss for handling class imbalance."""

    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(tf.squeeze(y_true), depth=len(CLASS_NAMES))

        # Label smoothing
        if self.label_smoothing > 0:
            y_true_one_hot = y_true_one_hot * (
                1 - self.label_smoothing
            ) + self.label_smoothing / len(CLASS_NAMES)

        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Focal loss calculation
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
# ADVANCED DATA AUGMENTATION
# ============================================================
def heavy_augment(img, is_normal=False):
    """Apply heavy augmentation, especially for normal class."""
    augmented = img.copy().astype(np.float32)

    # Random horizontal flip
    if np.random.random() > 0.5:
        augmented = np.fliplr(augmented)

    # Random vertical flip (rare but useful)
    if np.random.random() > 0.9:
        augmented = np.flipud(augmented)

    # Random rotation (-30 to +30 degrees)
    angle = np.random.uniform(-30, 30)
    h, w = augmented.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    augmented = cv2.warpAffine(
        augmented.astype(np.uint8), M, (w, h), borderMode=cv2.BORDER_REFLECT
    )

    # Random brightness (more aggressive)
    brightness = np.random.uniform(0.5, 1.5)
    augmented = np.clip(augmented * brightness, 0, 255)

    # Random contrast
    factor = np.random.uniform(0.7, 1.3)
    mean = np.mean(augmented, axis=(0, 1), keepdims=True)
    augmented = np.clip((augmented - mean) * factor + mean, 0, 255)

    # Random saturation (color shift)
    if np.random.random() > 0.5:
        hsv = cv2.cvtColor(augmented.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(
            np.float32
        )
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.7, 1.3), 0, 255)
        augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(
            np.float32
        )

    # Random zoom/crop
    if np.random.random() > 0.3:
        zoom = np.random.uniform(0.75, 1.0)
        crop_size = int(IMG_SIZE * zoom)
        max_start = IMG_SIZE - crop_size
        start_x = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        start_y = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        cropped = augmented[
            start_y : start_y + crop_size, start_x : start_x + crop_size
        ]
        augmented = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))

    # Random noise
    if np.random.random() > 0.5:
        noise = np.random.normal(0, np.random.uniform(5, 15), augmented.shape)
        augmented = np.clip(augmented + noise, 0, 255)

    # Extra augmentations for normal class (most confused)
    if is_normal:
        # Random blur
        if np.random.random() > 0.5:
            ksize = np.random.choice([3, 5])
            augmented = cv2.GaussianBlur(
                augmented.astype(np.uint8), (ksize, ksize), 0
            ).astype(np.float32)

        # Random sharpening
        if np.random.random() > 0.7:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            augmented = cv2.filter2D(augmented.astype(np.uint8), -1, kernel).astype(
                np.float32
            )

    return augmented.astype("float32")


def mixup(x1, y1, x2, y2, alpha=0.2):
    """Mixup augmentation - blends two images."""
    lam = np.random.beta(alpha, alpha)
    x = lam * x1 + (1 - lam) * x2
    y = lam * y1 + (1 - lam) * y2
    return x, y


def load_and_augment_data_advanced(data_root, target_per_class=200):
    """Load dataset with advanced augmentation and class balancing."""
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

    # Augment with extra focus on Normal class
    print(f"\n🔄 Heavy augmentation to {target_per_class} samples per class...")

    frames, labels = [], []
    for class_idx in range(len(CLASS_NAMES)):
        original = class_frames[class_idx]
        if len(original) == 0:
            print(f"  ⚠️ No images for {CLASS_NAMES[class_idx]}")
            continue

        is_normal = class_idx == 0
        # Normal class gets 1.5x more samples due to higher confusion
        class_target = int(target_per_class * 1.5) if is_normal else target_per_class

        # Add all originals
        for img in original:
            frames.append(img)
            labels.append(class_idx)

        # Augment to reach target
        current = len(original)
        while current < class_target:
            idx = np.random.randint(0, len(original))
            aug_img = heavy_augment(original[idx], is_normal=is_normal)
            frames.append(aug_img)
            labels.append(class_idx)
            current += 1

        print(f"  {CLASS_NAMES[class_idx]}: {len(original)} -> {current}")

    print(f"\n✅ Total samples: {len(frames)}")
    return np.array(frames), np.array(labels)


# ============================================================
# MODEL ARCHITECTURE - EfficientNetV2
# ============================================================
def build_efficientnet_model(num_classes=5):
    """Build EfficientNetV2B2 model with advanced head."""

    # Data augmentation layer
    data_augmentation = models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.3),
            layers.RandomZoom(0.3),
            layers.RandomContrast(0.3),
            layers.RandomBrightness(0.3),
            layers.RandomTranslation(0.2, 0.2),
        ],
        name="augmentation",
    )

    # EfficientNetV2B2 - better accuracy than MobileNetV2
    base_model = EfficientNetV2B2(
        weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    # Build model
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)

    # Advanced classification head with SE-like attention
    x = layers.Dense(1024, kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)  # Swish activation (better than ReLU)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("swish")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs), base_model


# ============================================================
# TRAINING
# ============================================================
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
    plt.title("Confusion Matrix - Advanced Model", fontsize=14)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"\n📊 Confusion matrix saved: {save_path}")

    print("\n📈 Per-Class Accuracy:")
    print("-" * 60)
    for i, name in enumerate(CLASS_NAMES):
        total = cm[i].sum()
        correct = cm[i][i]
        acc = correct / total if total > 0 else 0
        confused_with = []
        for j, conf_name in enumerate(CLASS_NAMES):
            if i != j and cm[i][j] > 0:
                confused_with.append(f"{conf_name}:{cm[i][j]}")
        conf_str = ", ".join(confused_with) if confused_with else "none"
        print(f"  {name}: {acc:.1%} ({correct}/{total}) | Confused: {conf_str}")


def train_advanced():
    """Main training function with advanced techniques."""
    print("=" * 70)
    print("  ADVANCED MULTI-CLASS TRAINING - EfficientNetV2 + Focal Loss")
    print("=" * 70)

    # Load data with heavy augmentation
    X, y = load_and_augment_data_advanced(
        VIDEO_DATASET, target_per_class=TARGET_PER_CLASS
    )

    if len(X) == 0:
        print("❌ No data found!")
        return

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.18, random_state=42, stratify=y_temp
    )

    print(f"\n📊 Data Split:")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Preprocess for EfficientNet (different preprocessing than MobileNet)
    X_train = tf.keras.applications.efficientnet_v2.preprocess_input(X_train.copy())
    X_val = tf.keras.applications.efficientnet_v2.preprocess_input(X_val.copy())
    X_test = tf.keras.applications.efficientnet_v2.preprocess_input(X_test.copy())

    # Compute class weights with extra boost for Normal
    weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    weights[0] *= 2.0  # Double weight for Normal class (most confused)
    class_weight_dict = dict(enumerate(weights))
    print(f"\n⚖️ Class weights (Normal boosted): {class_weight_dict}")

    # Build model
    print("\n🏗️ Building EfficientNetV2B2 model...")
    model, base_model = build_efficientnet_model(num_classes=len(CLASS_NAMES))

    # Use Focal Loss with label smoothing
    focal_loss = FocalLoss(gamma=2.0, alpha=0.25, label_smoothing=0.1)

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001),
        loss=focal_loss,
        metrics=["accuracy"],
    )

    # Callbacks
    checkpoint = callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
    )

    # Phase 1: Train head
    print("\n" + "=" * 70)
    print("🚀 PHASE 1: Training Classification Head (Frozen Base)")
    print("=" * 70)

    history1 = model.fit(
        X_train,
        y_train,
        epochs=40,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=[checkpoint, reduce_lr],
        verbose=1,
    )

    # Load best model from phase 1
    print("\n📥 Loading best model from Phase 1...")
    model = tf.keras.models.load_model(
        MODEL_SAVE_PATH, custom_objects={"FocalLoss": FocalLoss}
    )

    # Phase 2: Fine-tune entire model
    print("\n" + "=" * 70)
    print("🔧 PHASE 2: Fine-Tuning Entire Model")
    print("=" * 70)

    # Unfreeze all layers
    for layer in model.layers:
        layer.trainable = True

    # Recompile with very low learning rate
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=0.0001),
        loss=focal_loss,
        metrics=["accuracy"],
    )

    history2 = model.fit(
        X_train,
        y_train,
        epochs=60,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=[checkpoint, reduce_lr],
        verbose=1,
    )

    # Load best model
    print("\n📥 Loading best model for evaluation...")
    model = tf.keras.models.load_model(
        MODEL_SAVE_PATH, custom_objects={"FocalLoss": FocalLoss}
    )

    # Final evaluation
    print("\n" + "=" * 70)
    print("📈 FINAL EVALUATION ON TEST SET")
    print("=" * 70)

    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3))

    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print(f"\n✅ Test Accuracy: {accuracy:.2%}")
    print(f"✅ Weighted F1: {f1_weighted:.2%}")
    print(f"✅ Macro F1: {f1_macro:.2%}")

    # Save confusion matrix
    cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    plot_confusion_matrix(y_test, y_pred, cm_path)

    # Update config
    update_config()

    # Summary
    print("\n" + "=" * 70)
    print("📋 TRAINING COMPLETE")
    print("=" * 70)

    if accuracy >= 0.80:
        print("✅ Model accuracy is GOOD!")
    elif accuracy >= 0.70:
        print("🟡 Model accuracy is FAIR - consider adding more data")
    else:
        print("⚠️ Model needs improvement - review dataset quality")

    print(f"\n💾 Model saved: {MODEL_SAVE_PATH}")
    print("=" * 70)


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
    train_advanced()
