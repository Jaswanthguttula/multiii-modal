import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def load_video_frames(data_root, target_shape=(64, 64)):
    frames = []
    labels = [] # 0 normal, 1 anomaly
    
    # Define categories
    categories = {
        "normal": 0,
        "anomoly": 1
    }
    
    for category_name, category_idx in categories.items():
        category_dir = os.path.join(data_root, category_name)
        if not os.path.exists(category_dir):
            continue
            
        # Recursive search for images in all subfolders
        print(f"Loading frames from {category_name}...")
        for root, dirs, files in os.walk(category_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = cv2.resize(img, target_shape)
                        frames.append(img.flatten() / 255.0)
                        labels.append(category_idx)
                        
    return np.array(frames), np.array(labels)

def train_video_autoencoder(data_root):
    X, y = load_video_frames(data_root)
    
    if X.size == 0:
        print("No video frames found.")
        return
        
    # Isolation Forest prefers normal data only, or use labels 
    # but the user asked for an Autoencoder.
    # Autoencoder is usually trained on normal data to reconstruct it.
    
    normal_frames = X[y == 0]
    if len(normal_frames) == 0:
        print("No normal frames to train Autoencoder. Training on all available data.")
        normal_frames = X
        
    input_dim = X.shape[1]
    
    # Simple Autoencoder architecture
    encoder = models.Sequential([
        layers.InputLayer(input_shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'), # bottleneck
    ])
    
    decoder = models.Sequential([
        layers.InputLayer(input_shape=(32,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(input_dim, activation='sigmoid'),
    ])
    
    autoencoder = models.Model(inputs=encoder.input, outputs=decoder(encoder.output))
    autoencoder.compile(optimizer='adam', loss='mse')
    
    print("Training Autoencoder on normal frames...")
    autoencoder.fit(normal_frames, normal_frames, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
    
    # Evaluate reconstruction error for both normal and anomalies
    reconstructions = autoencoder.predict(X)
    reconstruction_error = np.mean(np.square(X - reconstructions), axis=1)
    
    # Use reconstruction error as an anomaly score
    # High error = Anomaly
    
    # Basic evaluation: Check if anomaly frames have higher error than normal frames
    normal_errors = reconstruction_error[y == 0]
    anomaly_errors = reconstruction_error[y == 1]
    
    if len(anomaly_errors) > 0:
        print(f"Mean error (Normal): {np.mean(normal_errors):.6f}")
        print(f"Mean error (Anomaly): {np.mean(anomaly_errors):.6f}")
        
        # Test if reconstruction error can separate the classes
        threshold = np.percentile(normal_errors, 95) # Top 5% of normals are flagged
        preds = (reconstruction_error > threshold).astype(int)
        accuracy = np.mean(preds == y)
        print(f"Reconstruction Error Accuracy (threshold @ 95th p-tile): {accuracy * 100:.2f}%")
        
    autoencoder.save("video_autoencoder.h5")
    print("Video Autoencoder model saved as video_autoencoder.h5")

if __name__ == "__main__":
    video_path = "d:\\multimodel-surveillence\\DATASET\\dataset.videos"
    train_video_autoencoder(video_path)
