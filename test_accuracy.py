import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models

# Load the updated model
VIDEO_MODEL_PATH = 'video_transfer_model.h5'
IMG_SIZE = 96
TEST_IMAGES = [
    r'd:\multimodel-surveillence\DATASET\dataset.videos\anomoly\fire explotion\fire1.png',
    r'd:\multimodel-surveillence\DATASET\dataset.videos\anomoly\accidents\accident18.png'
]

print(f"Loading model from {VIDEO_MODEL_PATH}...")
model = models.load_model(VIDEO_MODEL_PATH)

def test_image(path):
    print(f"\nProcessing: {path}")
    img = cv2.imread(path)
    if img is None:
        print(f"Error: Could not load image at {path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    
    # Preprocess
    img_input = np.expand_dims(img_resized.astype('float32'), axis=0)
    img_processed = tf.keras.applications.mobilenet_v2.preprocess_input(img_input)
    
    # Predict
    pred = model.predict(img_processed, verbose=0)
    score = float(pred[0][0])
    
    print(f"Anomaly Score: {score:.4f}")
    if score > 0.5:
        print("RESULT: ANOMALY (Correct!)")
    else:
        print("RESULT: NORMAL (Fail/Needs More Training)")

if __name__ == "__main__":
    for img in TEST_IMAGES:
        test_image(img)
