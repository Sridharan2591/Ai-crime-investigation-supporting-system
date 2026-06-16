import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "casia_tampering_model.h5")

def predict_image_tampering(image_path):
    if not os.path.exists(MODEL_PATH):
        return "Model Missing", 0
        
    model = load_model(MODEL_PATH)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128)).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img, verbose=0)[0][0]
    result = "Tampered" if prediction > 0.5 else "Authentic"
    confidence = round(float(prediction if prediction > 0.5 else 1-prediction)*100, 2)
    return result, confidence