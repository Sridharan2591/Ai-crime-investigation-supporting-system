import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "emotion_model.h5")
current_emotion = "Neutral"

def get_current_face_emotion():
    return current_emotion

def start_webcam_detection():
    global current_emotion
    try:
        model = load_model(MODEL_PATH)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # CAP_DSHOW is more stable on Windows
        if not cap.isOpened():
            print("Webcam Busy or Not Found")
            return

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                roi = cv2.resize(frame[y:y+h, x:x+w], (96, 96)).astype("float32") / 255.0
                roi = np.expand_dims(roi, axis=0)
                pred = model.predict(roi, verbose=0)[0]
                current_emotion = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"][np.argmax(pred)]
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, current_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            cv2.imshow('Crime AI Vision', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Webcam Error: {e}")