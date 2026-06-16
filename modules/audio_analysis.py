import numpy as np
import librosa
import tensorflow as tf
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Pre-load models at the module level (same as your working version)
tamper_model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "..", "models", "audio_tamper_model.h5")
)
emotion_model = tf.keras.models.load_model(
    os.path.join(BASE_DIR, "..", "models", "voice_emotion_model.h5")
)
voice_labels = np.load(
    os.path.join(BASE_DIR, "..", "models", "voice_labels.npy"),
    allow_pickle=True
)

def analyze_audio_file(file_path):
    try:
        # Use absolute path to avoid "File Not Found" during Flask upload
        abs_path = os.path.abspath(file_path)
        
        audio, sr = librosa.load(abs_path, sr=22050, mono=True)

        if len(audio) == 0:
            return "Error: Empty audio", "Error"

        # Normalize
        audio = audio / (np.max(np.abs(audio)) + 1e-9)

        # -------- TAMPER --------
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T

        if mfcc.shape[0] < 174:
            mfcc = np.pad(mfcc, ((0, 174 - mfcc.shape[0]), (0, 0)))
        else:
            mfcc = mfcc[:174]

        mfcc = mfcc[np.newaxis, ..., np.newaxis]
        tamper_pred = tamper_model.predict(mfcc, verbose=0)[0]
        
        # Using your exact threshold logic
        tamper_status = "Tampered" if tamper_pred[1] > 0.65 else "Original"

        # -------- EMOTION --------
        mfcc2 = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc2 = np.mean(mfcc2.T, axis=0).reshape(1, -1)

        emo_pred = emotion_model.predict(mfcc2, verbose=0)[0]
        emotion_status = str(voice_labels[np.argmax(emo_pred)])

        return f"Forensic: {tamper_status} | Emotion: {emotion_status}", "Success"

    except Exception as e:
        print("AUDIO ERROR:", e)
        return f"Analysis Error: {str(e)}", "Error"