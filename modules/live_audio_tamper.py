import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import time

# ---------------- CONFIG ----------------
MODEL_PATH = "model/audio_tamper_model.h5"
SAMPLE_RATE = 22050
DURATION = 3          # seconds
N_MFCC = 40
MAX_LEN = 174         # must match training
CONF_THRESHOLD = 0.65
# ----------------------------------------

print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded successfully")

def extract_mfcc(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.T

    if mfcc.shape[0] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)))
    else:
        mfcc = mfcc[:MAX_LEN]

    return mfcc[np.newaxis, ..., np.newaxis]

print("\n🎤 LIVE AUDIO TAMPER DETECTION STARTED")
print("Press Ctrl+C to stop\n")

while True:
    print("Listening...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32")
    sd.wait()
    audio = audio.flatten()

    # ---- SILENCE CHECK ----
    if np.max(np.abs(audio)) < 0.01:
        print("⚠ Silence detected — skipping\n")
        continue

    # ---- NORMALIZATION ----
    audio = audio / np.max(np.abs(audio))

    # ---- FEATURE EXTRACTION ----
    mfcc = extract_mfcc(audio, SAMPLE_RATE)

    # ---- PREDICTION ----
    pred = model.predict(mfcc, verbose=0)[0]
    original_score = pred[0]
    tampered_score = pred[1]

    # ---- DECISION ----
    if tampered_score > CONF_THRESHOLD:
        result = "🔴 TAMPERED"
        conf = tampered_score
    elif original_score > CONF_THRESHOLD:
        result = "🟢 ORIGINAL"
        conf = original_score
    else:
        result = "🟡 UNCERTAIN"
        conf = max(original_score, tampered_score)

    print(f"RESULT → {result} | confidence = {conf:.2f}\n")
    time.sleep(0.5)