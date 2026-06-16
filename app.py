from flask import Flask, render_template, request, jsonify, send_file
import os
import threading
import torch
from datetime import datetime

# --- Internal Module Imports ---
from modules.image_tampering import predict_image_tampering
from modules.audio_analysis import analyze_audio_file
from modules.chatbot import get_chatbot_response
from modules.face_emotion_live import start_webcam_detection, get_current_face_emotion
from modules.inference import predict_vedio
from modules.model import ImprovedEfficientViT
from modules.report_gen import generate_forensic_report

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Persistent session results for PDF Generation
session_data = {
    "image": "Not Analyzed",
    "video": "Not Analyzed",
    "audio": "Not Analyzed"
}

# ==========================================
# 1. MODEL INITIALIZATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
video_model = ImprovedEfficientViT()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_MODEL_PATH = os.path.join(BASE_DIR, "models", "video_model.pth")

if os.path.exists(VIDEO_MODEL_PATH):
    try:
        video_model.load_state_dict(torch.load(VIDEO_MODEL_PATH, map_location=device))
        video_model.to(device).eval()
        print("✅ Forensic Video Engine Online")
    except Exception as e:
        print(f"❌ Initialization Error: {e}")

# ==========================================
# 2. ANALYSIS ROUTES
# ==========================================

@app.route('/')
def home():
    dt_string = datetime.now().strftime("%A, %B %d, %Y")
    return render_template('index.html', current_date=dt_string)

@app.route('/analyze_image', methods=['POST'])
def img_route():
    file = request.files['file']
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)
    result, conf = predict_image_tampering(path)
    session_data["image"] = f"{result} ({conf}%)"
    return jsonify({"result": session_data["image"]})

@app.route('/analyze_video', methods=['POST'])
def video_route():
    file = request.files['file']
    path = os.path.join(app.config['UPLOAD_FOLDER'], "v_temp.mp4")
    file.save(path)
    data = predict_vedio(path, video_model)
    session_data["video"] = f"{data['result']} ({data['confidence']}%)"
    return jsonify({"result": session_data["video"]})

@app.route('/analyze_audio', methods=['POST'])
def audio_route():
    file = request.files['file']
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)
    # Using your specific threshold logic inside this module
    result_str, _ = analyze_audio_file(path)
    session_data["audio"] = result_str
    return jsonify({"result": result_str})

# ==========================================
# 3. UTILITY ROUTES (PDF, Chat, Webcam)
# ==========================================

@app.route('/generate_report')
def report_route():
    pdf_path = generate_forensic_report(
        session_data["image"], 
        session_data["video"], 
        session_data["audio"]
    )
    return send_file(pdf_path, as_attachment=True)

@app.route('/start_live_face')
def live_face():
    threading.Thread(target=start_webcam_detection, daemon=True).start()
    return jsonify({"status": "Active"})

@app.route('/chat', methods=['POST'])
def chat_route():
    user_msg = request.json.get("message")
    emotion = get_current_face_emotion()
    response = get_chatbot_response(f"Context: User is {emotion}. Query: {user_msg}")
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5050, use_reloader=False)