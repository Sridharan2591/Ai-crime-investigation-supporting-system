import os
import cv2
import torch
import numpy as np
from mtcnn import MTCNN
from torchvision import transforms
from .model import ImprovedEfficientViT 

def predict_vedio(video_path, model_video):
    try:
        detector = MTCNN()
        cap = cv2.VideoCapture(os.path.abspath(video_path))
        
        frames = []
        count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # We take 5 high-quality samples instead of 10 to reduce noise
        while len(frames) < 5 and count < total_frames:
            ret, frame = cap.read()
            if not ret: break
            count += 1
            
            # Check a frame every 15% of the video length for diversity
            if count % max(1, total_frames // 7) == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = detector.detect_faces(rgb)
                if res:
                    x, y, w, h = res[0]['box']
                    # Buffer the crop so the face isn't cut too tightly
                    face = rgb[max(0, y-20):y+h+20, max(0, x-20):x+w+20]
                    if face.size != 0:
                        frames.append(cv2.resize(face, (224, 224)))
        cap.release()

        if not frames:
            return {"result": "No face detected", "confidence": 0}
        
        # IMPROVED TRANSFORM: Added specific ImageNet normalization
        transform = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_video.to(device)
        
        results = []
        for f in frames:
            with torch.no_grad():
                inp = transform(f).unsqueeze(0).to(device)
                output = model_video(inp)
                # Apply sigmoid and get probability
                prob = torch.sigmoid(output).item()
                results.append(prob)
        
        avg_prob = sum(results) / len(results)
        
        # ADJUSTED THRESHOLD:
        # If the average probability of being fake is > 0.7, call it Fake.
        # Otherwise, call it Real. This reduces False Positives.
        is_fake = avg_prob > 0.7 
        
        return {
            "result": "Fake/Deepfake" if is_fake else "Real/Original", 
            "confidence": round(avg_prob * 100, 2) if is_fake else round((1 - avg_prob) * 100, 2)
        }
    
    except Exception as e:
        print(f"VIDEO ERROR: {e}")
        return {"result": "Processing Error", "confidence": 0}