import base64
from fastapi import Form
from flask import Flask, json, request, jsonify
import os
import cv2
import torch
import numpy as np
from transformers import TimesformerForVideoClassification, AutoImageProcessor
from flask_cors import CORS  
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)  

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TimesformerForVideoClassification.from_pretrained("custom_timesformer").to(device)
processor = AutoImageProcessor.from_pretrained("custom_timesformer")


labels = ["đấm", "đá", "tát"]

class Mau:
    def __init__(self, videopath):
        self.videopath = videopath

class MoHinh:
    def __init__(self, id, ten, phienban, dochinhxac):
        self.id = id
        self.ten = ten
        self.phienban = phienban
        self.dochinhxac = dochinhxac

class HanhVi:
    def __init__(self, ten):
        self.ten = ten
    
    def to_dict(self):
        return {
            "ten": self.ten,
        }

def extract_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        print("⚠ Video quá ngắn!")
        return None

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames if len(frames) == num_frames else None

@app.route("/predict/camera", methods=["POST"])
def predict_camera():
    try:
        moHinh = request.form.get("moHinh")
        mo_hinh_dict = json.loads(moHinh)
        mo_hinh_obj = MoHinh(**mo_hinh_dict)
        if mo_hinh_obj.id != 1:
            print("⚠️ Mô hình đang được phát triển")
            hanhViList = [HanhVi(ten="khong xac dinh")]
            return jsonify({"actions": [hanhVi.to_dict() for hanhVi in hanhViList]})
        mau = request.form.get("mau")
        mau_dict = json.loads(mau)
        mau_obj = Mau(**mau_dict)
        print(mau_obj.videopath)
        frames = extract_frames(mau_obj.videopath, num_frames=8)
        if frames is None:
            action = "khong xac dinh"
        else:
            inputs = processor(images=frames, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()
            action = labels[predicted_class] if 0 <= predicted_class < len(labels) else "khong xac dinh"
        print("➡️ Dự đoán hành vi:", action)
        hanhViList = [HanhVi(ten=action)]
        return jsonify({"actions": [hanhVi.to_dict() for hanhVi in hanhViList]})
    
    except Exception as e:
        print("❌ Lỗi xử lý:", str(e))
        hanhViList = [HanhVi(ten="khong xac dinh")]
        return jsonify({"actions": [hanhVi.to_dict() for hanhVi in hanhViList]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
