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

def extract_gap_sequences(video_path, num_frames=8, gap=4, start_frame=1):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    sequences = []
    frame_start = start_frame

    while frame_start + (num_frames - 1) * gap < total_frames:
        frames = []
        for i in range(num_frames):
            frame_idx = frame_start + i * gap
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        if len(frames) == num_frames:
            sequences.append(frames)
        frame_start += num_frames * gap
    cap.release()
    return sequences

@app.route("/predict/camera", methods=["POST"])
def predictcamera():
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
        sequences = extract_gap_sequences(mau_obj.videopath, num_frames=8, gap=4, start_frame=1)
        if not sequences:
            return jsonify({"error": "Video quá ngắn hoặc không đủ frame cho bất kỳ đoạn nào"}), 400

        actions = []

        for frames in sequences:
            inputs = processor(images=frames, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            predicted_class = torch.argmax(outputs.logits, dim=1).item()
            action = labels[predicted_class] if 0 <= predicted_class < len(labels) else "khong xac dinh"
            actions.append(action)
        print(actions)
        actions = list(set(actions))
        if len(actions) > 1 and "khong xac dinh" in actions:
            actions.remove("khong xac dinh")
        print(actions)    
        hanhViList = [HanhVi(ten=action) for action in actions]
        return jsonify({"actions": [hanhVi.to_dict() for hanhVi in hanhViList]})
    
    except Exception as e:
        print("❌ Lỗi xử lý:", str(e))
        hanhViList = [HanhVi(ten="khong xac dinh")]
        return jsonify({"actions": [hanhVi.to_dict() for hanhVi in hanhViList]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
