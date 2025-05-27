import base64
from flask import Flask, json, request, jsonify
import os
import cv2
import torch
import numpy as np
from transformers import TimesformerForVideoClassification, AutoImageProcessor
from flask_cors import CORS  
from io import BytesIO
from PIL import Image
import random

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


@app.route('/predict/video', methods=['POST'])
def predictvideo():
    if 'mau' not in request.json or 'moHinh' not in request.json:
        return jsonify({"error": "Thiếu dữ liệu cần thiết"}), 400
    mau_raw = request.json.get('mau')
    moHinh_raw = request.json.get('moHinh')
    mau_data = json.loads(mau_raw[0])
    moHinh_data = json.loads(moHinh_raw[0])
    mau = Mau(**mau_data)
    moHinh = MoHinh(**moHinh_data)
    video_path = mau.videopath
    id_model = moHinh.id
    if id_model != 1:
        return jsonify({"error": "Mô hình đang trong thời gian phát triển"}), 400
    video_path = mau.videopath
    # cap = cv2.VideoCapture(video_path)
    # if not cap.isOpened():
    #     return jsonify({"error": "Không thể mở video"}), 400

    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret:
    #         break  

    #     cv2.imshow("Video Trước Khi Dự Đoán", frame)
    #     if cv2.waitKey(25) & 0xFF == ord('q'):  
    #         break

    # cap.release()
    # cv2.destroyAllWindows()

    # frames = extract_frames(video_path, num_frames=8)
    # if frames is None:
    #     return jsonify({"error": "Video quá ngắn hoặc lỗi khi trích xuất frames"}), 400

    # inputs = processor(images=frames, return_tensors="pt").to(device)

    # with torch.no_grad():
    #     outputs = model(**inputs)
    # predicted_class = torch.argmax(outputs.logits, dim=1).item()
    # action = labels[predicted_class] if 0 <= predicted_class < len(labels) else "khong xac dinh"
    # actions = [action]
    sequences = extract_gap_sequences(video_path, num_frames=8, gap=4, start_frame=1)
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
