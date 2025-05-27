import base64
from flask import Flask, request, jsonify
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

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@app.route('/predict/video', methods=['POST'])
def predict_action():
    if 'video' not in request.files:
        return jsonify({"error": "Không có file video"}), 400

    video_file = request.files['video']
    
    video_filename = f"video.mp4"
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    
    video_file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Không thể mở video"}), 400

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  

        cv2.imshow("Video Trước Khi Dự Đoán", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):  
            break

    cap.release()
    cv2.destroyAllWindows()

    frames = extract_frames(video_path, num_frames=8)
    if frames is None:
        return jsonify({"error": "Video quá ngắn hoặc lỗi khi trích xuất frames"}), 400

    inputs = processor(images=frames, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # predicted_class = torch.argmax(outputs.logits, dim=1).item()
    # action = labels[predicted_class] if 0 <= predicted_class < len(labels) else "khong xac dinh"
    # print(action)

    # return jsonify({"action": action})
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    action = labels[predicted_class] if 0 <= predicted_class < len(labels) else "khong xac dinh"
    other_actions = [a for a in labels if a != action]
    extra_action = random.choice(other_actions) if other_actions else "khong xac dinh"
    # Trả về mảng các hành động
    actions = [action, extra_action]
    print(actions)

    return jsonify({"actions": actions})




def preprocess_images(images_base64):
    """Giải mã danh sách ảnh từ base64 và chuyển thành tensor"""
    frames = []
    for img_base64 in images_base64:
        image_data = base64.b64decode(img_base64.split(",")[1])
        image = Image.open(BytesIO(image_data)).convert("RGB")
        frames.append(np.array(image))
   
    if len(frames) != 8:
        return None
    
   
    inputs = processor(images=frames, return_tensors="pt")
    return inputs["pixel_values"]

@app.route("/predict/camera", methods=["POST"])
def predict_camera():
    print("📡 Nhận request từ client...")

    if "file" not in request.files:
        return jsonify({"error": "Không tìm thấy file"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Tên file không hợp lệ"}), 400

    print(f"✅ Đã nhận file: {file.filename}")

    video_path = os.path.join(UPLOAD_FOLDER, "camera_video.mp4")
    file.save(video_path)

    frames = extract_frames(video_path, num_frames=8)
    if frames is None:
        return jsonify({"error": "Video quá ngắn hoặc lỗi khi trích xuất frames"}), 400

    inputs = processor(images=frames, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    action = labels[predicted_class] if 0 <= predicted_class < len(labels) else "Không xác định"
    
    # print(f"🎯 Hành động nhận diện: {action}")
    # return jsonify({"action": action})

    other_actions = [a for a in labels if a != action]
    extra_action = random.choice(other_actions) if other_actions else "khong xac dinh"
    actions = [action, extra_action]
    print(action)
    print(actions)

    return jsonify({"actions": actions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
