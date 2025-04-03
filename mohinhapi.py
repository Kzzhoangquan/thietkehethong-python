import base64
from flask import Flask, request, jsonify
import os
import cv2
import torch
import numpy as np
from transformers import TimesformerForVideoClassification, AutoImageProcessor
from flask_cors import CORS  # Hỗ trợ CORS
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)  # Bật CORS cho tất cả các route

# Load mô hình đã train
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TimesformerForVideoClassification.from_pretrained("custom_timesformer").to(device)
processor = AutoImageProcessor.from_pretrained("custom_timesformer")

# Nhãn hành động
labels = ["đấm", "đá", "tát"]

# Hàm trích xuất frames từ video
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

# API nhận video và dự đoán
# @app.route('/predict', methods=['POST'])
# def predict_action():
#     if 'video' not in request.files:
#         return jsonify({"error": "Không có file video"}), 400

#     video_file = request.files['video']
#     video_path = "temp_video.mp4"
#     video_file.save(video_path)

#     frames = extract_frames(video_path, num_frames=8)
#     if frames is None:
#         return jsonify({"error": "Video quá ngắn hoặc lỗi khi trích xuất frames"}), 400

#     inputs = processor(images=frames, return_tensors="pt").to(device)
    
#     with torch.no_grad():
#         outputs = model(**inputs)

#     predicted_class = torch.argmax(outputs.logits, dim=1).item()
#     action = labels[predicted_class] if 0 <= predicted_class < len(labels) else "Không xác định"

#     return jsonify({"action": action})

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
@app.route('/predict', methods=['POST'])
def predict_action():
    if 'video' not in request.files:
        return jsonify({"error": "Không có file video"}), 400

    video_file = request.files['video']
    
    # Tạo tên file duy nhất
    video_filename = f"video.mp4"
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    
    video_file.save(video_path)

    # Hiển thị video trước khi nhận dạng
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": "Không thể mở video"}), 400

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Kết thúc video

        cv2.imshow("Video Trước Khi Dự Đoán", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):  # Nhấn 'q' để thoát
            break

    cap.release()
    cv2.destroyAllWindows()

    # Tiến hành nhận dạng sau khi hiển thị video
    frames = extract_frames(video_path, num_frames=8)
    if frames is None:
        return jsonify({"error": "Video quá ngắn hoặc lỗi khi trích xuất frames"}), 400

    inputs = processor(images=frames, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    action = labels[predicted_class] if 0 <= predicted_class < len(labels) else "khong xac dinh"
    print(action)

    return jsonify({"action": action})

def preprocess_images(images_base64):
    """Giải mã danh sách ảnh từ base64 và chuyển thành tensor"""
    frames = []
    for img_base64 in images_base64:
        image_data = base64.b64decode(img_base64.split(",")[1])
        image = Image.open(BytesIO(image_data)).convert("RGB")
        frames.append(np.array(image))
    
    # Đảm bảo có đúng 8 frames
    if len(frames) != 8:
        return None
    
    # Tiền xử lý frames để đưa vào model
    inputs = processor(images=frames, return_tensors="pt")
    return inputs["pixel_values"]

@app.route("/predict_camera", methods=["POST"])
def predict_camera():
    print("📡 Nhận request từ client...")

    # Kiểm tra có file hay không
    if "file" not in request.files:
        return jsonify({"error": "Không tìm thấy file"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Tên file không hợp lệ"}), 400

    print(f"✅ Đã nhận file: {file.filename}")

    # Lưu file video tạm thời
    video_path = os.path.join(UPLOAD_FOLDER, "camera_video.mp4")
    file.save(video_path)

    # Trích xuất frames từ video
    frames = extract_frames(video_path, num_frames=8)
    if frames is None:
        return jsonify({"error": "Video quá ngắn hoặc lỗi khi trích xuất frames"}), 400

    # Tiền xử lý frames để đưa vào model
    inputs = processor(images=frames, return_tensors="pt").to(device)

    # Dự đoán hành động
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    action = labels[predicted_class] if 0 <= predicted_class < len(labels) else "Không xác định"
    
    print(f"🎯 Hành động nhận diện: {action}")
    return jsonify({"action": action})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

# import base64
# from collections import deque
# from flask import Flask, request, jsonify
# import os
# import cv2
# import torch
# import numpy as np
# from transformers import TimesformerForVideoClassification, AutoImageProcessor
# from flask_cors import CORS  # Hỗ trợ CORS
# from io import BytesIO
# from PIL import Image
# import torchvision.transforms as T


# app = Flask(__name__)
# CORS(app)  # Bật CORS cho tất cả các route

# # Load mô hình đã train
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = TimesformerForVideoClassification.from_pretrained("custom_timesformer").to(device)
# processor = AutoImageProcessor.from_pretrained("custom_timesformer")

# # Nhãn hành động
# labels = ["đấm", "đá", "tát"]

# frame_buffer = deque(maxlen=15)

# def save_video(frames, filename="output.avi"):
#     """Lưu 15 frame thành video nếu phát hiện hành động."""
#     height, width, _ = frames[0].shape  # Thêm _ để lấy số kênh màu
#     out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 10, (width, height), isColor=True)

#     for frame in frames:
#         out.write(frame)

#     out.release()

# @app.route("/upload_frame", methods=["POST"])
# def upload_frame():
#     global frame_buffer
    
#     # Nhận danh sách frame từ Java
#     files = request.files.getlist("frames")
#     if not files:
#         return jsonify({"error": "No frames received"}), 400
    
#     # Xử lý từng frame
#     for file in files:
#         nparr = np.frombuffer(file.read(), np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Đọc ảnh có màu
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển BGR → RGB
#         img = cv2.resize(img, (224, 224))  # Resize ảnh về 224x224
#         frame_buffer.append(img)
    
#     # Nếu có đủ 15 frame, thực hiện dự đoán
#     if len(frame_buffer) == 15:
#         try:
#             # Đầu tiên, chuyển đổi frames thành định dạng đặc biệt cho TimesformerForVideoClassification
#             # Timesformer mong đợi đầu vào là tensor có hình dạng (batch_size, num_frames, num_channels, height, width)
#             frames_array = np.array(frame_buffer)  # Shape: (15, 224, 224, 3)
#             # Chuyển đổi thành định dạng (1, 15, 3, 224, 224)
#             video_tensor = torch.from_numpy(frames_array).float() / 255.0
#             video_tensor = video_tensor.permute(0, 3, 1, 2)  # (15, 3, 224, 224)
#             video_tensor = video_tensor.unsqueeze(0)  # (1, 15, 3, 224, 224)
#             video_tensor = video_tensor.to(device)
            
#             # Gọi mô hình với tham số đúng
#             with torch.no_grad():
#                 outputs = model(pixel_values=video_tensor)
#                 predicted_class = torch.argmax(outputs.logits, dim=1).item()
#                 action = labels[predicted_class]
            
#             # Lưu video nếu có hành động
#             save_video(list(frame_buffer), f"action_{action}.avi")
            
#             # Reset buffer để nhận tiếp 15 frame mới
#             frame_buffer.clear()
            
#             return jsonify({"action": action})
#         except Exception as e:
#             import traceback
#             traceback_str = traceback.format_exc()
#             print(f"Error: {str(e)}\n{traceback_str}")
#             return jsonify({"error": str(e), "traceback": traceback_str}), 500
    
#     # Trả về thông báo nếu chưa đủ 15 frame để dự đoán
#     return jsonify({"action": "waiting"})  # Chưa đủ 15 frame để dự đoán



# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)

