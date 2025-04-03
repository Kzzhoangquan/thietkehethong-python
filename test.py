import torch
import cv2
import numpy as np
from transformers import TimesformerForVideoClassification, AutoImageProcessor  # Import đúng

# Tải mô hình TimeSformer đã huấn luyện trên Kinetics-400
model_name = "facebook/timesformer-base-finetuned-k400"
model = TimesformerForVideoClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)  # Dùng AutoImageProcessor thay thế

model.eval()  # Đưa mô hình về chế độ đánh giá

# Hàm trích xuất frames từ video
def extract_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print("❌ Không thể đọc video hoặc video rỗng!")
        return None

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Chuyển về RGB
        frames.append(frame)

    cap.release()
    return frames if len(frames) == num_frames else None

# Hàm hiển thị video
def show_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Video", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Hàm dự đoán hành động
def predict_action(video_path):
    show_video(video_path)  # Hiển thị video trước khi nhận dạng
    frames = extract_frames(video_path, num_frames=8)
    if frames is None:
        print("❌ Video quá ngắn hoặc lỗi khi trích xuất frames.")
        return

    inputs = processor(images=frames, return_tensors="pt")  # Tiền xử lý frames
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Load nhãn của Kinetics-400
    with open("kinetics_400_labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]

    print("📌 Dự đoán hành động:", labels[predicted_class])

# Chạy nhận dạng với video đầu vào
if __name__ == "__main__":
    video_path = "V_997.mp4"  # Đổi thành đường dẫn video của bạn
    predict_action(video_path)
