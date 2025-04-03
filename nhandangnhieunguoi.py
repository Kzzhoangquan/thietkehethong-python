import torch
import cv2
import numpy as np
from ultralytics import YOLO
from transformers import TimesformerForVideoClassification, AutoImageProcessor

# Load mô hình phát hiện người (YOLOv8)
yolo_model = YOLO("yolov8n.pt")  # Có thể thay bằng yolov8s.pt để chính xác hơn

# Load mô hình nhận dạng hành động (TimeSformer)
action_model_name = "facebook/timesformer-base-finetuned-k400"
action_model = TimesformerForVideoClassification.from_pretrained(action_model_name)
processor = AutoImageProcessor.from_pretrained(action_model_name)
action_model.eval()

# Load nhãn hành động của Kinetics-400
with open("kinetics_400_labels.txt", "r") as f:
    action_labels = [line.strip() for line in f.readlines()]

# Hàm phát hiện người trong khung hình
def detect_people(frame):
    results = yolo_model(frame)
    persons = []
    
    for result in results:
        for box in result.boxes:
            cls = int(box.cls.item())  # Lấy class
            if cls == 0:  # Class 0 là "person" trong YOLO
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                persons.append((x1, y1, x2, y2))
    
    return persons

# Hàm hiển thị video với khung người
def show_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        persons = detect_people(frame)

        for x1, y1, x2, y2 in persons:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ khung xanh

        cv2.imshow("Video", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):  # Nhấn 'q' để thoát
            break

    cap.release()
    cv2.destroyAllWindows()

# Hàm trích xuất frames của từng người
def extract_frames(video_path, persons, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    frames_dict = {i: [] for i in range(len(persons))}

    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        for i, (x1, y1, x2, y2) in enumerate(persons):
            person_crop = frame[y1:y2, x1:x2]
            person_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            frames_dict[i].append(person_crop)

    cap.release()
    return frames_dict

# Nhận dạng hành động sau khi video chạy xong
def recognize_actions(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Không thể đọc video!")
        return

    # Phát hiện người trong khung đầu tiên
    persons = detect_people(first_frame)
    if not persons:
        print("❌ Không phát hiện người nào!")
        return

    print(f"👥 Phát hiện {len(persons)} người trong video.")
    
    # Hiển thị video với khung trước khi nhận dạng
    show_video(video_path)

    # Trích xuất frames sau khi video chạy xong
    frames_dict = extract_frames(video_path, persons, num_frames=8)
    actions = {}

    # Nhận dạng hành động của từng người
    for person_id, frames in frames_dict.items():
        if len(frames) < 8:
            print(f"⚠️ Không đủ frames cho người {person_id}, bỏ qua...")
            continue
        
        inputs = processor(images=frames, return_tensors="pt")
        
        with torch.no_grad():
            outputs = action_model(**inputs)

        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        action = action_labels[predicted_class]
        actions[person_id] = action

    # Hiển thị kết quả sau khi video kết thúc
    print("📌 Kết quả dự đoán:")
    for person_id, action in actions.items():
        print(f"🧍 Người {person_id}: {action}")

# Chạy nhận dạng
if __name__ == "__main__":
    video_path = "1.mp4"  # Đổi thành đường dẫn video
    recognize_actions(video_path)
