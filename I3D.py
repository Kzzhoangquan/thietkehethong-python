import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from ultralytics import YOLO
from PIL import Image
from torchvision.models.video import r2plus1d_18  # Load mô hình R(2+1)D-18

# Load mô hình YOLO để phát hiện người
yolo_model = YOLO("yolov8n.pt")

# Load mô hình R(2+1)D-18 đã huấn luyện trên Kinetics-400
num_classes = 400  # Kinetics-400 có 400 hành động
r2plus1d_model = r2plus1d_18(pretrained=True)
r2plus1d_model.fc = torch.nn.Linear(r2plus1d_model.fc.in_features, num_classes)  # Điều chỉnh lớp cuối
r2plus1d_model.eval()

# Load nhãn hành động
with open("kinetics_400_labels.txt", "r") as f:
    action_labels = [line.strip() for line in f.readlines()]

print(f"Số lượng nhãn hành động: {len(action_labels)}")  # Kiểm tra số lượng nhãn

# Chuyển đổi video clip thành tensor phù hợp với R(2+1)D-18
def transform_video(frames):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])
    return torch.stack([transform(frame) for frame in frames]).permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)

# Xử lý video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_buffer = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_buffer.append(frame)
    cap.release()
    
    if len(frame_buffer) < 16:
        print("Video quá ngắn để nhận dạng")
        return
    
    for i in range(0, len(frame_buffer) - 16, 4):  # Lấy từng đoạn 16 frame
        frames = frame_buffer[i:i+16]
        results = yolo_model(frames[0])  # Phát hiện người trên frame đầu tiên
        
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                person_frames = [Image.fromarray(cv2.cvtColor(f[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)) for f in frames]
                if not person_frames:
                    continue
                
                video_tensor = transform_video(person_frames)
                
                with torch.no_grad():
                    output = r2plus1d_model(video_tensor)
                    prediction = torch.argmax(output, dim=1).item()
                    
                    if prediction >= len(action_labels):
                        print(f"Lỗi: Chỉ số dự đoán {prediction} vượt quá số lượng nhãn ({len(action_labels)})")
                        action_name = "Unknown"
                    else:
                        action_name = action_labels[prediction]
                
                cv2.rectangle(frames[0], (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frames[0], action_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow("Action Recognition", frames[0])
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

# Gọi hàm nhận diện
test_video = "1.mp4"  # Thay đường dẫn video
process_video(test_video)
