import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18, R3D_18_Weights
from PIL import Image
import numpy as np
import asyncio
from googletrans import Translator
from ultralytics import YOLO

# Load pre-trained ResNet3D model
weights = R3D_18_Weights.KINETICS400_V1
model = r3d_18(weights=weights)
model.eval()

# Load YOLOv8 model for person detection
yolo_model = YOLO("yolov8n.pt")

# Load full Kinetics-400 labels
def load_kinetics400_labels():
    kinetics400_labels = {}
    with open("kinetics_400_labels.txt", "r") as f:
        for i, line in enumerate(f.readlines()):
            kinetics400_labels[i] = line.strip()
    return kinetics400_labels

action_labels = load_kinetics400_labels()

# Translator instance
translator = Translator()

def translate_to_vietnamese(text):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    translation = loop.run_until_complete(translator.translate(text, src="en", dest="vi"))
    return translation.text

# Video processing function
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])
    
    while len(frames) < 16:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform(frame)
        frames.append(frame)
    
    cap.release()
    if len(frames) < 16:
        return None  # Need at least 16 frames
    
    video_tensor = torch.stack(frames[:16])  # Take first 16 frames
    video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # Convert to (1, 3, 16, 112, 112)
    return video_tensor

# Function to display video with bounding boxes
def show_video_with_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect people in the frame
        results = yolo_model(frame)
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imshow("Video", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Predict function
def predict_action(video_path):
    show_video_with_detection(video_path)
    video_tensor = process_video(video_path)
    if video_tensor is None:
        print("Video quá ngắn để nhận dạng")
        return
    
    with torch.no_grad():
        output = model(video_tensor)
        prediction = torch.argmax(output, dim=1).item()
    
    if prediction in action_labels:
        action_en = action_labels[prediction]
        # action_vn = translate_to_vietnamese(action_en)  # Gọi hàm dịch đồng bộ
        print(f"Dự đoán hành động: {action_en}")
    else:
        print("Dự đoán không xác định")

# Example usage
video_path = "V_797.mp4"  # Đổi đường dẫn thành video cần nhận dạng
predict_action(video_path)
