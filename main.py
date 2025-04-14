# main.py
# ตรวจจับลูกบอคเซียสีแดงด้วย YOLOv5 (PyTorch) โดยไม่ใช้ torch.hub

import cv2
import torch
import numpy as np
import sys
from pathlib import Path

# === ชี้ path ไปยังโฟลเดอร์ yolov5 ===
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / 'yolov5'
sys.path.append(str(ROOT))

# === import ฟังก์ชันที่จำเป็นจาก YOLOv5 ===
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# === โหลดโมเดล ===
device = select_device('')
model = DetectMultiBackend('best.pt', device=device)
model.eval()

# === เปิดกล้อง ===
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # === เตรียมภาพ ===
    img = cv2.resize(frame, (640, 640))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)

    # === รัน YOLO ===
    pred = model(img_tensor, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.5)

    # === วาดกล่องบน frame ต้นฉบับ ===
    for det in pred:
        if len(det):
            for *xyxy, conf, cls in det:
                xyxy = [int(x.item()) for x in xyxy]
                label = f'redball {conf:.2f}'
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
                cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('YOLOv5 Red Ball Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()