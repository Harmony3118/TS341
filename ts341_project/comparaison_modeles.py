# pyright: reportPrivateImportUsage=none

import cv2
from ultralytics import YOLO
from bytetrack_imm import IMMByteTrack
import matplotlib.pyplot as plt

# Charger le modèle YOLO
model = YOLO("best_yolo8s.pt")

cap = cv2.VideoCapture("../videos/arriere.mp4")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

yolo_detected = []
imm_detected = []

tracker = IMMByteTrack(max_age=30)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- YOLO standard ---
    results = model(frame, show=False)
    has_yolo = False
    for result in results:
        boxes = result.boxes
        if boxes is not None and len(boxes.xyxy) > 0:
            has_yolo = True
            break
    yolo_detected.append(1 if has_yolo else 0)

    # --- YOLO + IMM ---
    detections = []
    for result in results:
        if result.boxes is not None:
            detections.extend(result.boxes.xyxy.numpy())

    # Passer les détections au tracker IMM
    tracks = tracker.update(detections)
    imm_detected.append(1 if len(tracks) > 0 else 0)
    
    frame_idx += 1

cap.release()

# --- Tracer le graphique ---
plt.figure(figsize=(12,4))
plt.scatter(range(len(yolo_detected)), yolo_detected, color='blue', s=10, label='YOLO standard')
plt.scatter(range(len(imm_detected)), imm_detected, color='red', s=10, label='YOLO + IMM')
plt.xlabel("Frame index")
plt.ylabel("Détection (1=oui, 0=non)")
plt.title("Comparaison des détections par frame")
plt.legend()
plt.show()
