# pyright: reportPrivateImportUsage=none

"""
Compare le temps de détection entre :
- YOLO standard
- YOLO + Simple Tracker
- YOLO + IMM Tracker
Affiche un graphique des détections par frame.
NB : un temps de détection élevé ne signifie pas forcément une meilleure détection.
"""

import cv2
from ultralytics import YOLO
from bytetrack_imm import IMMByteTrack
from simple_bytetrack import SimpleTrack
import matplotlib.pyplot as plt

# Charger le modèle YOLO
model = YOLO("best_yolo8s.pt")

cap = cv2.VideoCapture("../videos/ogives.mp4")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

yolo_detected = []
simple_tracker_detected = []
imm_detected = []

tracker_1 = SimpleTrack(max_age=80)
tracker_2 = IMMByteTrack(max_age=100)

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

    # --- YOLO + Personnalized tracker ---
    detections = []
    for result in results:
        if result.boxes is not None:
            detections.extend(result.boxes.xyxy.numpy())

    # Simple tracker
    tracks_simple = tracker_1.update(detections)
    imm_detected.append(1 if len(tracks_simple) > 0 else 0)

    # IMM
    tracks_imm = tracker_2.update(detections)
    simple_tracker_detected.append(1 if len(tracks_imm) > 0 else 0)
    
    frame_idx += 1

cap.release()

# --- Tracer le graphique ---
plt.figure(figsize=(12,4))
plt.scatter(range(len(yolo_detected)), yolo_detected, color='blue', s=4, label='YOLO standard')
plt.scatter(range(len(simple_tracker_detected)), simple_tracker_detected, color='green', s=4, label='YOLO + Simple Tracker')
plt.scatter(range(len(imm_detected)), imm_detected, color='red', s=4, label='YOLO + IMM')
plt.xlabel("Frame index")
plt.ylabel("Détection (1=oui, 0=non)")
plt.title("Comparaison des détections par frame")
plt.legend()
plt.show()
