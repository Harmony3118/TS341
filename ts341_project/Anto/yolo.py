# install dependencies first (run this in your terminal if needed):
# pip install ultralytics opencv-python

from ultralytics import YOLO
import cv2
import numpy as np

# Load a pretrained YOLOv8 model (you can choose 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', etc.)
model = YOLO("best_yolo8s.pt")

# Open webcam (0 = default webcam)
cap = cv2.VideoCapture("../videos/cam1_10_31_11_50_04.mkv")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame

    results = model(frame, show=False)  # disable YOLOv8 window
    annotated_frame = results[0].plot()
    annotated_frame = cv2.resize(annotated_frame, (1920, 1080))
    cv2.imshow("YOLOv8 Detection", annotated_frame)


    # Optional: access detection results programmatically
    # for result in results:
    #     boxes = result.boxes
    #     print(boxes.xyxy, boxes.cls, boxes.conf)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
