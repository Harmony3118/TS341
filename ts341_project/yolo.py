# pyright: reportPrivateImportUsage=none

import cv2
from ultralytics import YOLO
import numpy as np

# Load a pretrained YOLOv8 model
# Options: 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', etc.
model = YOLO("best_yolo8s.pt")

# Open video file or webcam (0 = default webcam)
cap = cv2.VideoCapture("../videos/arriere.mp4")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


def main():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 inference on the frame with ByteTrack
        results = model(
            frame,
            show=False,               # Disable YOLOv8 window
            tracker="bytetrack.yaml"  # Use ByteTrack for tracking
        )

        # Annotate the frame with bounding boxes and tracking info
        annotated_frame = results[0].plot()
        annotated_frame = cv2.resize(annotated_frame, (720, 720))
        cv2.imshow("YOLOv8 Detection", annotated_frame)

        # Access detection results programmatically
        for result in results:
            boxes = result.boxes
            print(
                f"Coords: {boxes.xyxy}, \n"
                f"Box id: {boxes.cls}, \n"
                f"Confidence: {boxes.conf}"
            )

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
