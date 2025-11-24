# install dependencies first (run this in your terminal if needed):
# pip install ultralytics opencv-python
from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
import sys

def find_videos(base_path='videos', extensions=('.mkv', '.mp4', '.avi')):
    """Find all video files recursively"""
    videos = []
    base = Path(base_path)

    # Check if the directory exists
    if not base.exists():
        print(f"Error: Directory '{base_path}' not found!")
        print(f"Current working directory: {Path.cwd()}")
        return []

    for ext in extensions:
        videos.extend(base.rglob(f'*{ext}'))

    return sorted(videos)

def select_video():
    """Interactive video selection"""
    videos = find_videos()

    if not videos:
        print("No videos found!")
        return None

    print("\n=== Available Videos ===")
    print("0. Webcam")
    for idx, video in enumerate(videos, 1):
        # Show relative path for cleaner display
        try:
            rel_path = video.relative_to(Path.cwd())
            print(f"{idx}. {video.name} (in {rel_path.parent})")
        except ValueError:
            print(f"{idx}. {video.name} ({video.parent})")

    while True:
        try:
            choice = input("\nSelect video number (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                return None

            if choice.lower() == 'rand':
                return videos[np.random.randint(0, len(videos)-1)]

            if int(choice) == 0:
                return 0

            idx = int(choice) - 1
            if 0 <= idx < len(videos):
                return videos[idx]
            else:
                print(f"Please enter a number between 1 and {len(videos)}")
        except ValueError:
            print("Please enter a valid number")
            return videos[np.random.randint(0, len(videos)-1)]

# Load a pretrained YOLOv8 model
# model = YOLO("ts341_project/Anto/best_yolo8s_pretrained_model.pt")
model = YOLO("ts341_project/execution/best_weights_finetuned.pt")

video_path = select_video()

if video_path is None:
    print("No video selected. Exiting.")
    sys.exit(0)

print(f"\n✓ Selected: {video_path}")

# Open video file
if video_path == 0 :
    cap = cv2.VideoCapture(0)
else :
    cap = cv2.VideoCapture(str(video_path))

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    sys.exit(1)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video info: {fps:.2f} FPS, Resolution: {width}x{height}")

# === CONFIGURATION: Process one frame every X seconds ===
PROCESS_INTERVAL_SECONDS = 2  # Change this value as needed
frames_to_skip = int(fps * PROCESS_INTERVAL_SECONDS)

print(f"Processing 1 frame every {PROCESS_INTERVAL_SECONDS} seconds (every {frames_to_skip} frames)")
print("Processing video... Press 'q' to quit\n")

frame_count = 0
processed_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print(f"\nEnd of video reached")
        break

    # Only process every Nth frame
    if frame_count % frames_to_skip == 0:
        # Run YOLOv8 inference on the frame
        results = model(frame, show=False)
        annotated_frame = results[0].plot()
        annotated_frame = cv2.resize(annotated_frame, (1280, 720))

        # Display processing info on frame
        timestamp = frame_count / fps
        info_text = f"Time: {timestamp:.2f}s | Frame: {frame_count} | Processed: {processed_count}"
        cv2.putText(annotated_frame, info_text,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("YOLO Detection", annotated_frame)
        print(f"Processed frame {frame_count} at {timestamp:.2f}s")
        processed_count += 1

    frame_count += 1

    # Press 'q' to quit (small delay to allow key detection)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nStopped by user")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print(f"\nDone! Processed {processed_count} frames out of {frame_count} total frames")