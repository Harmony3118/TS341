# pyright: reportPrivateImportUsage=none

import cv2
from ultralytics import YOLO
from simple_bytetrack import SimpleTrack
import math
from typing import List, Dict, Any


# 1. Video loading
# ================
print("\nHello !\n\nSystem is ready for use. First, choose a video.\nFormat example : video_1.mkv")
print("Note: the video won't be loaded if there are whitespaces in the name.")

video: str = input("\nVideo: ").strip()
cap: cv2.VideoCapture = cv2.VideoCapture(f"../videos/{video}")

if not cap.isOpened():
    print("\nError: Could not open video. You may have typed it wrong, or placed it in the wrong folder.\n\n")
    exit()


# 2. Camera choice
# ================
print("\nVideo found.\nThen, choose the corresponding camera model:")
print("0 = Wide e-CAM")
print("1 = Narrow e-CAM")
print("2 = e-CAM20")
camera: str = input("\nCamera: ").strip()

if camera == "0":
    # -------- Wide e-CAM --------
    image_width_px: int  = 3840
    image_height_px: int = 2160
    FOV_v_deg: float = 67.04  # Vertical FOV
    print("Selected camera: Wide e-CAM")

elif camera == "1":
    # -------- Narrow e-CAM --------
    image_width_px  = 3840
    image_height_px = 2160
    FOV_v_deg = 38.83
    print("Selected camera: Narrow e-CAM")

elif camera == "2":
    # -------- e-CAM20 --------
    image_width_px  = 2432
    image_height_px = 2048
    FOV_v_deg = 67
    print("Selected camera: e-CAM20")

else:
    print("Invalid choice.")
    exit()


# 3. YOLO Version
# ===============
print("\nThe detection works with the YOLO model. Which parameters would you like to try?")
print("0 = Default parameters")
print("1 = Fine-tuned parameters")
yolo_param: str = input("\nParameters: ").strip()

if yolo_param == "0":
    # -------- Default --------
    model: YOLO = YOLO("best_yolo8s.pt")
    print("Selected model: default")

else:
    # -------- Fine-tuned --------
    model: YOLO = YOLO("best_finetuned_2.pt")
    print("Selected model: fine-tuned")

tracker: SimpleTrack = SimpleTrack(max_age=50) # The age is the numer of frame the tracker will keep predicting


# 4. Video output
# ===============
print("\nWould you like to activate the video output?")
print("0 = No")
print("1 = Yes")
video_output: str = input("\nVideo output: ").strip()

if video_output == "0":
    # -------- Off --------
    show_video: bool = False
    print("The video won't show up.\nQuit: 'Ctrl+C'")

else:
    # -------- On --------
    show_video: bool = True
    print("You will see the video.\nQuit: 'q'")


# 5. Focal computation in px
# ==========================
FOV_v_rad: float = math.radians(FOV_v_deg)
f: float = image_height_px / (2 * math.tan(FOV_v_rad / 2))

# Image center
cx0: float = image_width_px / 2
cy0: float = image_height_px / 2

H_drone: float = 0.1352  # Height in metters of the drone


# 6. Main loop
# ============
def main() -> None:
    frame_id: int = 0

    while True:
        ret: bool
        frame: Any
        ret, frame = cap.read()
        if not ret:
            break

        results: Any = model(frame, show=False, tracker=tracker)

        # Detections
        boxes: List[List[float]] = []
        for result in results:
            if result.boxes is not None:
                for b, conf in zip(result.boxes.xyxy.cpu().numpy(),
                                   result.boxes.conf.cpu().numpy()):
                    if conf >= 0.5:
                        boxes.append(b)

        tracks: List[Dict[str, Any]] = tracker.update(boxes)

        for tr in tracks:
            x1, y1, x2, y2 = tr["bbox"]
            cx: float = (x1 + x2) / 2
            cy: float = (y1 + y2) / 2
            h_px: float = y2 - y1

            # 7. 3D position estimation
            # =========================
            Z: float = f * H_drone / h_px
            X: float = Z * (cx - cx0) / f
            Y: float = Z * (cy - cy0) / f

            # Display
            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)
            cv2.putText(frame,
                        f"ID:{tr['track_id']} Z={Z:.2f}m",
                        (int(cx) + 5, int(cy) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)

            print(f"[Track {frame_id}]  X={X:.2f} m,  Y={Y:.2f} m,  Z={Z:.2f} m")
            frame_id += 1

        if show_video:
            cv2.imshow("YOLO + IMM ByteTrack + Position 3D", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
