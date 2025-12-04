# pyright: reportPrivateImportUsage=none

import cv2
from ultralytics import YOLO
from simple_bytetrack import SimpleTrack
import math
from typing import List, Dict, Any


# 1. Video loading
# ================
print("\nHello !\n\nSystem is ready for use. First, choose a video.")
print("Notes: \n+ The video won't be loaded if there are whitespaces in the name.\n+ Do not include the extension, just the name is fine.\n")

video: str = input("Video name : ").strip()
cap: cv2.VideoCapture = cv2.VideoCapture(f"../videos/{video}.mp4")

if not cap.isOpened():
    print("\nError: Could not open video. You may have typed it wrong, or placed it in the wrong folder.\n\n")
    exit()


# 2. Camera choice
# ================
print("\nPerfect. Then, choose the corresponding camera model :")
print("0 = Wide e-CAM")
print("1 = Narrow e-CAM")
print("2 = e-CAM20")
camera: str = input("Camera : ").strip()

if camera == "0":
    # -------- Wide e-CAM --------
    image_width_px: int  = 3840
    image_height_px: int = 2160
    FOV_v_deg: float = 67.04  # Vertical FOV
    print("Selected camera : Wide e-CAM")

elif camera == "1":
    # -------- Narrow e-CAM --------
    image_width_px  = 3840
    image_height_px = 2160
    FOV_v_deg = 38.83
    print("Selected camera : Narrow e-CAM")

elif camera == "2":
    # -------- e-CAM20 --------
    image_width_px  = 2432
    image_height_px = 2048
    FOV_v_deg = 67
    print("Selected camera : e-CAM20")

else:
    print("Invalid choice.")
    exit()


# 3. Focal computation in px
# ==========================
FOV_v_rad: float = math.radians(FOV_v_deg)
f: float = image_height_px / (2 * math.tan(FOV_v_rad / 2))

# Image center
cx0: float = image_width_px / 2
cy0: float = image_height_px / 2

H_drone: float = 0.1352  # Height in metters of the drone


# 4. YOLO + Tracker loading
# =========================
model: YOLO = YOLO("best_yolo8s.pt")
tracker: SimpleTrack = SimpleTrack(max_age=80)


# 5. Main loop
# ============
def main() -> None:
    frame_id: int = 0

    while True:
        ret: bool
        frame: Any
        ret, frame = cap.read()
        if not ret:
            break

        results: Any = model(frame, show=False)

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

            # 5. 3D position estimation
            # =========================
            Z: float = f * H_drone / h_px
            X: float = Z * (cx - cx0) / f
            Y: float = Z * (cy - cy0) / f

            # Affichage
            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)
            cv2.putText(frame,
                        f"ID:{tr['track_id']} Z={Z:.2f}m",
                        (int(cx) + 5, int(cy) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)

            print(f"[Track {frame_id}]  X={X:.2f} m,  Y={Y:.2f} m,  Z={Z:.2f} m")
            frame_id += 1

        cv2.imshow("YOLO + IMM ByteTrack + Position 3D", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
