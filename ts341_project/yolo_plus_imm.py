# pyright: reportPrivateImportUsage=none

import cv2
from ultralytics import YOLO
from simple_bytetrack import SimpleTrack
import math


# 1. Camera choice
# ================
print("Choose a camera model :")
print("1 = e-CAM20")
print("2 = Wide e-CAM")
choice = input("Camera (1/2) : ").strip()

if choice == "1":
    # -------- e-CAM20 --------
    image_width_px  = 2432
    image_height_px = 2048
    FOV_v_deg = 67          # Vertical FOV
    H_drone = 0.1352        # Height in metters of the drone
    print("Selected camera : e-CAM20")

elif choice == "2":
    # -------- Wide e-CAM --------
    image_width_px  = 2432
    image_height_px = 2048
    FOV_v_deg = 67.04
    H_drone = 0.1352
    print("Selected camera : Wide e-CAM")

else:
    print("Invalid choice.")
    exit()


# 2. Video loading
# ================
cap = cv2.VideoCapture("../videos/arriere.mp4")

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


# 3. Focal computation in px
# ==========================
FOV_v_rad = math.radians(FOV_v_deg)
f = image_height_px / (2 * math.tan(FOV_v_rad / 2))

# Image center
cx0 = image_width_px / 2
cy0 = image_height_px / 2


# 4. YOLO + Tracker loading
# =========================
model = YOLO("best_yolo8s.pt")
tracker = SimpleTrack(max_age=80)


# 5. Main loop
# ============
def main():
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, show=False)

        # Detections
        boxes = []
        for result in results:
            if result.boxes is not None:
                for b, conf in zip(result.boxes.xyxy.cpu().numpy(),
                                   result.boxes.conf.cpu().numpy()):
                    if conf >= 0.5:
                        boxes.append(b)

        tracks = tracker.update(boxes)

        for tr in tracks:
            x1, y1, x2, y2 = tr["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            h_px = y2 - y1

            # 5. 3D position estimation
            # =========================
            Z = f * H_drone / h_px
            X = Z * (cx - cx0) / f
            Y = Z * (cy - cy0) / f

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