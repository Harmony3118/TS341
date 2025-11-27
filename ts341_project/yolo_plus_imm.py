# pyright: reportPrivateImportUsage=none

import cv2
from ultralytics import YOLO
from bytetrack_imm import IMMByteTrack
import numpy as np
import math

# Paramètres de la caméra
image_height_px = 2048  # par ex. pour e-CAM_CUOAGK
FOV_v_deg = 85.64       # angle de vue vertical
H_drone = 0.24          # hauteur réelle du drone en mètres

# Calcul de la focale en pixels
FOV_v_rad = math.radians(FOV_v_deg)
f = image_height_px / (2 * math.tan(FOV_v_rad / 2))

# Centre de l'image
cx0 = image_height_px / 2
cy0 = image_height_px / 2  # si l'image n'est pas carrée, utiliser image_width / 2 pour cx0

# Load YOLO et tracker
model = YOLO("best_yolo8s.pt")
tracker = IMMByteTrack(max_age=60)

cap = cv2.VideoCapture("../videos/ogives.mp4")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


def main():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, show=False)

        boxes = []
        for result in results:
            if result.boxes is not None:
                for b, conf in zip(result.boxes.xyxy.cpu().numpy(),
                                   result.boxes.conf.cpu().numpy()):
                    if conf < 0.5:
                        continue
                    boxes.append(b)

        tracks = tracker.update(boxes)

        for tr in tracks:
            x1, y1, x2, y2 = tr["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            h_px = y2 - y1

            # Estimation distance Z
            Z = f * H_drone / h_px

            # Position X,Y réelle
            X = Z * (cx - cx0) / f # Horizintal
            Y = Z * (cy - cy0) / f # Vertical

            # Affichage
            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)
            cv2.putText(frame, f"ID:{tr['track_id']} Depth:{Z:.2f}m", (int(cx)+5,int(cy)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # On peut aussi afficher X,Y si nécessaire
            print(f"Track {tr['track_id']}: X (horizontal)={X:.2f}, Y (vertical)={Y:.2f}, Depth={Z:.2f}")

        cv2.imshow("YOLO + IMM ByteTrack", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
