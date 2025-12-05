# pyright: reportPrivateImportUsage=none
import cv2
from ultralytics import YOLO
import math
from typing import Optional, Any, Tuple


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

BBox = Tuple[float, float, float, float] # Type alias for bounding box (x1, y1, x2, y2)


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
    """
    Runs YOLO object detection with ByteTrack tracker.
    Selects a single object to track across frames.
    Computes 3D position from bounding boxes and camera parameters.
    Optionally displays video with annotations.
    """

    frame_id: int = 0
    target_id: Optional[int] = None  # Track the first detected object

    while True:
        ret, frame = cap.read()
        frame: Any
        if not ret:
            break

        # YOLO detection + ByteTrack
        results = model(frame, show=False, tracker="bytetrack.yaml")

        detected_bbox: Optional[BBox] = None

        # Iterate through detection results
        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            ids   = result.boxes.id.cpu().numpy() if result.boxes.id is not None else [None]*len(boxes)

            for b, conf, tid in zip(boxes, confs, ids):
                if conf < 0.6:
                    continue

                # Select first object to track
                if target_id is None:
                    target_id = tid

                # Only track the selected object
                if tid == target_id:
                    detected_bbox = tuple(b)
                    break

            if detected_bbox is not None:
                break

        if detected_bbox is None:
            # Skip frame if object not detected
            continue

        # Compute 3D position
        x1, y1, x2, y2 = detected_bbox
        cx: float = (x1 + x2) / 2
        cy: float = (y1 + y2) / 2
        h_px: float = y2 - y1

        Z: float = f * H_drone / h_px
        X: float = Z * (cx - cx0) / f
        Y: float = Z * (cy - cy0) / f

        print(f"[Frame {frame_id}]  X={X:.2f} m, Y={Y:.2f} m, Z={Z:.2f} m")
        frame_id += 1

        # Optional display
        if show_video:
            cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)
            cv2.putText(frame, f"Z={Z:.2f}m", (int(cx)+5, int(cy)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("YOLO + ByteTrack + 3D Position", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()