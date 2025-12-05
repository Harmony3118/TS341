# pyright: reportPrivateImportUsage=none
import json
import cv2
from ultralytics import YOLO
from single_object_tracker import SingleObjectTracker
import math
from typing import Tuple


def video_to_json_output(v_filename):
    """Given a video filename, return a new filename to output JSON data to"""
    return v_filename.replace(".mp4", "_yolo.mp4.json")


# 1. Video loading
# ================
print(
    "\nHello !\n\nSystem is ready for use. First, choose a video.\nFormat example : video_1.mkv"
)
print("Note: the video won't be loaded if there are whitespaces in the name.")

video: str = input("\nVideo: ").strip()
video_filename: str = f"../videos/{video}"
cap: cv2.VideoCapture = cv2.VideoCapture(video_filename)

if not cap.isOpened():
    print(
        "\nError: Could not open video. You may have typed it wrong, or placed it in the wrong folder.\n\n"
    )
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
    image_width_px: int = 3840
    image_height_px: int = 2160
    FOV_v_deg: float = 67.04  # Vertical FOV
    print("Selected camera: Wide e-CAM")

elif camera == "1":
    # -------- Narrow e-CAM --------
    image_width_px = 3840
    image_height_px = 2160
    FOV_v_deg = 38.83
    print("Selected camera: Narrow e-CAM")

elif camera == "2":
    # -------- e-CAM20 --------
    image_width_px = 2432
    image_height_px = 2048
    FOV_v_deg = 67
    print("Selected camera: e-CAM20")

else:
    print("Invalid choice.")
    exit()


# 3. YOLO Version
# ===============
print(
    "\nThe detection works with the YOLO model. Which parameters would you like to try?"
)
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

BBox = Tuple[float, float, float, float]  # Bounding Box for the tracker


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
    Run YOLO object detection and single-object tracking on a video.
    Updates track, computes 3D position, optionally displays the video,
    and prints X, Y, Z coordinates for each frame.
    """

    if show_video:
        win_name = "YOLO + IMM ByteTrack + Position 3D"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    centers = []
    frame_id: int = 0
    tracker: SingleObjectTracker | None = None  # Will initialize after first detection

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, show=False)

        # Get first detection with confidence >= 0.5
        detected_bbox: BBox | None = None
        for result in results:
            if result.boxes is not None:
                for b, conf in zip(
                    result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()
                ):
                    if conf >= 0.5:
                        detected_bbox = tuple(b)
                        break
            if detected_bbox is not None:
                break

        if detected_bbox is None:
            # No detection in this frame, skip
            continue

        # Initialize tracker on first detection
        if tracker is None:
            tracker = SingleObjectTracker(detected_bbox)

        # Update tracker
        tracker.update(detected_bbox)
        x1, y1, x2, y2 = tracker.bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        h_px = y2 - y1

        # 3D position computation
        Z = f * H_drone / h_px
        X = Z * (cx - cx0) / f
        Y = Z * (cy - cy0) / f

        # save tracked drone center
        centers.append((int(cx), int(cy)))

        print(f"[Frame {frame_id}]  X={X:.2f} m,  Y={Y:.2f} m,  Z={Z:.2f} m")
        frame_id += 1

        # Display
        if show_video:
            cv2.circle(frame, (int(cx), int(cy)), 8, (0, 255, 0), -1)
            cv2.putText(
                frame,
                f"Z={Z:.2f}m",
                (int(cx) + 5, int(cy) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            # cv2.imshow("YOLO + Single Object Tracker + Position 3D", frame)
            cv2.imshow(win_name, frame)
            cv2.resizeWindow(win_name, 1280, 720)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    # write list of centers to JSON file
    # to be used for performance measurement
    with open(video_to_json_output(video_filename), "w") as file:
        file.write(json.dumps(centers))  # assume only one drone in image


if __name__ == "__main__":
    main()
