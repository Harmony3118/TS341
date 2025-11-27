import cv2
from pathlib import Path


def extract_frames(video_path, output_dir, interval_seconds=2):
    """Extract frames from video at regular intervals"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)

    frame_count = 0
    saved_count = 0

    print(f"Processing {video_path.name}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Save frame
            video_name = video_path.stem
            frame_name = f"{video_name}_frame_{saved_count:05d}.jpg"
            cv2.imwrite(str(output_dir / frame_name), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames from {video_path.name}")
    return saved_count


def extract_all_videos(videos_dir, output_dir, interval_seconds=2):
    """Extract frames from all videos in directory"""
    videos_dir = Path(videos_dir)
    output_dir = Path(output_dir)

    # Find all videos
    video_files = list(videos_dir.rglob("*.mkv")) + list(videos_dir.rglob("*.mp4"))

    print(f"Found {len(video_files)} videos")

    total_frames = 0
    for video in video_files:
        count = extract_frames(video, output_dir, interval_seconds)
        total_frames += count

    print(f"\nTotal: {total_frames} frames extracted")


if __name__ == "__main__":
    extract_all_videos(
        videos_dir="videos",
        output_dir="datasets/my_drones/unlabeled_frames",
        interval_seconds=1,  # Extract 1 frame every seconds
    )
