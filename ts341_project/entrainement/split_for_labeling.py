# split_for_labeling.py
from pathlib import Path
import shutil


def split_into_batches(images_dir, output_dir, batch_size=100):
    """Split images into small batches for easy upload"""
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all images
    images = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))

    print(f"Found {len(images)} images")
    print(f"Creating batches of {batch_size} images...")

    # Split into batches
    num_batches = (len(images) + batch_size - 1) // batch_size

    for i in range(0, len(images), batch_size):
        batch_num = i // batch_size + 1
        batch_dir = output_dir / f"batch_{batch_num:02d}"
        batch_dir.mkdir(exist_ok=True)

        batch_images = images[i : i + batch_size]

        for img in batch_images:
            shutil.copy(img, batch_dir / img.name)

        print(f"  Batch {batch_num:02d}/{num_batches}: {len(batch_images)} images")

    print(f"\n✓ Created {num_batches} batches in: {output_dir}")
    print(f"\nTo label:")
    print(f"1. Start Label Studio: label-studio start")
    print(f"2. Create project and upload each batch")
    print(f"3. Label the images")
    print(f"4. Move to next batch")


if __name__ == "__main__":
    split_into_batches(
        images_dir="ts341_project/Anto/datasets/my_drones/unlabeled_frames",
        output_dir="ts341_project/Anto/datasets/my_drones/batches_for_upload",
        batch_size=100,  # Small batches = stable uploads
    )
