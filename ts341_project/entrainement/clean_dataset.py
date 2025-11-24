from pathlib import Path
import shutil

def remove_unlabeled_images(dataset_dir):
    """Remove images that have empty label files"""
    dataset_path = Path(dataset_dir)

    for split in ['train', 'valid']:
        print(f"\nCleaning {split} set...")

        img_dir = dataset_path / split / "images"
        lbl_dir = dataset_path / split / "labels"

        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

        removed_images = 0
        removed_labels = 0
        kept = 0

        for img in images:
            label_file = lbl_dir / f"{img.stem}.txt"

            if not label_file.exists():
                # No label file at all
                print(f"  Removing {img.name} (no label file)")
                img.unlink()
                removed_images += 1
            elif label_file.stat().st_size == 0:
                # Empty label file
                print(f"  Removing {img.name} (empty label)")
                img.unlink()
                label_file.unlink()
                removed_images += 1
                removed_labels += 1
            else:
                kept += 1

        print(f"  Kept: {kept} images")
        print(f"  Removed: {removed_images} images, {removed_labels} empty labels")

def verify_remaining_dataset(dataset_dir):
    """Show stats after cleaning"""
    dataset_path = Path(dataset_dir)

    print("\n" + "=" * 60)
    print("CLEANED DATASET SUMMARY")
    print("=" * 60)

    for split in ['train', 'valid']:
        img_dir = dataset_path / split / "images"
        lbl_dir = dataset_path / split / "labels"

        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        labels = list(lbl_dir.glob("*.txt"))

        # Count annotations
        total_annotations = 0
        for lbl in labels:
            with open(lbl, 'r') as f:
                total_annotations += len([l for l in f.readlines() if l.strip()])

        print(f"\n{split.upper()}:")
        print(f"  Images: {len(images)}")
        print(f"  Labels: {len(labels)}")
        print(f"  Total annotations: {total_annotations}")

        if len(images) < 10:
            print(f"  ⚠️  WARNING: Very few images! You need more labeled data.")

if __name__ == "__main__":
    remove_unlabeled_images("datasets/my_drones/my_drones_final")
    verify_remaining_dataset("datasets/my_drones/my_drones_final")