# merge_labeled_batches.py (UPDATED - Fix the split function)
from pathlib import Path
import shutil
import zipfile
import tempfile


def extract_and_merge_exports(exports_dir, output_dir, original_images_dir):
    """
    Merge multiple Label Studio YOLO exports into one dataset

    Args:
        exports_dir: Directory containing exported ZIP files from Label Studio
        output_dir: Where to create the merged dataset
        original_images_dir: Directory with original unlabeled images
    """
    exports_path = Path(exports_dir)
    output_path = Path(output_dir)
    original_images_path = Path(original_images_dir)

    # Create output structure
    for split in ["images", "labels"]:
        (output_path / split).mkdir(parents=True, exist_ok=True)

    # Find all export ZIPs
    export_zips = list(exports_path.glob("*.zip"))

    if not export_zips:
        print(f"No ZIP files found in {exports_dir}")
        print("Export your labeled batches from Label Studio first!")
        return

    print(f"Found {len(export_zips)} export files")

    # Build a mapping of original images by filename
    print(f"\nIndexing original images from: {original_images_path}")
    original_images_map = {}
    for img in original_images_path.rglob("*.jpg"):
        original_images_map[img.name] = img
    for img in original_images_path.rglob("*.png"):
        original_images_map[img.name] = img

    print(f"Found {len(original_images_map)} original images")

    total_images = 0
    total_labels = 0

    # Process each export
    for zip_file in export_zips:
        print(f"\nProcessing: {zip_file.name}")

        # Extract to temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            temp_path = Path(temp_dir)

            # Find labels in the export
            labels_src = None

            # Common patterns for labels
            for pattern in ["labels", "train/labels", "valid/labels"]:
                candidate = temp_path / pattern
                if candidate.exists():
                    labels_src = candidate
                    break

            # Also check if files are in root
            if not labels_src:
                lbls = list(temp_path.rglob("*.txt"))
                if lbls and lbls[0].name != "classes.txt":
                    labels_src = lbls[0].parent

            if labels_src and labels_src.exists():
                labels = [
                    l for l in labels_src.glob("*.txt") if l.name != "classes.txt"
                ]

                for lbl in labels:
                    # Label Studio adds a prefix to filenames, remove it
                    # Find and copy corresponding image from original images
                    img_name_jpg = lbl.stem[9:] + ".jpg"  # Remove 'tasks_1_' prefix
                    img_name_png = lbl.stem[9:] + ".png"

                    if img_name_jpg in original_images_map:
                        # Copy image with SAME name as label (without the prefix)
                        new_img_name = lbl.stem + ".jpg"
                        shutil.copy(
                            original_images_map[img_name_jpg],
                            output_path / "images" / new_img_name,
                        )
                        total_images += 1
                    elif img_name_png in original_images_map:
                        # Copy image with SAME name as label (without the prefix)
                        new_img_name = lbl.stem + ".png"
                        shutil.copy(
                            original_images_map[img_name_png],
                            output_path / "images" / new_img_name,
                        )
                        total_images += 1
                    else:
                        print(f"  ⚠️  Warning: No image found for {lbl.name}")
                        print(f"      Looking for: {img_name_jpg} or {img_name_png}")

                    # Copy label
                    shutil.copy(lbl, output_path / "labels" / lbl.name)

                total_labels += len(labels)
                print(f"  Copied {len(labels)} labels")
                print(f"  Copied images")

    print(f"\n{'='*60}")
    print(f"MERGE COMPLETE")
    print(f"{'='*60}")
    print(f"Total images: {total_images}")
    print(f"Total labels: {total_labels}")
    print(f"Output: {output_path}")

    if total_images == 0:
        print("\n⚠️  WARNING: No images were copied!")
        print("Check that original_images_dir is correct:")
        print(f"  {original_images_path}")
        return None

    # Create data.yaml with correct class name
    yaml_content = f"""path: {output_path.absolute()}
train: images
val: images  # Will split later

names:
  0: drones
"""
    with open(output_path / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"✓ Created data.yaml")

    return total_images, total_labels


def split_into_train_valid(merged_dir, train_ratio=0.8):
    """Split merged dataset into train and valid"""
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("Error: sklearn not installed")
        print("Install it with: poetry add scikit-learn")
        return

    merged_path = Path(merged_dir)

    # Get all labels first (they determine what we have)
    labels = list((merged_path / "labels").glob("*.txt"))

    if not labels:
        print("Error: No labels found in merged dataset!")
        return

    # For each label, find matching image
    labeled_pairs = []
    missing_images = []

    for lbl in labels:
        # Try to find matching image
        img_jpg = (merged_path / "images" / lbl.stem).with_suffix(".jpg")
        img_png = (merged_path / "images" / lbl.stem).with_suffix(".png")

        if img_jpg.exists():
            labeled_pairs.append((img_jpg, lbl))
        elif img_png.exists():
            labeled_pairs.append((img_png, lbl))
        else:
            missing_images.append(lbl.name)

    print(f"\n{'='*60}")
    print(f"SPLITTING DATASET")
    print(f"{'='*60}")
    print(f"Total labels: {len(labels)}")
    print(f"Matched image+label pairs: {len(labeled_pairs)}")

    if missing_images:
        print(f"⚠️  Missing images for {len(missing_images)} labels:")
        for name in missing_images[:5]:  # Show first 5
            print(f"    - {name}")
        if len(missing_images) > 5:
            print(f"    ... and {len(missing_images) - 5} more")

    if len(labeled_pairs) == 0:
        print("Error: No valid image+label pairs found!")
        print("\nDebug info:")
        print(f"Labels dir: {merged_path / 'labels'}")
        print(f"Images dir: {merged_path / 'images'}")

        # Show a few examples
        if labels:
            print(f"\nExample label: {labels[0].name}")

        images = list((merged_path / "images").glob("*"))
        if images:
            print(f"Example image: {images[0].name}")

        return

    # Split
    train_pairs, valid_pairs = train_test_split(
        labeled_pairs, train_size=train_ratio, random_state=42
    )

    print(f"Train: {len(train_pairs)}")
    print(f"Valid: {len(valid_pairs)}")

    # Create directories
    final_path = merged_path.parent / "my_drones_final"
    for split in ["train", "valid"]:
        (final_path / split / "images").mkdir(parents=True, exist_ok=True)
        (final_path / split / "labels").mkdir(parents=True, exist_ok=True)

    # Copy train files
    for img, lbl in train_pairs:
        shutil.copy(img, final_path / "train" / "images" / img.name)
        shutil.copy(lbl, final_path / "train" / "labels" / lbl.name)

    # Copy valid files
    for img, lbl in valid_pairs:
        shutil.copy(img, final_path / "valid" / "images" / img.name)
        shutil.copy(lbl, final_path / "valid" / "labels" / lbl.name)

    # Create final data.yaml
    yaml_content = f"""path: {final_path.absolute()}
train: train/images
val: valid/images

names:
  0: drones
"""
    with open(final_path / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"\n✓ Final dataset ready: {final_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Step 1: Merge all labeled batches
    result = extract_and_merge_exports(
        exports_dir="datasets/my_drones/label_studio_exports",
        output_dir="datasets/my_drones/merged",
        original_images_dir="datasets/my_drones/unlabeled_frames",
    )

    # Step 2: Split into train/valid (only if merge succeeded)
    if result:
        split_into_train_valid(merged_dir="datasets/my_drones/merged", train_ratio=0.8)
