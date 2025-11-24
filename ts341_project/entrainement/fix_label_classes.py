from pathlib import Path

def fix_label_classes(dataset_dir):
    """
    Fix label files that have incorrect class IDs
    Convert all class IDs to 0 (since we only have one class: drones)
    """
    dataset_path = Path(dataset_dir)

    for split in ['train', 'valid']:
        lbl_dir = dataset_path / split / "labels"

        if not lbl_dir.exists():
            continue

        label_files = list(lbl_dir.glob("*.txt"))

        print(f"\n{split.upper()}:")
        print(f"Checking {len(label_files)} label files...")

        fixed_count = 0
        removed_count = 0

        for lbl_file in label_files:
            with open(lbl_file, 'r') as f:
                lines = f.readlines()

            if not lines:
                # Empty file, remove it
                lbl_file.unlink()
                # Also remove corresponding image
                img_jpg = dataset_path / split / "images" / f"{lbl_file.stem}.jpg"
                img_png = dataset_path / split / "images" / f"{lbl_file.stem}.png"
                if img_jpg.exists():
                    img_jpg.unlink()
                if img_png.exists():
                    img_png.unlink()
                removed_count += 1
                continue

            # Fix class IDs
            new_lines = []
            changed = False

            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id, x, y, w, h = parts

                # Convert any class ID to 0
                if class_id != '0':
                    class_id = '0'
                    changed = True

                new_lines.append(f"{class_id} {x} {y} {w} {h}\n")

            if changed:
                with open(lbl_file, 'w') as f:
                    f.writelines(new_lines)
                fixed_count += 1

        print(f"  Fixed: {fixed_count} files")
        print(f"  Removed: {removed_count} empty files")

    print("\n" + "=" * 60)
    print("LABELS FIXED!")
    print("=" * 60)

def verify_labels(dataset_dir):
    """Verify all labels are correct now"""
    dataset_path = Path(dataset_dir)

    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    for split in ['train', 'valid']:
        lbl_dir = dataset_path / split / "labels"

        if not lbl_dir.exists():
            continue

        label_files = list(lbl_dir.glob("*.txt"))

        total_annotations = 0
        class_counts = {}
        issues = []

        for lbl_file in label_files:
            with open(lbl_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        issues.append(f"{lbl_file.name}:{line_num} - Wrong format")
                        continue

                    try:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:])

                        # Check ranges
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            issues.append(f"{lbl_file.name}:{line_num} - Out of bounds")
                            continue

                        if class_id != 0:
                            issues.append(f"{lbl_file.name}:{line_num} - Class ID {class_id} (should be 0)")
                            continue

                        total_annotations += 1
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1

                    except ValueError:
                        issues.append(f"{lbl_file.name}:{line_num} - Invalid values")

        print(f"\n{split.upper()}:")
        print(f"  Label files: {len(label_files)}")
        print(f"  Total annotations: {total_annotations}")
        print(f"  Class distribution: {class_counts}")

        if issues:
            print(f"\n  ⚠️  Issues found ({len(issues)}):")
            for issue in issues[:10]:  # Show first 10
                print(f"    - {issue}")
            if len(issues) > 10:
                print(f"    ... and {len(issues) - 10} more")
        else:
            print(f"  ✓ All labels valid!")

if __name__ == "__main__":
    fix_label_classes("datasets/my_drones/my_drones_final")
    verify_labels("datasets/my_drones/my_drones_final")