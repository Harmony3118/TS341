from ultralytics import YOLO
from pathlib import Path
import shutil


# finetune_yolo.py - CPU optimized version
def finetune_on_custom_data(
    pretrained_weights="best_yolo8s.pt",
    data_yaml="datasets/my_drones/my_drones_final/data.yaml",
    epochs=50,  # Reduced from 100
    batch_size=8,  # Reduced from 16
    imgsz=640,
    project_name="drone_finetune",
):
    """
    Fine-tune existing drone detection model on your custom data

    Args:
        pretrained_weights: Path to your existing drone weights
        data_yaml: Path to your dataset configuration
        epochs: Number of training epochs
        batch_size: Batch size (reduce if GPU memory issues)
        imgsz: Image size for training
        project_name: Name for this training run
    """

    print("=" * 70)
    print("FINE-TUNING YOLO MODEL ON CUSTOM DATA")
    print("=" * 70)
    print(f"Starting weights: {pretrained_weights}")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}")
    print("=" * 70)

    # Load your pre-trained drone detection model
    model = YOLO(pretrained_weights)

    # Fine-tune on your custom dataset
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        name=project_name,
        patience=15,  # Reduced patience
        save=True,
        save_period=10,
        lr0=0.001,
        lrf=0.01,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        device="cpu",  # CPU training
        workers=4,  # Reduced workers
        cache=True,
        val=True,
        plots=True,
        project="runs/detect",
        exist_ok=True,
        pretrained=False,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)

    best_model_path = Path(f"runs/detect/{project_name}/weights/best.pt")
    last_model_path = Path(f"runs/detect/{project_name}/weights/last.pt")

    print(f"Best model: {best_model_path}")
    print(f"Last model: {last_model_path}")

    # Copy best model to easy-to-find location
    output_model = Path("my_custom_drone_model.pt")
    shutil.copy(best_model_path, output_model)
    print(f"\n✓ Best model copied to: {output_model}")

    return str(best_model_path)


def validate_model(model_path, data_yaml):
    """Validate the fine-tuned model"""
    print("\n" + "=" * 70)
    print("VALIDATING MODEL")
    print("=" * 70)

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)

    print(f"\nValidation Metrics:")
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")
    print("=" * 70)

    return metrics


def test_on_video(model_path, video_path):
    """Quick test on a video"""
    print("\n" + "=" * 70)
    print("TESTING ON VIDEO")
    print("=" * 70)

    model = YOLO(model_path)

    # Run inference on video
    results = model.predict(
        source=video_path,
        save=True,  # Save annotated video
        conf=0.25,  # Confidence threshold
        project="runs/detect",
        name="test_video",
    )

    print(f"✓ Results saved to: runs/detect/test_video")
    print("=" * 70)


if __name__ == "__main__":
    # Fine-tune the model
    best_model = finetune_on_custom_data()

    # Validate the fine-tuned model
    validate_model(best_model, "datasets/my_drones/my_drones_final/data.yaml")

    # Optional: Test on a video
    # test_on_video(best_model, "../../videos/cam0_10_31_11_19_51.mkv")
