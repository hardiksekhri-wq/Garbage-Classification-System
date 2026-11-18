"""Fine-tune a small YOLO detector on a converted TACO YOLO dataset.

Run in Google Colab:
python train_yolo.py --data data/taco_yolo/data.yaml --epochs 30
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main(data_yaml, epochs, image_size, batch, model_name):
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"YOLO data YAML not found: {data_yaml}. Run prepare_taco_yolo.py first.")
    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        project="runs/garbage_detection",
        name="taco_yolov8n",
        patience=8,
        workers=2,
        exist_ok=True
    )
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        Path("models").mkdir(exist_ok=True)
        import shutil
        shutil.copy2(best, "models/best.pt")
        print("Best weights copied to models/best.pt")
    print("Training complete. Review Ultralytics validation metrics in the run directory.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/taco_yolo/data.yaml")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--model", default="yolov8n.pt")
    a = p.parse_args()
    main(a.data, a.epochs, a.imgsz, a.batch, a.model)
