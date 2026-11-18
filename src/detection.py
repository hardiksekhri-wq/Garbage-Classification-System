"""YOLO inference wrapper. Supply custom TACO-trained weights for litter-specific detection."""
from pathlib import Path


def detect_image(image_path, weights="models/best.pt", confidence=0.25):
    """Run YOLO detection and return annotated image plus structured detections.

    `weights` should point to weights trained/fine-tuned on a garbage/litter dataset.
    The generic yolov8n.pt fallback detects COCO objects, not all litter categories.
    """
    from ultralytics import YOLO
    weights_path = Path(weights)
    if not weights_path.exists() and str(weights) != "yolov8n.pt":
        raise FileNotFoundError(
            f"Detection weights not found: {weights_path}. Train YOLO on TACO or set weights='yolov8n.pt' "
            "for a generic demonstration (not a garbage-specific detector)."
        )
    model = YOLO(str(weights) if weights_path.exists() else "yolov8n.pt")
    results = model.predict(source=str(image_path), conf=confidence, verbose=False)
    result = results[0]
    annotated_bgr = result.plot()
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            detections.append({
                "class_id": cls_id,
                "label": model.names.get(cls_id, str(cls_id)),
                "confidence": float(box.conf.item()),
                "xyxy": [float(v) for v in box.xyxy[0].tolist()]
            })
    return annotated_bgr, detections
