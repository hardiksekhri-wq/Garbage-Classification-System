"""Simple Streamlit interface for classification and YOLO detection."""
from pathlib import Path
import tempfile
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.classification import load_classifier, predict_image, CLASS_NAMES
from src.detection import detect_image

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "mobilenet_garbage.keras"

st.set_page_config(page_title="Garbage Classification System", page_icon="♻️", layout="wide")
st.title("♻️ Garbage Classification System")
st.caption("OpenCV preprocessing • Visual feature extraction • Deep-learning classification and detection")

page = st.sidebar.radio("Navigate", ["Project Overview", "Garbage Classification", "Object Detection", "Evaluation"])

if page == "Project Overview":
    st.subheader("Project overview")

    st.write(
        "This project processes waste images using OpenCV for image "
        "preprocessing and visual feature extraction. It uses a "
        "MobileNetV2-based deep learning model for garbage classification "
        "and provides a YOLO-based object detection interface."
    )

    st.markdown("""
    **Classification classes:** cardboard, glass, metal, paper, plastic, trash.

    **Classification model:** MobileNetV2 (Transfer Learning).

    **Pipeline:** Input image → OpenCV preprocessing → MobileNetV2 classification
    / YOLO object detection → Evaluation.

    **Validation accuracy:** 81.98% (505 validation images).

    **Model path:** `models/mobilenet_garbage.keras`
    """)

    st.info(
        "The trained MobileNetV2 model is already included in the "
        "models folder. No additional training is required."
    )
elif page == "Garbage Classification":
    uploaded = st.file_uploader("Upload one garbage image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", width=380)
        if st.button("Classify image"):
            try:
                model = load_classifier(MODEL_PATH)
                result = predict_image(model, image)
                st.success(f"Prediction: {result['label'].title()}")
                st.metric("Model confidence", f"{result['confidence'] * 100:.2f}%")
                st.bar_chart(result["probabilities"])
            except Exception as exc:
                st.error(str(exc))
elif page == "Object Detection":
    uploaded = st.file_uploader("Upload an image for detection", type=["jpg", "jpeg", "png"], key="det")
    weights = st.text_input("YOLO weights path", "models/best.pt")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    if uploaded and st.button("Detect objects"):
        suffix = Path(uploaded.name).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        try:
            annotated, detections = detect_image(tmp_path, weights=weights, confidence=confidence)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="YOLO detections", use_container_width=True)
            st.write(detections if detections else "No objects detected.")
            if weights == "yolov8n.pt":
                st.warning("This is a generic COCO detector, not a litter-specific detector. Use TACO-trained best.pt for garbage-specific detection.")
        except Exception as exc:
            st.error(str(exc))
        finally:
            Path(tmp_path).unlink(missing_ok=True)
elif page == "Evaluation":
    metrics_path = ROOT / "outputs" / "classification_metrics.json"
    if metrics_path.exists():
        import json
        metrics = json.loads(metrics_path.read_text())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        c2.metric("Precision", f"{metrics['weighted_precision']:.3f}")
        c3.metric("Recall", f"{metrics['weighted_recall']:.3f}")
        c4.metric("F1-score", f"{metrics['weighted_f1']:.3f}")
        cm_path = ROOT / "outputs" / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion matrix")
    else:
        st.info("Evaluation results are not available. Please check the outputs folder.")
