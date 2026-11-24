
"""Garbage Classification System - Streamlit Application."""

from pathlib import Path
import tempfile

import cv2
import streamlit as st
import json
from PIL import Image

from src.classification import load_classifier, predict_image
from src.detection import detect_image


# ==========================================
# PATH CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent

MODEL_PATH = ROOT / "models" / "mobilenet_garbage.keras"
YOLO_MODEL_PATH = ROOT / "models" / "best.pt"


# ==========================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Garbage Classification System",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ Garbage Classification System")

st.caption(
    "OpenCV preprocessing • Visual feature extraction • "
    "Deep-learning classification and object detection"
)


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

page = st.sidebar.radio(
    "Navigate",
    [
        "Project Overview",
        "Garbage Classification",
        "Object Detection",
        "Evaluation"
    ]
)


# ==========================================
# PAGE 1: PROJECT OVERVIEW
# ==========================================

if page == "Project Overview":

    st.subheader("Project Overview")

    st.write(
        "This project processes waste images using OpenCV for image "
        "preprocessing and visual feature extraction. It uses a "
        "MobileNetV2-based deep learning model for garbage classification "
        "and a TACO-trained YOLOv8n model for litter object detection."
    )

    st.markdown("""
    **Classification classes:** cardboard, glass, metal, paper, plastic, trash.

    **Classification model:** MobileNetV2 (Transfer Learning).

    **Object detection model:** YOLOv8n trained on the TACO dataset (60 categories).

    **Pipeline:** Input image → OpenCV preprocessing → MobileNetV2 classification
    / YOLO object detection → Evaluation.

    **Classification validation accuracy:** 81.98% (505 validation images).

    **Classification model:** `models/mobilenet_garbage.keras`

    **Detection model:** `models/best.pt`
    """)

    st.info(
        "The application includes a MobileNetV2 classification model "
        "and a TACO-trained YOLOv8n detection model."
    )


# ==========================================
# PAGE 2: GARBAGE CLASSIFICATION
# ==========================================

elif page == "Garbage Classification":

    st.subheader("Garbage Classification")

    uploaded = st.file_uploader(
        "Upload one garbage image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            width=380
        )

        if st.button("Classify Image"):

            try:

                model = load_classifier(MODEL_PATH)

                result = predict_image(model, image)

                st.success(
                    f"Prediction: {result['label'].title()}"
                )

                st.metric(
                    "Model Confidence",
                    f"{result['confidence'] * 100:.2f}%"
                )

                st.subheader("Class Probabilities")

                st.bar_chart(result["probabilities"])

            except Exception as exc:

                st.error(f"Classification error: {exc}")


# ==========================================
# PAGE 3: OBJECT DETECTION
# ==========================================

elif page == "Object Detection":

    st.subheader("Garbage Object Detection")

    st.write(
        "Upload an image to detect litter objects using "
        "the TACO-trained YOLOv8n model."
    )

    st.caption(
        "Using TACO-trained YOLOv8n model (60 litter categories)"
    )

    uploaded = st.file_uploader(
        "Upload an image for detection",
        type=["jpg", "jpeg", "png"],
        key="det"
    )

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05
    )

    if uploaded and st.button("Detect Objects"):

        if not YOLO_MODEL_PATH.exists():

            st.error(
                f"YOLO model not found: {YOLO_MODEL_PATH}"
            )

        else:

            suffix = Path(uploaded.name).suffix or ".jpg"

            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False
            ) as tmp:

                tmp.write(uploaded.getvalue())

                tmp_path = tmp.name

            try:

                with st.spinner("Detecting garbage objects..."):

                    annotated, detections = detect_image(
                        tmp_path,
                        weights=str(YOLO_MODEL_PATH),
                        confidence=confidence
                    )

                annotated_rgb = cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    annotated_rgb,
                    caption="YOLO TACO Detection Results",
                    use_container_width=True
                )

                if detections:

                    st.success(
                        f"{len(detections)} object(s) detected!"
                    )

                    st.subheader("Detected Objects")

                    for detection in detections:

                        st.write(
                            f"**{detection['label']}** — "
                            f"{detection['confidence'] * 100:.2f}% confidence"
                        )

                else:

                    st.info(
                        "No objects detected. Try another image "
                        "or adjust the confidence threshold."
                    )

            except Exception as exc:

                st.error(f"Detection error: {exc}")

            finally:

                Path(tmp_path).unlink(missing_ok=True)


# ==========================================
# PAGE 4: MODEL EVALUATION
# ==========================================

elif page == "Evaluation":

    st.subheader("Model Evaluation")

    metrics_path = ROOT / "outputs" / "classification_metrics.json"

    if metrics_path.exists():

        metrics = json.loads(
            metrics_path.read_text()
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Accuracy",
            f"{metrics['accuracy']:.3f}"
        )

        c2.metric(
            "Precision",
            f"{metrics['weighted_precision']:.3f}"
        )

        c3.metric(
            "Recall",
            f"{metrics['weighted_recall']:.3f}"
        )

        c4.metric(
            "F1-score",
            f"{metrics['weighted_f1']:.3f}"
        )

        cm_path = ROOT / "outputs" / "confusion_matrix.png"

        if cm_path.exists():

            st.image(
                str(cm_path),
                caption="Classification Confusion Matrix"
            )

    else:

        st.info(
            "Evaluation results are not available. "
            "Please check the outputs folder."
        )
