"""Load and run the trained Keras garbage image classifier."""
from pathlib import Path
import numpy as np
from PIL import Image
from .preprocessing import preprocess_bgr
import cv2

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def load_classifier(model_path):
    import tensorflow as tf
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Classifier model not found: {model_path}. Train it using notebooks/garbage_pipeline_colab.ipynb."
        )
    return tf.keras.models.load_model(model_path)


def predict_image(model, pil_image, class_names=None):
    class_names = class_names or CLASS_NAMES
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # The trained CNN includes a Rescaling(1./255) layer, so pass pixel values in [0, 255].
    x = (preprocess_bgr(bgr) * 255.0)[None, ...]
    probabilities = model.predict(x, verbose=0)[0]
    index = int(np.argmax(probabilities))
    return {
        "label": class_names[index],
        "confidence": float(probabilities[index]),
        "probabilities": {name: float(prob) for name, prob in zip(class_names, probabilities)}
    }
