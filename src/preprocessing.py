"""OpenCV preprocessing utilities for the Garbage Classification System."""
from pathlib import Path
import cv2
import numpy as np


def load_image(path):
    """Load an image with OpenCV and return BGR array; raise a clear error if invalid."""
    path = str(path)
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def preprocess_bgr(image, size=(224, 224), denoise=True):
    """Resize, optionally denoise, convert BGR->RGB, and scale pixels to [0, 1]."""
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Expected a valid OpenCV image array.")
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    if denoise:
        resized = cv2.GaussianBlur(resized, (3, 3), 0)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    return normalized


def preprocess_path(path, size=(224, 224), denoise=True):
    return preprocess_bgr(load_image(path), size=size, denoise=denoise)


def save_preprocessing_preview(path, output_path, size=(224, 224)):
    """Save side-by-side original and processed preview."""
    image = load_image(path)
    processed = preprocess_bgr(image, size=size)
    processed_bgr = cv2.cvtColor((processed * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    original = cv2.resize(image, size)
    preview = cv2.hconcat([original, processed_bgr])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), preview):
        raise IOError(f"Could not save preview to {output_path}")
    return output_path
