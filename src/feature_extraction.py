"""Handcrafted visual features: color histogram, edges, and GLCM texture."""
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def extract_features(image_bgr):
    """Return a 1-D feature vector from a BGR image.

    Features are descriptive measurements for analysis/classical baselines.
    The CNN does not consume these features unless a separate fusion model is built.
    """
    if image_bgr is None:
        raise ValueError("Input image is empty.")
    resized = cv2.resize(image_bgr, (224, 224))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    hist_features = []
    for channel, bins, ranges in [(0, 16, [0, 180]), (1, 16, [0, 256]), (2, 16, [0, 256])]:
        hist = cv2.calcHist([hsv], [channel], None, [bins], ranges).flatten()
        hist = hist / (hist.sum() + 1e-8)
        hist_features.extend(hist.tolist())

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))

    # Quantize grayscale to 8 levels to keep GLCM computation lightweight.
    quantized = (gray // 32).astype(np.uint8)
    glcm = graycomatrix(
        quantized, distances=[1, 3], angles=[0, np.pi / 4],
        levels=8, symmetric=True, normed=True
    )
    texture = []
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        texture.extend(graycoprops(glcm, prop).flatten().tolist())

    return np.asarray(hist_features + [edge_density] + texture, dtype=np.float32)


def feature_names():
    names = []
    for channel in ("hue", "saturation", "value"):
        names.extend([f"{channel}_hist_{i}" for i in range(16)])
    names.append("edge_density")
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        for distance in (1, 3):
            for angle in ("0", "45"):
                names.append(f"glcm_{prop}_d{distance}_a{angle}")
    return names
