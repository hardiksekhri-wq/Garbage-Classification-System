import numpy as np
from src.feature_extraction import extract_features, feature_names

def test_feature_length():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    features = extract_features(image)
    assert len(features) == len(feature_names()) == 65
