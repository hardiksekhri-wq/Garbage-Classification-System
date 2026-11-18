# Feature extraction notes

`extract_features()` returns:
- 48 HSV histogram values (16 bins per H, S, V channel).
- 1 Canny edge-density value.
- 16 GLCM texture statistics (4 properties × 2 distances × 2 angles).

Total: 65 features.

HSV histograms summarize color distribution. Canny edge density measures the fraction of edge pixels. GLCM properties summarize grayscale texture. These features are useful for visual analysis or a separate classical ML baseline; the CNN independently learns features from image pixels.
