# Garbage Classification System

**Official topic:** Computer vision pipeline using OpenCV for image preprocessing, feature extraction, deep-learning-based classification/detection, and model evaluation.

## Scope

This student project contains:
1. OpenCV image preprocessing (resize, denoise, BGR-to-RGB conversion, normalization).
2. Traditional feature extraction (HSV color histograms, Canny edge density, GLCM texture).
3. CNN image classification for six TrashNet classes.
4. YOLO object-detection inference, with support for custom TACO-trained weights.
5. Classification metrics and confusion-matrix visualization.
6. A simple Streamlit demonstration application.

## Important distinction

TrashNet is a classification dataset and does not provide object bounding boxes. YOLO garbage/litter detection needs detection annotations such as TACO and a model trained/fine-tuned on those annotations. The app can run a generic `yolov8n.pt` detector for a software demonstration, but that is **not** a garbage-specific detector. Do not report generic COCO detections as trained garbage detection results.

## Datasets

- TrashNet: https://github.com/garythung/trashnet
- TACO: https://github.com/pedropro/TACO

Download datasets from their official repositories and place the classification images into:
`data/classification/<class_name>/<image_file>`

Expected class folder names: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.

## Environment

Python 3.12.10 is intended. On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If TensorFlow installation on Windows is problematic, use the supplied Colab notebook for training. Local inference requires a compatible TensorFlow installation; alternatively run the app in Colab or use a supported Python environment.

## Train the CNN

Run in Google Colab or a suitable TensorFlow environment:

```bash
python train_classifier.py --data data/classification --output models --epochs 15
```

The model and training plots are saved under `models/`.

## Evaluate

Use an independent test folder with the same class subfolder structure:

```bash
python evaluate_classifier.py --data data/classification_test --model models/garbage_cnn.keras
```

Outputs are saved in `outputs/`. For a proper evaluation, do not evaluate on the same images used for training. The training script currently creates a train/validation split; use a separate test folder for final metrics.

## Run the app

```bash
streamlit run app.py
```

For classification, put the trained model at `models/garbage_cnn.keras`.
For garbage-specific detection, put TACO-trained YOLO weights at `models/best.pt`.

## Feature extraction example

```python
import cv2
from src.feature_extraction import extract_features, feature_names

image = cv2.imread("sample.jpg")
features = extract_features(image)
print(len(features), len(feature_names()))
```

These handcrafted features are computed for analysis and demonstrate feature extraction. The CNN learns its own features from pixels; handcrafted features are not silently mixed into CNN inputs.

## Metrics

Classification: accuracy, weighted precision, weighted recall, weighted F1-score, classification report, confusion matrix.

Detection: IoU and mAP should be reported from a validation run on annotated detection data and a TACO-trained model. Do not invent detection metrics.

## Limitations

- Performance depends on dataset quality, class balance, image background, and training duration.
- A CNN trained on TrashNet may not generalize to every real-world waste image.
- Generic YOLO weights detect COCO classes and are not a substitute for a TACO-trained litter detector.
- TACO category mapping to six broad classes requires a documented mapping and review of category names.

## Optional genuine TACO YOLO training

After downloading TACO images and its COCO annotation JSON, convert the annotations:

```bash
python prepare_taco_yolo.py --images data/detection --annotations data/detection/annotations.json --output data/taco_yolo
```

Then train in Google Colab:

```bash
python train_yolo.py --data data/taco_yolo/data.yaml --epochs 30 --batch 8
```

The script uses TACO's original categories (not the six TrashNet classes). Review the generated class list and data split. The best weights are copied to `models/best.pt` when training completes. Validate on the held-out TACO validation set and report the actual mAP values from Ultralytics results.


## Model input consistency
The CNN includes a Keras `Rescaling(1./255)` layer. The local prediction wrapper therefore converts the OpenCV-preprocessed image back to the 0–255 pixel range before calling the model, matching the training input convention.

## Project status
This ZIP is a source-code starter package. It does not contain downloaded datasets or trained weights. Run the Colab notebook and detection training workflow before presenting measured results.
