# Garbage Classification System — Project Report

## Abstract
This project demonstrates a computer-vision pipeline for waste-image analysis. OpenCV is used for image preprocessing, traditional descriptors are extracted for visual analysis, and a convolutional neural network is trained to classify images into six waste categories. A separate YOLO module provides object-detection inference and can use weights fine-tuned on a litter detection dataset such as TACO. Model performance is evaluated using standard classification metrics and, when annotated detection predictions are available, detection metrics.

## 1. Introduction
Explain the need for waste segregation and automated image-based waste identification.

## 2. Problem Statement
Manual waste sorting is time-consuming and inconsistent. The project explores image-based classification and object localization to support waste categorization.

## 3. Objectives
- Preprocess images using OpenCV.
- Extract color, edge, and texture features.
- Train a CNN classifier for six waste classes.
- Demonstrate YOLO object detection with suitable weights.
- Evaluate classification and detection performance using valid ground truth.

## 4. Dataset
Describe the exact TrashNet/TACO versions downloaded, image counts, class counts, train/validation/test split, and any category mapping. Fill in only after inspecting the downloaded data.

## 5. Methodology
Describe preprocessing, handcrafted features, CNN architecture, YOLO pipeline, and evaluation.

## 6. Implementation
Add screenshots of actual notebook execution and Streamlit output.

## 7. Results
Insert only metrics produced by your actual test run. Include confusion matrix and training curves. If TACO fine-tuning was not completed, state that honestly and label generic YOLO output as a demonstration only.

## 8. Limitations
Discuss dataset size, class imbalance, background variation, compute limits, and domain generalization.

## 9. Conclusion and Future Scope
Summarize the implemented pipeline and possible future work such as larger datasets, more waste categories, and deployment on edge devices.

## References
- TrashNet repository: https://github.com/garythung/trashnet
- TACO repository: https://github.com/pedropro/TACO
- OpenCV documentation: https://docs.opencv.org/
- TensorFlow documentation: https://www.tensorflow.org/
- Ultralytics documentation: https://docs.ultralytics.com/
