# Google Colab Guide

1. Upload `notebooks/garbage_pipeline_colab.ipynb` to https://colab.research.google.com/.
2. In Runtime → Change runtime type, select a GPU if one is available.
3. Upload the TrashNet dataset ZIP or mount Google Drive and set `DATA_DIR` in the notebook.
4. Run cells from top to bottom.
5. The notebook trains a compact CNN and saves `garbage_cnn.keras`.
6. Download the model and place it in the local `models/` folder.
7. Run `streamlit run app.py` locally.

The notebook also contains an optional YOLO/TACO section. TACO data conversion and fine-tuning may take longer and require a valid TACO COCO annotation JSON plus image files. It is not automatically run unless you set the paths and enable the optional section.

Colab runtimes can disconnect. Save checkpoints and outputs to Google Drive during longer runs.
