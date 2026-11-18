"""Evaluate a saved CNN on a held-out test folder.

Expected folder layout is the same class-subfolder layout as training.
For a strict independent test set, pass a separate folder to --data.
"""
from pathlib import Path
import argparse
import numpy as np
import tensorflow as tf
from src.evaluation import classification_metrics


def main(data_dir, model_path, output_dir):
    model = tf.keras.models.load_model(model_path)
    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, image_size=(224, 224), batch_size=32, shuffle=False
    )
    class_names = ds.class_names
    y_true = np.concatenate([labels.numpy() for _, labels in ds])
    probs = model.predict(ds, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    result = classification_metrics(y_true, y_pred, class_names, output_dir)
    print(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="models/garbage_cnn.keras")
    p.add_argument("--output", default="outputs")
    a = p.parse_args()
    main(a.data, a.model, a.output)
