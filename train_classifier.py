"""Train a compact CNN on a folder-based TrashNet dataset.

Expected layout:
data/classification/
  cardboard/*.jpg
  glass/*.jpg
  metal/*.jpg
  paper/*.jpg
  plastic/*.jpg
  trash/*.jpg

Run in Google Colab (GPU recommended), not on the CPU-only laptop.
"""
from pathlib import Path
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def build_model(num_classes):
    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.10)(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main(data_dir, output_dir, epochs):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=SEED,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="int"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=SEED,
        image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode="int"
    )
    class_names = train_ds.class_names
    if len(class_names) < 2:
        raise ValueError("Expected class subfolders: cardboard, glass, metal, paper, plastic, trash.")
    (output_dir / "class_names.txt").write_text("\\n".join(class_names), encoding="utf-8")
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(500).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    model = build_model(len(class_names))
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(output_dir / "garbage_cnn.keras", save_best_only=True)
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    model.save(output_dir / "garbage_cnn.keras")

    import matplotlib.pyplot as plt
    for metric in ("accuracy", "loss"):
        plt.figure()
        plt.plot(history.history[metric], label=f"train_{metric}")
        plt.plot(history.history[f"val_{metric}"], label=f"val_{metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric.title())
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"training_{metric}.png", dpi=150)
        plt.close()
    print("Saved model and training plots to", output_dir)
    print("Class order:", class_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/classification")
    parser.add_argument("--output", default="models")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    main(args.data, args.output, args.epochs)
