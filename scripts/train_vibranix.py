"""
train_vibranix.py
=================
Train Vibranix Acoustic Radar classifier using MobileNetV2 transfer learning.

Classes: [Pest, Beneficial, Noise]
Inputs:  mel-spectrogram tensors saved by scripts/process_audio.py
Export:  models/vibranix_engine.tflite (Float16 quantization)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input


# Help TLS validation for Keras weight downloads (common macOS cert issue).
try:
    import certifi  # type: ignore

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"

MODELS_DIR = PROJECT_ROOT / "models"
EXPORT_PATH = MODELS_DIR / "vibranix_engine.tflite"
LABELS_PATH = MODELS_DIR / "class_names.json"

CLASS_NAMES = ["Pest", "Beneficial", "Noise"]
NUM_CLASSES = len(CLASS_NAMES)

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3


def read_manifest(path: Path) -> Tuple[List[str], List[int]]:
    paths: List[str] = []
    labels: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.append(row["path"])
            labels.append(int(row["label_idx"]))
    return paths, labels


def make_tf_dataset(paths: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> tf.data.Dataset:
    paths = paths.astype("str")
    labels = labels.astype("int32")

    def load_npy(path_tensor: tf.Tensor) -> tf.Tensor:
        def _load(p) -> np.ndarray:
            if isinstance(p, (bytes, bytearray)):
                p_str = p.decode("utf-8")
            else:
                p_val = p.numpy() if hasattr(p, "numpy") else p
                if isinstance(p_val, (bytes, bytearray)):
                    p_str = p_val.decode("utf-8")
                else:
                    p_str = str(p_val)
            arr = np.load(p_str).astype(np.float32)
            if arr.shape != (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
                raise ValueError(f"Bad shape {arr.shape} for {p_str}")
            return arr

        out = tf.py_function(_load, inp=[path_tensor], Tout=tf.float32)
        out.set_shape((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        return out

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (load_npy(p), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model() -> Tuple[tf.keras.Model, tf.keras.Model]:
    raw_inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=tf.float32, name="raw_spectrogram")
    x = tf.keras.layers.Lambda(preprocess_input, name="mobilenet_preprocess")(raw_inputs)

    base = MobileNetV2(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs=raw_inputs, outputs=outputs, name="vibranix_mobilenetv2")
    return model, base


def compute_class_weights(labels: List[int]) -> dict:
    arr = np.array(labels, dtype=np.int32)
    counts = np.bincount(arr, minlength=NUM_CLASSES)
    total = float(np.sum(counts))
    weights = {}
    for i, c in enumerate(counts):
        weights[i] = 0.0 if c <= 0 else total / (NUM_CLASSES * float(c))
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Vibranix and export Float16 TFLite.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--fine_tune_epochs", type=int, default=4)
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5)
    parser.add_argument("--unfreeze_last_n_layers", type=int, default=50)
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing {MANIFEST_PATH}. Run scripts/process_audio.py first.")

    paths, labels = read_manifest(MANIFEST_PATH)
    if len(paths) < 10:
        print("[train_vibranix] Warning: very small dataset; training will be low quality.")

    labels_arr = np.array(labels, dtype=np.int32)
    counts = np.bincount(labels_arr, minlength=NUM_CLASSES)
    present = counts[counts > 0]
    stratify = labels if (len(present) > 1 and int(np.min(present)) >= 2) else None

    X_train, X_val, y_train, y_val = train_test_split(
        paths,
        labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=stratify,
    )

    ds_train = make_tf_dataset(np.array(X_train), np.array(y_train), batch_size=args.batch_size, shuffle=True, seed=args.seed)
    ds_val = make_tf_dataset(np.array(X_val), np.array(y_val), batch_size=args.batch_size, shuffle=False, seed=args.seed)

    model, base = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    class_weights = compute_class_weights(y_train)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, min_lr=1e-7),
    ]

    model.fit(ds_train, validation_data=ds_val, epochs=args.epochs, class_weight=class_weights, callbacks=callbacks, verbose=1)

    # Fine-tune last layers
    if args.fine_tune_epochs > 0 and args.unfreeze_last_n_layers > 0:
        base.trainable = True
        for layer in base.layers[:-args.unfreeze_last_n_layers]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(ds_train, validation_data=ds_val, epochs=args.fine_tune_epochs, callbacks=callbacks, verbose=1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text(json.dumps(CLASS_NAMES, indent=2), encoding="utf-8")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    EXPORT_PATH.write_bytes(tflite_model)

    print(f"[train_vibranix] Exported: {EXPORT_PATH}")
    print(f"[train_vibranix] Labels: {LABELS_PATH}")


if __name__ == "__main__":
    main()

