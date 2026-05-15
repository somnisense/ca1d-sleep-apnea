#!/usr/bin/env python3
"""
This paper — Multi-seed reproducibility experiments
================================================

Trains 3 architectures (Original CNN baseline, SE-Attn CNN, Coord-Attn CNN)
across 5 random seeds (42, 123, 456, 789, 2026) on the 2953-sample sleep
apnea dataset (binary classification: Normal vs Apnea+Hypopnea).

Output: results/metrics_multiseed.csv
  - One row per (seed, model) combination
  - Columns: accuracy, precision, recall, specificity, f1, auc_roc, auc_pr,
             num_params, tn, fp, fn, tp, train_seconds

Usage:
    cd paper_C_ca1d_apnea/code
    python run_experiments.py                  # Run all 15 experiments
    python run_experiments.py --seeds 42       # Single seed
    python run_experiments.py --models coord   # Single model
    python run_experiments.py --smoke          # Smoke test (1 seed, all models)
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    Input,
    Layer,
    MaxPooling1D,
    Multiply,
    Reshape,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

# ============================================================
# Config
# ============================================================
SCRIPT_DIR = Path(__file__).parent.resolve()
PAPER_DIR = SCRIPT_DIR.parent
DATA_DIR = PAPER_DIR / "data" / "full"
RESULTS_DIR = PAPER_DIR / "code" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 2026]
MODEL_NAMES = ["original", "se", "coord"]
INPUT_SHAPE = (200, 3)
BATCH_SIZE = 32
EPOCHS = 100
EARLY_STOP_PATIENCE = 15
LR_REDUCE_PATIENCE = 8


# ============================================================
# Data loading
# ============================================================
def prepare_data(data_dir, data, labels, label_value):
    """Load all .txt files from data_dir and append to lists."""
    for file_name in os.listdir(data_dir):
        if not file_name.endswith(".txt"):
            continue
        file_path = os.path.join(data_dir, file_name)
        matrix = np.loadtxt(file_path, delimiter=":")
        assert matrix.shape == (200, 3), (
            f"File {file_name} has invalid shape: {matrix.shape}"
        )
        data.append(matrix)
        labels.append(label_value)


def load_dataset_binary(base_dir):
    """Load entire dataset and convert to binary (Normal vs Apnea+Hypopnea)."""
    data, labels = [], []
    counts = {}
    for label in [0, 1, 2]:
        dir_path = os.path.join(base_dir, str(label))
        if os.path.exists(dir_path):
            before = len(data)
            prepare_data(dir_path, data, labels, label)
            counts[label] = len(data) - before
    X = np.array(data)
    y_orig = np.array(labels)
    y_bin = np.where(y_orig > 0, 1, 0)
    return X, y_bin, counts


# ============================================================
# Attention blocks
# ============================================================
class SEBlock1D(Layer):
    """Squeeze-and-Excitation block adapted to 1D time-series."""

    def __init__(self, channels, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.reduced_channels = max(channels // reduction, 1)

    def build(self, input_shape):
        self.global_pool = GlobalAveragePooling1D()
        self.fc1 = Dense(
            self.reduced_channels, activation="relu",
            kernel_initializer="he_normal",
        )
        self.fc2 = Dense(
            self.channels, activation="sigmoid",
            kernel_initializer="he_normal",
        )
        self.reshape = Reshape((1, self.channels))
        super().build(input_shape)

    def call(self, inputs):
        s = self.global_pool(inputs)
        s = self.fc1(s)
        s = self.fc2(s)
        s = self.reshape(s)
        return Multiply()([inputs, s])


class CoordinateAttention1D(Layer):
    """1D adaptation of Coordinate Attention (Hou et al., CVPR 2021).

    Uses a global-local feature fusion design: GAP context tiled + concatenated
    with local features → bottleneck Conv1D + BN + ReLU → Conv1D + Sigmoid →
    T×C attention map → element-wise multiply with input.
    """

    def __init__(self, channels, reduction=16, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.reduced_channels = max(channels // reduction, 1)

    def build(self, input_shape):
        self.global_pool = GlobalAveragePooling1D(keepdims=True)
        self.conv_reduce = Conv1D(
            filters=self.reduced_channels, kernel_size=1, padding="same",
            use_bias=False, kernel_initializer="he_normal",
        )
        self.bn = BatchNormalization()
        self.conv_channel = Conv1D(
            filters=self.channels, kernel_size=1, padding="same",
            use_bias=False, kernel_initializer="he_normal",
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        timesteps = tf.shape(inputs)[1]
        g = self.global_pool(inputs)
        g_tiled = tf.repeat(g, repeats=timesteps, axis=1)
        h = tf.concat([inputs, g_tiled], axis=-1)
        u = self.conv_reduce(h)
        u = self.bn(u, training=training)
        u = tf.nn.relu(u)
        a = self.conv_channel(u)
        a = tf.nn.sigmoid(a)
        return inputs * a

    def compute_output_shape(self, input_shape):
        return input_shape


# ============================================================
# Model builders
# ============================================================
def build_original(input_shape=INPUT_SHAPE):
    """Baseline 1D CNN: Conv → MaxPool ×3 → Flatten → Dense ×3 → Sigmoid."""
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv1D(16, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Conv1D(32, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Conv1D(64, kernel_size=3, activation="relu"),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dropout(0.1),
            Dense(1, activation="sigmoid"),
        ],
        name="Original_CNN",
    )
    return model


def build_se(input_shape=INPUT_SHAPE):
    """SE-Attention CNN: Conv + BN + SE → MaxPool ×3 → GAP → Dense → Sigmoid."""
    inputs = Input(shape=input_shape)
    x = Conv1D(16, kernel_size=3, activation="relu", padding="same")(inputs)
    x = BatchNormalization()(x)
    x = SEBlock1D(channels=16, reduction=4)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(32, kernel_size=3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = SEBlock1D(channels=32, reduction=8)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(64, kernel_size=3, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = SEBlock1D(channels=64, reduction=16)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    return Model(inputs=inputs, outputs=outputs, name="SE_CNN")


def build_coord(input_shape=INPUT_SHAPE):
    """Coordinate-Attention CNN: Conv + BN + CA-1D → MaxPool ×3 → GAP → Dense → Sigmoid."""
    inputs = Input(shape=input_shape)
    x = Conv1D(
        16, kernel_size=3, activation="relu", padding="same",
        kernel_initializer="he_normal",
    )(inputs)
    x = BatchNormalization()(x)
    x = CoordinateAttention1D(channels=16, reduction=4)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(
        32, kernel_size=3, activation="relu", padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = BatchNormalization()(x)
    x = CoordinateAttention1D(channels=32, reduction=8)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(
        64, kernel_size=3, activation="relu", padding="same",
        kernel_initializer="he_normal",
    )(x)
    x = BatchNormalization()(x)
    x = CoordinateAttention1D(channels=64, reduction=16)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu", kernel_initializer="he_normal")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    return Model(inputs=inputs, outputs=outputs, name="Coord_CNN")


MODEL_BUILDERS = {
    "original": build_original,
    "se": build_se,
    "coord": build_coord,
}


# ============================================================
# Training + evaluation
# ============================================================
def set_global_seed(seed):
    """Fix all relevant RNGs for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Compile, train (with early stopping + LR schedule), then evaluate on test."""
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    cw = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1]), y=y_train,
    )
    class_weight = {0: cw[0], 1: cw[1]}

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=LR_REDUCE_PATIENCE, min_lr=1e-6, verbose=0,
        ),
    ]

    t0 = time.time()
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        class_weight=class_weight, callbacks=callbacks, verbose=0,
    )
    train_seconds = time.time() - t0

    # Evaluate at default threshold 0.5
    y_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_proba >= 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred).ravel()
    tn, fp, fn, tp = (int(x) for x in cm)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, y_proba),
        "auc_pr": average_precision_score(y_test, y_proba),
        "num_params": int(model.count_params()),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "train_seconds": round(train_seconds, 1),
    }
    return metrics, y_proba


# ============================================================
# Main runner
# ============================================================
def run_one(seed, model_name, X, y, csv_writer, csv_file, probas_dir):
    """Run a single (seed, model) experiment and append result to CSV."""
    print(f"\n{'=' * 70}")
    print(f"  Seed = {seed} | Model = {model_name}")
    print(f"{'=' * 70}")

    set_global_seed(seed)

    # Stratified 60/20/20 split using same seed
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp,
    )
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    builder = MODEL_BUILDERS[model_name]
    model = builder()
    print(f"  Params: {model.count_params():,}")

    metrics, y_proba = train_and_evaluate(
        model, X_train, y_train, X_val, y_val, X_test, y_test,
    )

    # Save per-sample probabilities for downstream Bootstrap CI / ROC / threshold plots
    proba_path = probas_dir / f"{model_name}_seed{seed}.npz"
    np.savez_compressed(
        proba_path, y_test=y_test, y_proba=y_proba,
    )

    row = {"seed": seed, "model": model_name, **metrics}
    csv_writer.writerow(row)
    csv_file.flush()

    print(
        f"  → Acc: {metrics['accuracy']*100:.2f}% | "
        f"Sens: {metrics['recall']*100:.2f}% | "
        f"Spec: {metrics['specificity']*100:.2f}% | "
        f"F1: {metrics['f1']*100:.2f}% | "
        f"AUC: {metrics['auc_roc']:.4f} | "
        f"time: {metrics['train_seconds']}s"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated seed list (default: all 5).",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated model list: original,se,coord (default: all 3).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test: 1 seed (42), all 3 models.",
    )
    parser.add_argument(
        "--output", type=str, default="metrics_multiseed.csv",
        help="Output CSV filename (in code/results/).",
    )
    args = parser.parse_args()

    # Resolve seeds + models
    if args.smoke:
        seeds = [42]
        models = MODEL_NAMES
    else:
        seeds = (
            [int(s) for s in args.seeds.split(",")]
            if args.seeds else SEEDS
        )
        models = (
            args.models.split(",") if args.models else MODEL_NAMES
        )
    for m in models:
        assert m in MODEL_BUILDERS, f"Unknown model: {m}"

    # Load dataset
    print(f"Loading dataset from {DATA_DIR}")
    t0 = time.time()
    X, y, counts = load_dataset_binary(str(DATA_DIR))
    print(f"  Loaded {len(X)} samples in {time.time()-t0:.1f}s")
    print(f"  Original counts: {counts}")
    print(
        f"  Binary: Normal={int(np.sum(y==0))} | Abnormal={int(np.sum(y==1))}"
    )
    print(f"  Data shape: {X.shape}")

    # Prepare CSV
    out_path = RESULTS_DIR / args.output
    probas_dir = RESULTS_DIR / "probas"
    probas_dir.mkdir(exist_ok=True)
    is_new = not out_path.exists()
    fieldnames = [
        "seed", "model", "accuracy", "precision", "recall", "specificity",
        "f1", "auc_roc", "auc_pr", "num_params", "tn", "fp", "fn", "tp",
        "train_seconds",
    ]
    csv_file = open(out_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if is_new:
        writer.writeheader()
        csv_file.flush()

    total = len(seeds) * len(models)
    done = 0
    t_start = time.time()
    for seed in seeds:
        for m in models:
            done += 1
            print(f"\n## Experiment {done}/{total}")
            try:
                run_one(seed, m, X, y, writer, csv_file, probas_dir)
            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                import traceback
                traceback.print_exc()

    csv_file.close()
    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  DONE — {done} experiments in {elapsed/60:.1f} min")
    print(f"  Results: {out_path}")
    print(f"  Per-sample probabilities: {probas_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
