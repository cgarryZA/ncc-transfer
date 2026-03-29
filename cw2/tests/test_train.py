"""Smoke tests for training routines (option_a_train.py).

These tests verify that a single training step runs without error
on a tiny synthetic batch. They do NOT test convergence.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pytest
import tensorflow as tf
import keras
import keras_hub

from option_a_model import CFG, build_model


@pytest.fixture(scope="module")
def tiny_dataset():
    """Create a tiny tokenised dataset for smoke testing."""
    preprocessor = keras_hub.models.RobertaPreprocessor.from_preset(
        CFG["MODEL_PRESET"], sequence_length=CFG["MAX_SEQ_LEN"]
    )
    texts = np.array([
        "Replace air filter unit", "Door roller stuck",
        "Headlight replacement needed", "Traction motor overheating",
    ] * 8)  # 32 samples
    labels = np.array([2, 0, 3, 5] * 8)

    ds = tf.data.Dataset.from_tensor_slices((texts, labels))
    ds = ds.batch(8).map(
        lambda x, y: (preprocessor(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds


def test_single_training_step(tiny_dataset):
    """A single .fit() call on a small batch should run without error."""
    backbone = keras_hub.models.RobertaBackbone.from_preset(
        CFG["MODEL_PRESET"]
    )
    model = build_model(backbone, CFG["NUM_CLASSES"])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    hist = model.fit(tiny_dataset, epochs=1, verbose=0)
    assert "loss" in hist.history
    assert len(hist.history["loss"]) == 1


def test_loss_decreases_on_overfit(tiny_dataset):
    """Overfitting a small batch for a few epochs should reduce loss."""
    backbone = keras_hub.models.RobertaBackbone.from_preset(
        CFG["MODEL_PRESET"]
    )
    model = build_model(backbone, CFG["NUM_CLASSES"])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    hist = model.fit(tiny_dataset, epochs=10, verbose=0)
    losses = hist.history["loss"]
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )
