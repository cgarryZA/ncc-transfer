"""Tests for the model architecture (option_a_model.py)."""
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
def backbone_and_model():
    """Build a model once for all tests in this module."""
    backbone = keras_hub.models.RobertaBackbone.from_preset(
        CFG["MODEL_PRESET"]
    )
    model = build_model(backbone, CFG["NUM_CLASSES"])
    return backbone, model


def test_model_output_shape(backbone_and_model):
    """Output should be (batch_size, NUM_CLASSES) with softmax probabilities."""
    _, model = backbone_and_model
    # The model expects tokenised input; check output layer shape.
    output_layer = model.layers[-1]
    assert output_layer.output.shape[-1] == CFG["NUM_CLASSES"]


def test_model_has_softmax_activation(backbone_and_model):
    """Final layer must use softmax for probability output."""
    _, model = backbone_and_model
    last_dense = model.layers[-1]
    assert hasattr(last_dense, "activation")
    assert last_dense.activation.__name__ == "softmax"


def test_model_name(backbone_and_model):
    """Model should have a descriptive name."""
    _, model = backbone_and_model
    assert "RoBERTa" in model.name


def test_model_compiles(backbone_and_model):
    """Model should compile without errors."""
    _, model = backbone_and_model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def test_cfg_has_required_keys():
    """CFG dict must contain all expected hyperparameters."""
    required = [
        "MAX_SEQ_LEN", "BATCH_SIZE",
        "EPOCHS_STAGE1", "EPOCHS_STAGE2",
        "LR_STAGE1", "LR_STAGE2",
        "LORA_RANK", "LR_LORA", "EPOCHS_LORA",
        "MODEL_PRESET", "NUM_CLASSES",
    ]
    for key in required:
        assert key in CFG, f"Missing CFG key: {key}"
