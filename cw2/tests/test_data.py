"""Tests for the data pipeline (option_a_data.py)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from option_a_data import (
    CLASS_MAP, LABEL_ORDER, SEED,
    load_and_prepare_data, augment_text, build_augmented_training_set,
)


def test_class_map_values_are_valid():
    """Every mapped value must be one of the six LABEL_ORDER categories."""
    for component, category in CLASS_MAP.items():
        assert category in LABEL_ORDER, (
            f"CLASS_MAP['{component}'] = '{category}' not in LABEL_ORDER"
        )


def test_label_order_has_six_classes():
    """We expect exactly 6 functional categories."""
    assert len(LABEL_ORDER) == 6
    assert "Other" in LABEL_ORDER


def test_load_and_prepare_data_shapes():
    """Data splits are non-empty and labels are in the correct range."""
    (X_train, y_train, X_val, y_val,
     X_test, y_test, label_names) = load_and_prepare_data()

    assert len(X_train) > 0 and len(X_val) > 0 and len(X_test) > 0
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)
    assert set(label_names) == set(LABEL_ORDER)

    # Labels should be integers in [0, 5].
    for y in (y_train, y_val, y_test):
        assert y.min() >= 0
        assert y.max() < len(label_names)


def test_load_and_prepare_data_no_nan():
    """No NaN values in text or labels."""
    (X_train, y_train, X_val, y_val,
     X_test, y_test, _) = load_and_prepare_data()

    for texts in (X_train, X_val, X_test):
        assert all(isinstance(t, str) and len(t) > 0 for t in texts)
    for labels in (y_train, y_val, y_test):
        assert not np.any(np.isnan(labels))


def test_augment_text_returns_string():
    """Augmented text must still be a non-empty string."""
    text = "Replace the saloon light bulb unit"
    result = augment_text(text)
    assert isinstance(result, str) and len(result) > 0


def test_augment_text_short_input():
    """Short texts (< 4 words) are returned unchanged."""
    text = "fuse blown"
    assert augment_text(text) == text


def test_build_augmented_training_set():
    """Augmented set should be (n_copies + 1) times the original."""
    X = np.array(["fault A description", "fault B description", "fault C check"])
    y = np.array([0, 1, 2])
    X_aug, y_aug = build_augmented_training_set(X, y, n_copies=2)
    assert len(X_aug) == 3 * len(X)
    assert len(y_aug) == 3 * len(y)
    # Original samples preserved at positions 0, 3, 6.
    assert X_aug[0] == X[0]
    assert X_aug[3] == X[1]
