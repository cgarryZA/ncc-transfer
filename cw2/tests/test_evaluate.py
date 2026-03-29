"""Tests for evaluation utilities (option_a_evaluate.py)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from option_a_evaluate import evaluate_predictions


def test_evaluate_predictions_perfect():
    """Perfect predictions should give accuracy and F1 of 1.0."""
    y = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3])
    probs = np.eye(6)[y]  # one-hot
    result = evaluate_predictions(y, y, probs, ["A", "B", "C", "D", "E", "F"])
    assert abs(result["test_acc"] - 1.0) < 1e-6
    assert abs(result["macro_f1"] - 1.0) < 1e-6


def test_evaluate_predictions_random():
    """Random predictions should give accuracy well below 1.0."""
    rng = np.random.default_rng(42)
    y_test = rng.integers(0, 6, size=200)
    y_pred = rng.integers(0, 6, size=200)
    probs = rng.random((200, 6))
    probs /= probs.sum(axis=1, keepdims=True)

    result = evaluate_predictions(
        y_test, y_pred, probs, ["A", "B", "C", "D", "E", "F"]
    )
    assert result["test_acc"] < 0.5  # random chance ~0.167
    assert 0 < result["macro_f1"] < 1


def test_evaluate_predictions_returns_dict():
    """Return value should be a dict with the three expected keys."""
    y = np.array([0, 1, 2])
    probs = np.eye(6)[:3]
    result = evaluate_predictions(y, y, probs, ["A", "B", "C", "D", "E", "F"])
    assert isinstance(result, dict)
    assert "test_acc" in result
    assert "macro_f1" in result
    assert "weighted_f1" in result
