"""
option_a_evaluate.py — Baseline computation and evaluation utilities.
CW2 Option A: Fine-tuning RoBERTa for fault classification.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from option_a_data import SEED

OUTPUT_DIR = Path("output")


def compute_baselines(X_train_raw, y_train, X_test_raw, y_test):
    """Compute majority-class and TF-IDF + LogReg baselines.

    Returns
    -------
    dict with keys 'majority_acc' and 'tfidf_acc'.
    """
    majority_class = np.bincount(y_train).argmax()
    majority_acc = float(np.mean(y_test == majority_class))
    print(f"Baseline 0 (majority class): {majority_acc:.4f}")

    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train_raw)
    X_test_tfidf = tfidf.transform(X_test_raw)
    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(X_train_tfidf, y_train)
    lr_acc = float(accuracy_score(y_test, lr.predict(X_test_tfidf)))
    print(f"Baseline 1 (TF-IDF + LogReg): {lr_acc:.4f}")

    baselines = {"majority_acc": majority_acc, "tfidf_acc": lr_acc}
    with open(OUTPUT_DIR / "baselines.json", "w") as f:
        json.dump(baselines, f, indent=2)
    return baselines


def evaluate_predictions(y_test, y_pred, y_pred_probs, label_names):
    """Compute and print test set metrics.

    Returns
    -------
    dict with 'test_acc', 'macro_f1', 'weighted_f1'.
    """
    test_acc = float(np.mean(y_pred == y_test))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))

    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Macro F1:      {macro_f1:.4f}")
    print(f"Weighted F1:   {weighted_f1:.4f}")
    return {"test_acc": test_acc, "macro_f1": macro_f1,
            "weighted_f1": weighted_f1}
