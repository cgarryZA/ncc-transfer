"""
CW2 Option A — Training entry point.
Fine-tune RoBERTa for fault classification on SMRT maintenance logs.

Usage:
    python option_a_main.py          # run locally
    sbatch jobscript.sh              # submit via SLURM on NCC

Saves all artefacts to output/ for the evaluation notebook.
"""
import os, json, random, warnings
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
import keras
import keras_hub
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

from option_a_data import (
    SEED, CLASS_MAP, LABEL_ORDER,
    load_and_prepare_data, build_augmented_training_set, build_datasets,
)
from option_a_model import CFG, build_model
from option_a_train import (
    sanity_check, train_stage1, train_stage2, train_lora,
)
from option_a_evaluate import compute_baselines

# Reproducibility.
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    print(f"TensorFlow {tf.__version__}, Keras {keras.__version__}")
    if tf.config.list_physical_devices("GPU"):
        keras.config.set_dtype_policy("mixed_float16")
        print("Mixed precision enabled (float16)")
    print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

    # ── Data ────────────────────────────────────────────────────
    (X_train_raw, y_train, X_val_raw, y_val,
     X_test_raw, y_test, label_names) = load_and_prepare_data()
    CFG["LABEL_NAMES"] = label_names

    # Class weights for imbalanced categories.
    class_weights_arr = compute_class_weight(
        "balanced", classes=np.arange(CFG["NUM_CLASSES"]), y=y_train
    )
    class_weight_dict = {i: float(w) for i, w in enumerate(class_weights_arr)}

    # Text augmentation (3x training set).
    X_train_aug, y_train_aug = build_augmented_training_set(
        X_train_raw, y_train, n_copies=2
    )

    # ── Baselines ───────────────────────────────────────────────
    baselines = compute_baselines(X_train_raw, y_train, X_test_raw, y_test)

    # ── Tokenisation and datasets ───────────────────────────────
    preprocessor = keras_hub.models.RobertaPreprocessor.from_preset(
        CFG["MODEL_PRESET"], sequence_length=CFG["MAX_SEQ_LEN"]
    )
    train_ds, val_ds, test_ds = build_datasets(
        X_train_aug, y_train_aug, X_val_raw, y_val,
        X_test_raw, y_test, preprocessor, CFG["BATCH_SIZE"]
    )

    # ── Model ───────────────────────────────────────────────────
    backbone = keras_hub.models.RobertaBackbone.from_preset(
        CFG["MODEL_PRESET"]
    )
    model = build_model(backbone, CFG["NUM_CLASSES"])
    print(f"Model: {model.count_params():,} params")

    # ── Sanity check ────────────────────────────────────────────
    sanity_check(model, train_ds)

    # Rebuild with fresh weights after sanity check.
    backbone = keras_hub.models.RobertaBackbone.from_preset(
        CFG["MODEL_PRESET"]
    )
    model = build_model(backbone, CFG["NUM_CLASSES"])

    # ── Stage 1 ─────────────────────────────────────────────────
    history_s1, s1_val_acc = train_stage1(
        model, backbone, train_ds, val_ds, class_weight_dict
    )

    # ── Stage 2 ─────────────────────────────────────────────────
    history_s2, s2_val_acc = train_stage2(
        model, backbone, train_ds, val_ds, class_weight_dict
    )

    # ── Test evaluation ─────────────────────────────────────────
    model.load_weights(
        str(OUTPUT_DIR / "best_roberta_faultclassifier.keras")
    )
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    test_acc = float(np.mean(y_pred == y_test))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Macro F1:      {macro_f1:.4f}")

    # ── LoRA ────────────────────────────────────────────────────
    model_lora, history_lora, lora_trainable, lora_total = train_lora(
        train_ds, val_ds, class_weight_dict,
        str(OUTPUT_DIR / "best_roberta_stage1.keras"),
    )
    model_lora.load_weights(str(OUTPUT_DIR / "best_roberta_lora.keras"))
    y_pred_lora_probs = model_lora.predict(test_ds, verbose=0)
    y_pred_lora = np.argmax(y_pred_lora_probs, axis=1)
    lora_acc = float(np.mean(y_pred_lora == y_test))
    lora_f1 = float(f1_score(y_test, y_pred_lora, average="macro"))
    print(f"\nLoRA test accuracy: {lora_acc:.4f}")
    print(f"LoRA macro F1:      {lora_f1:.4f}")

    # ── Save predictions ────────────────────────────────────────
    np.savez(
        OUTPUT_DIR / "test_predictions.npz",
        y_test=y_test, y_pred=y_pred, y_pred_probs=y_pred_probs,
        y_pred_lora=y_pred_lora, y_pred_lora_probs=y_pred_lora_probs,
        label_names=label_names, X_test_raw=X_test_raw,
    )

    # ── Save config ─────────────────────────────────────────────
    save_cfg = {k: v for k, v in CFG.items()}
    save_cfg["CLASS_MAP"] = CLASS_MAP
    save_cfg["s1_val_acc"] = float(s1_val_acc)
    save_cfg["s2_val_acc"] = float(s2_val_acc)
    save_cfg["test_acc"] = test_acc
    save_cfg["macro_f1"] = macro_f1
    save_cfg["lora_acc"] = lora_acc
    save_cfg["lora_f1"] = lora_f1
    save_cfg["lora_trainable"] = int(lora_trainable)
    save_cfg["lora_total"] = int(lora_total)
    with open(OUTPUT_DIR / "config.json", "w") as f:
        json.dump(save_cfg, f, indent=2)

    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE -- Summary")
    print("=" * 60)
    print(f"  Majority baseline:   {baselines['majority_acc']:.4f}")
    print(f"  TF-IDF + LogReg:     {baselines['tfidf_acc']:.4f}")
    print(f"  Stage 1 val acc:     {s1_val_acc:.4f}")
    print(f"  Stage 2 val acc:     {s2_val_acc:.4f}")
    print(f"  Stage 2 test acc:    {test_acc:.4f} (macro F1: {macro_f1:.4f})")
    print(f"  LoRA test acc:       {lora_acc:.4f} (macro F1: {lora_f1:.4f})")
    print(f"\nArtefacts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
