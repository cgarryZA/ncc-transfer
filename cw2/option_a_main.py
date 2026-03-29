"""
CW2 Option A — Training Script
Fine-tune RoBERTa for fault classification on SMRT maintenance logs.
Run via SLURM (see jobscript.sh) or locally: python option_a_main.py

Saves all artifacts to output/ for the evaluation notebook.
"""
import os, warnings, random, json, argparse
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import layers, Model
import keras_hub
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── Configuration ──────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

CFG = {
    "MAX_SEQ_LEN": 64,
    "BATCH_SIZE": 32,
    "EPOCHS_STAGE1": 2,       # frozen backbone (guide: 1-2 epochs)
    "EPOCHS_STAGE2": 30,      # unfrozen fine-tuning (early stopping)
    "LR_STAGE1": 1e-3,        # head-only learning rate
    "LR_STAGE2": 5e-5,        # fine-tuning LR (guide: 5e-5)
    "LORA_RANK": 8,
    "LR_LORA": 2e-4,          # LoRA needs higher LR (adapters init near zero)
    "EPOCHS_LORA": 40,        # more epochs for parameter-efficient method
    "MODEL_PRESET": "roberta_base_en",
    "NUM_CLASSES": 6,
}

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Class grouping ─────────────────────────────────────────
# Group 766 sparse component labels into 6 functional categories.
# Guide (Sec 24.7.3.4): "pick 4-6 categories that are well-populated
# and clearly distinct, then group or discard the rest."
CLASS_MAP = {
    # HVAC & Climate Control
    "ACU": "HVAC", "Aircon Filter": "HVAC", "Recharge Freon": "HVAC",
    "Saloon Evaporator": "HVAC", "Saloon CCU": "HVAC",
    "Saloon CCU Fan": "HVAC", "Air Dryer": "HVAC",
    "Change Thermostat Setting": "HVAC",
    # Doors & Access
    "Door Roller": "Doors", "Cab Door Cable": "Doors",
    "Door Manifold": "Doors",
    # Electrical & Sensing
    "BCU": "Electrical", "CCD Fuse": "Electrical",
    "CCD Washer": "Electrical", "Test Point": "Electrical",
    "Speed Sensor": "Electrical", "Pressure Transducer": "Electrical",
    "Charging Resistor": "Electrical", "Battery": "Electrical",
    # Lighting & Signaling
    "Saloon Light": "Lighting", "Headlight": "Lighting",
    "Exterior Door Light": "Lighting",
    # Propulsion & Wheels
    "Traction Motor": "Propulsion", "Wheel Profiling": "Propulsion",
}
# Labels not in CLASS_MAP (Wiper, MPV, LFF, External Smoke Detector, etc.)
# intentionally fall through to "Other" in the grouping logic.
LABEL_ORDER = ["Doors", "Electrical", "HVAC", "Lighting", "Other", "Propulsion"]


def load_and_prepare_data():
    """Load SMRT dataset, apply functional grouping, split into train/val/test."""
    DATA_PATH = Path("data/smrt_maintenance_logs.csv")
    assert DATA_PATH.exists(), f"Dataset not found at {DATA_PATH}"

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} maintenance log entries")
    print(f"Original unique components: {df['label_name'].nunique()}")

    texts_all = df["fault_text"].values
    labels_raw = df["label_name"].values
    splits = df["split"].values

    # Step 1: frequency-based cutoff (>= 50 training samples)
    train_mask = splits == "train"
    label_counts = pd.Series(labels_raw[train_mask]).value_counts()
    top_labels = set(label_counts[label_counts >= 50].index)

    # Step 2: functional grouping
    labels_grouped = []
    for l in labels_raw:
        if l in top_labels and l in CLASS_MAP:
            labels_grouped.append(CLASS_MAP[l])
        else:
            labels_grouped.append("Other")
    labels_grouped = np.array(labels_grouped)

    label_names = LABEL_ORDER
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    labels_encoded = np.array([label_to_idx[l] for l in labels_grouped])

    print(f"\nFunctional categories ({len(label_names)}):")
    for name in label_names:
        count = (labels_grouped == name).sum()
        print(f"  {name:20s} {count:5d} ({100*count/len(df):.1f}%)")

    # Train/val/test split
    train_idx = np.where(splits == "train")[0]
    test_idx = np.where(splits == "test")[0]

    X_train_idx, X_val_idx = train_test_split(
        train_idx, test_size=0.15, random_state=SEED,
        stratify=labels_encoded[train_idx]
    )

    X_train_raw = texts_all[X_train_idx]
    y_train = labels_encoded[X_train_idx]
    X_val_raw = texts_all[X_val_idx]
    y_val = labels_encoded[X_val_idx]
    X_test_raw = texts_all[test_idx]
    y_test = labels_encoded[test_idx]

    print(f"\nTrain: {len(X_train_raw)}, Val: {len(X_val_raw)}, Test: {len(X_test_raw)}")

    return X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test, label_names


def augment_text(text, p_drop=0.15, p_swap=0.1):
    """Simple text augmentation: random word dropout + adjacent word swap."""
    words = text.split()
    if len(words) < 4:
        return text
    words = [w for w in words if random.random() > p_drop]
    if not words:
        return text
    for j in range(len(words) - 1):
        if random.random() < p_swap:
            words[j], words[j+1] = words[j+1], words[j]
    return " ".join(words)


def compute_baselines(X_train_raw, y_train, X_test_raw, y_test):
    """Compute majority-class and TF-IDF + LogReg baselines."""
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


def build_datasets(X_train_aug, y_train_aug, X_val_raw, y_val, X_test_raw, y_test, preprocessor):
    """Build tf.data pipelines with tokenization, caching, and prefetching."""
    def make_dataset(texts, labels, batch_size, shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices((texts, labels))
        if shuffle:
            ds = ds.shuffle(len(texts), seed=SEED)
        ds = ds.batch(batch_size)
        ds = ds.map(lambda x, y: (preprocessor(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache()
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_dataset(X_train_aug, y_train_aug, CFG["BATCH_SIZE"], shuffle=True)
    val_ds = make_dataset(X_val_raw, y_val, CFG["BATCH_SIZE"], shuffle=False)
    test_ds = make_dataset(X_test_raw, y_test, CFG["BATCH_SIZE"], shuffle=False)

    print(f"Train batches: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds


def build_model(backbone, num_classes):
    """RoBERTa backbone + classification head (guide architecture)."""
    inputs = backbone.input
    backbone_output = backbone(inputs)
    cls_output = backbone_output[:, 0, :]  # [CLS] token

    # Guide: Dropout(0.1) -> Dense(256, relu) -> Dropout(0.1) -> Dense(K, softmax)
    x = layers.Dropout(0.1)(cls_output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = Model(inputs=inputs, outputs=outputs, name="RoBERTa_FaultClassifier")
    return model


def sanity_check(model, train_ds):
    """Overfit a tiny subset to verify the pipeline is wired correctly."""
    print("\nSanity check: overfitting 64 static samples...")
    for batch_x, batch_y in train_ds.take(1):
        # Take first 64 samples from the batch
        mini_x = {k: v[:64] for k, v in batch_x.items()} if isinstance(batch_x, dict) else batch_x[:64]
        mini_y = batch_y[:64]
        break

    mini_ds = tf.data.Dataset.from_tensor_slices((mini_x, mini_y)).batch(64)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CFG["LR_STAGE1"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    hist = model.fit(mini_ds, epochs=50, verbose=0)
    final_loss = hist.history["loss"][-1]
    random_loss = np.log(CFG["NUM_CLASSES"])  # 1.79 for 6 classes
    print(f"  Final loss: {final_loss:.4f} (random={random_loss:.2f})")
    if final_loss < random_loss * 0.8:
        print("  PASSED -- pipeline is correctly wired.")
    else:
        print("  WARNING: loss did not drop sufficiently. Check pipeline.")
    return model


def train_stage1(model, backbone, train_ds, val_ds, class_weight_dict):
    """Stage 1: Freeze backbone, train classification head only."""
    backbone.trainable = False

    trainable_params = sum(p.numpy().size for p in model.trainable_weights)
    total_params = model.count_params()
    print(f"\nStage 1 — backbone FROZEN")
    print(f"  Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CFG["LR_STAGE1"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_s1 = [
        keras.callbacks.ModelCheckpoint(
            str(OUTPUT_DIR / "best_roberta_stage1.keras"),
            monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=2,
            restore_best_weights=True, verbose=1
        ),
    ]

    print(f"Training Stage 1 ({CFG['EPOCHS_STAGE1']} epochs, LR={CFG['LR_STAGE1']})...")
    history_s1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=CFG["EPOCHS_STAGE1"],
        callbacks=callbacks_s1,
        class_weight=class_weight_dict,
        verbose=1,
    )

    s1_val_acc = max(history_s1.history["val_accuracy"])
    print(f"\nStage 1 best val accuracy: {s1_val_acc:.4f}")

    with open(OUTPUT_DIR / "history_stage1.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history_s1.history.items()}, f)

    return history_s1, s1_val_acc


def train_stage2(model, backbone, train_ds, val_ds, class_weight_dict):
    """Stage 2: Unfreeze top 4 transformer layers, fine-tune with low LR."""
    # Guide: "Unfreeze the top 4 Transformer layers"
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False

    # Identify and unfreeze top 4 transformer layers
    transformer_layers = [l for l in backbone.layers if "transformer_layer" in l.name]
    if transformer_layers:
        for layer in transformer_layers[-4:]:
            layer.trainable = True
        n_unfrozen = sum(1 for l in transformer_layers if l.trainable)
        print(f"  Unfroze top {n_unfrozen} of {len(transformer_layers)} transformer layers")
    else:
        # Fallback: unfreeze top 30% of all layers
        n_layers = len(backbone.layers)
        freeze_until = int(n_layers * 0.7)
        for j, layer in enumerate(backbone.layers):
            layer.trainable = (j >= freeze_until)
        print(f"  Fallback: unfroze top 30% of {n_layers} layers")

    trainable_params = sum(p.numpy().size for p in model.trainable_weights)
    total_params = model.count_params()
    print(f"\nStage 2 — backbone UNFROZEN")
    print(f"  Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Warmup + cosine decay schedule
    total_steps = len(train_ds) * CFG["EPOCHS_STAGE2"]
    warmup_steps = int(0.1 * total_steps)
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=CFG["LR_STAGE2"],
        decay_steps=total_steps - warmup_steps,
        alpha=1e-7,
    )

    class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, warmup_steps, base_lr, decay_schedule):
            super().__init__()
            self.warmup_steps = warmup_steps
            self.base_lr = base_lr
            self.decay_schedule = decay_schedule

        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            warmup = self.base_lr * (step / tf.cast(self.warmup_steps, tf.float32))
            decayed = self.decay_schedule(step - tf.cast(self.warmup_steps, tf.float32))
            return tf.where(step < tf.cast(self.warmup_steps, tf.float32), warmup, decayed)

        def get_config(self):
            return {"warmup_steps": int(self.warmup_steps), "base_lr": float(self.base_lr)}

    lr_with_warmup = WarmupCosineDecay(warmup_steps, CFG["LR_STAGE2"], lr_schedule)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_with_warmup),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_s2 = [
        keras.callbacks.ModelCheckpoint(
            str(OUTPUT_DIR / "best_roberta_faultclassifier.keras"),
            monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4,
            restore_best_weights=True, verbose=1
        ),
    ]

    print(f"Training Stage 2 ({CFG['EPOCHS_STAGE2']} epochs, LR={CFG['LR_STAGE2']})...")
    history_s2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=CFG["EPOCHS_STAGE2"],
        callbacks=callbacks_s2,
        class_weight=class_weight_dict,
        verbose=1,
    )

    s2_val_acc = max(history_s2.history["val_accuracy"])
    print(f"\nStage 2 best val accuracy: {s2_val_acc:.4f}")

    with open(OUTPUT_DIR / "history_stage2.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history_s2.history.items()}, f)

    return history_s2, s2_val_acc


def train_lora(train_ds, val_ds, class_weight_dict, stage1_path):
    """LoRA fine-tuning comparison: low-rank adapters on attention layers."""
    backbone_lora = keras_hub.models.RobertaBackbone.from_preset(CFG["MODEL_PRESET"])
    backbone_lora.enable_lora(rank=CFG["LORA_RANK"])

    inputs_lora = backbone_lora.input
    backbone_output_lora = backbone_lora(inputs_lora)
    cls_output_lora = backbone_output_lora[:, 0, :]
    x = layers.Dropout(0.1)(cls_output_lora)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs_lora = layers.Dense(CFG["NUM_CLASSES"], activation="softmax", dtype="float32")(x)
    model_lora = Model(inputs=inputs_lora, outputs=outputs_lora, name="RoBERTa_LoRA")

    lora_trainable = sum(p.numpy().size for p in model_lora.trainable_weights)
    lora_total = model_lora.count_params()
    print(f"\nLoRA model — Trainable: {lora_trainable:,} / {lora_total:,} ({100*lora_trainable/lora_total:.2f}%)")

    # Transfer Stage 1 head weights (frozen-backbone alignment)
    stage1_model = keras.saving.load_model(stage1_path)
    for src_layer, dst_layer in zip(stage1_model.layers[-4:], model_lora.layers[-4:]):
        if type(src_layer) == type(dst_layer) and src_layer.get_weights():
            try:
                dst_layer.set_weights(src_layer.get_weights())
            except ValueError:
                pass
    del stage1_model
    print("Transferred Stage 1 head weights to LoRA model")

    model_lora.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CFG["LR_LORA"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_lora = [
        keras.callbacks.ModelCheckpoint(
            str(OUTPUT_DIR / "best_roberta_lora.keras"),
            monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1
        ),
    ]

    print(f"Training LoRA ({CFG['EPOCHS_LORA']} epochs, rank={CFG['LORA_RANK']}, LR={CFG['LR_LORA']})...")
    history_lora = model_lora.fit(
        train_ds, validation_data=val_ds,
        epochs=CFG["EPOCHS_LORA"],
        callbacks=callbacks_lora,
        class_weight=class_weight_dict,
        verbose=1,
    )

    with open(OUTPUT_DIR / "history_lora.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history_lora.history.items()}, f)

    return model_lora, history_lora, lora_trainable, lora_total


def main():
    print(f"TensorFlow {tf.__version__}, Keras {keras.__version__}")
    if tf.config.list_physical_devices("GPU"):
        keras.config.set_dtype_policy("mixed_float16")
        print("Mixed precision enabled (float16)")
    print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

    # ── Data ──
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test, label_names = load_and_prepare_data()
    CFG["LABEL_NAMES"] = label_names

    # Class weights
    class_weights_arr = compute_class_weight("balanced", classes=np.arange(CFG["NUM_CLASSES"]), y=y_train)
    class_weight_dict = {i: float(w) for i, w in enumerate(class_weights_arr)}

    # Augment training data (3x)
    aug_texts, aug_labels = [], []
    for text, label in zip(X_train_raw, y_train):
        aug_texts.append(text)
        aug_labels.append(label)
        for _ in range(2):
            aug_texts.append(augment_text(text))
            aug_labels.append(label)
    X_train_aug = np.array(aug_texts)
    y_train_aug = np.array(aug_labels)
    print(f"Training: {len(X_train_raw)} original -> {len(X_train_aug)} with augmentation (3x)")

    # ── Baselines ──
    baselines = compute_baselines(X_train_raw, y_train, X_test_raw, y_test)

    # ── Tokenization + datasets ──
    preprocessor = keras_hub.models.RobertaPreprocessor.from_preset(
        CFG["MODEL_PRESET"], sequence_length=CFG["MAX_SEQ_LEN"]
    )
    train_ds, val_ds, test_ds = build_datasets(
        X_train_aug, y_train_aug, X_val_raw, y_val, X_test_raw, y_test, preprocessor
    )

    # ── Model ──
    backbone = keras_hub.models.RobertaBackbone.from_preset(CFG["MODEL_PRESET"])
    model = build_model(backbone, CFG["NUM_CLASSES"])
    print(f"Model: {model.count_params():,} params")

    # ── Sanity check ──
    sanity_check(model, train_ds)

    # Rebuild with fresh weights after sanity check
    backbone = keras_hub.models.RobertaBackbone.from_preset(CFG["MODEL_PRESET"])
    model = build_model(backbone, CFG["NUM_CLASSES"])

    # ── Stage 1 ──
    history_s1, s1_val_acc = train_stage1(model, backbone, train_ds, val_ds, class_weight_dict)

    # ── Stage 2 ──
    history_s2, s2_val_acc = train_stage2(model, backbone, train_ds, val_ds, class_weight_dict)

    # ── Test evaluation ──
    model.load_weights(str(OUTPUT_DIR / "best_roberta_faultclassifier.keras"))
    y_pred_probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    test_acc = float(np.mean(y_pred == y_test))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Macro F1:      {macro_f1:.4f}")

    # ── LoRA ──
    model_lora, history_lora, lora_trainable, lora_total = train_lora(
        train_ds, val_ds, class_weight_dict, str(OUTPUT_DIR / "best_roberta_stage1.keras")
    )
    model_lora.load_weights(str(OUTPUT_DIR / "best_roberta_lora.keras"))
    y_pred_lora_probs = model_lora.predict(test_ds, verbose=0)
    y_pred_lora = np.argmax(y_pred_lora_probs, axis=1)
    lora_acc = float(np.mean(y_pred_lora == y_test))
    lora_f1 = float(f1_score(y_test, y_pred_lora, average="macro"))
    print(f"\nLoRA test accuracy: {lora_acc:.4f}")
    print(f"LoRA macro F1:      {lora_f1:.4f}")

    # ── Save predictions ──
    np.savez(
        OUTPUT_DIR / "test_predictions.npz",
        y_test=y_test, y_pred=y_pred, y_pred_probs=y_pred_probs,
        y_pred_lora=y_pred_lora, y_pred_lora_probs=y_pred_lora_probs,
        label_names=label_names, X_test_raw=X_test_raw,
    )

    # ── Save config ──
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

    # ── Summary ──
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — Summary")
    print("=" * 60)
    print(f"  Majority baseline:   {baselines['majority_acc']:.4f}")
    print(f"  TF-IDF + LogReg:     {baselines['tfidf_acc']:.4f}")
    print(f"  Stage 1 val acc:     {s1_val_acc:.4f}")
    print(f"  Stage 2 val acc:     {s2_val_acc:.4f}")
    print(f"  Stage 2 test acc:    {test_acc:.4f} (macro F1: {macro_f1:.4f})")
    print(f"  LoRA test acc:       {lora_acc:.4f} (macro F1: {lora_f1:.4f})")
    print(f"\nArtifacts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
