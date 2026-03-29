"""
option_a_data.py — Data loading, preprocessing, and augmentation.
CW2 Option A: Fine-tuning RoBERTa for fault classification.
"""
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split

SEED = 42

# ── Functional class mapping ────────────────────────────────────────────────
# 766 sparse component labels grouped into 6 functional fault categories.
# Guide (Ch 24): "pick 4-6 categories that are well-populated and clearly
# distinct, then group or discard the rest."
CLASS_MAP = {
    # HVAC and Climate Control
    "ACU": "HVAC", "Aircon Filter": "HVAC", "Recharge Freon": "HVAC",
    "Saloon Evaporator": "HVAC", "Saloon CCU": "HVAC",
    "Saloon CCU Fan": "HVAC", "Air Dryer": "HVAC",
    "Change Thermostat Setting": "HVAC",
    # Doors and Access
    "Door Roller": "Doors", "Cab Door Cable": "Doors",
    "Door Manifold": "Doors",
    # Electrical and Sensing
    "BCU": "Electrical", "CCD Fuse": "Electrical",
    "CCD Washer": "Electrical", "Test Point": "Electrical",
    "Speed Sensor": "Electrical", "Pressure Transducer": "Electrical",
    "Charging Resistor": "Electrical", "Battery": "Electrical",
    # Lighting and Signalling
    "Saloon Light": "Lighting", "Headlight": "Lighting",
    "Exterior Door Light": "Lighting",
    # Propulsion and Wheels
    "Traction Motor": "Propulsion", "Wheel Profiling": "Propulsion",
}
# Labels not in CLASS_MAP fall through to "Other" in the grouping logic.
LABEL_ORDER = ["Doors", "Electrical", "HVAC", "Lighting", "Other", "Propulsion"]


def load_and_prepare_data(data_path="data/smrt_maintenance_logs.csv"):
    """Load SMRT dataset, apply functional grouping, split into train/val/test.

    Returns
    -------
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test : arrays
    label_names : list of str
    """
    data_path = Path(data_path)
    assert data_path.exists(), f"Dataset not found at {data_path}"

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} maintenance log entries")
    print(f"Original unique components: {df['label_name'].nunique()}")

    texts_all = df["fault_text"].values
    labels_raw = df["label_name"].values
    splits = df["split"].values

    # Frequency-based cutoff: keep components with >= 50 training samples.
    train_mask = splits == "train"
    label_counts = pd.Series(labels_raw[train_mask]).value_counts()
    top_labels = set(label_counts[label_counts >= 50].index)

    # Functional grouping.
    labels_grouped = np.array([
        CLASS_MAP[l] if (l in top_labels and l in CLASS_MAP) else "Other"
        for l in labels_raw
    ])

    label_names = LABEL_ORDER
    label_to_idx = {name: i for i, name in enumerate(label_names)}
    labels_encoded = np.array([label_to_idx[l] for l in labels_grouped])

    print(f"\nFunctional categories ({len(label_names)}):")
    for name in label_names:
        count = (labels_grouped == name).sum()
        print(f"  {name:20s} {count:5d} ({100 * count / len(df):.1f}%)")

    # Train/val/test split (val carved from train split, stratified).
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
    """Simple text augmentation: random word dropout and adjacent word swap.

    Parameters
    ----------
    text : str
        Input fault description.
    p_drop : float
        Probability of dropping each word.
    p_swap : float
        Probability of swapping adjacent word pairs.

    Returns
    -------
    str : augmented text.
    """
    words = text.split()
    if len(words) < 4:
        return text
    words = [w for w in words if random.random() > p_drop]
    if not words:
        return text
    for j in range(len(words) - 1):
        if random.random() < p_swap:
            words[j], words[j + 1] = words[j + 1], words[j]
    return " ".join(words)


def build_augmented_training_set(X_train_raw, y_train, n_copies=2):
    """Augment training data by generating n_copies extra augmented versions.

    Parameters
    ----------
    X_train_raw : array of str
    y_train : array of int
    n_copies : int
        Number of augmented copies per original sample (default 2, giving 3x total).

    Returns
    -------
    X_aug, y_aug : augmented arrays.
    """
    aug_texts, aug_labels = [], []
    for text, label in zip(X_train_raw, y_train):
        aug_texts.append(text)
        aug_labels.append(label)
        for _ in range(n_copies):
            aug_texts.append(augment_text(text))
            aug_labels.append(label)
    X_aug = np.array(aug_texts)
    y_aug = np.array(aug_labels)
    print(f"Training: {len(X_train_raw)} original -> {len(X_aug)} with augmentation ({n_copies + 1}x)")
    return X_aug, y_aug


def build_datasets(X_train_aug, y_train_aug, X_val_raw, y_val,
                   X_test_raw, y_test, preprocessor, batch_size, seed=SEED):
    """Build tf.data pipelines with tokenisation, caching, and prefetching.

    Parameters
    ----------
    preprocessor : keras_hub preprocessor with sequence_length set.
    batch_size : int

    Returns
    -------
    train_ds, val_ds, test_ds : tf.data.Dataset
    """
    def make_dataset(texts, labels, shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices((texts, labels))
        if shuffle:
            ds = ds.shuffle(len(texts), seed=seed)
        ds = ds.batch(batch_size)
        ds = ds.map(lambda x, y: (preprocessor(x), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache()
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_dataset(X_train_aug, y_train_aug, shuffle=True)
    val_ds = make_dataset(X_val_raw, y_val, shuffle=False)
    test_ds = make_dataset(X_test_raw, y_test, shuffle=False)

    print(f"Train batches: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds
