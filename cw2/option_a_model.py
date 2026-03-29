"""
option_a_model.py — Model architecture and configuration.
CW2 Option A: Fine-tuning RoBERTa for fault classification.
"""
import tensorflow as tf
import keras
from keras import layers, Model

# ── Training configuration ──────────────────────────────────────────────────
CFG = {
    "MAX_SEQ_LEN": 64,
    "BATCH_SIZE": 32,
    "EPOCHS_STAGE1": 2,        # Frozen backbone (Ch 24: 1-2 epochs)
    "EPOCHS_STAGE2": 30,       # Unfrozen fine-tuning (early stopping)
    "LR_STAGE1": 1e-3,         # Head-only learning rate
    "LR_STAGE2": 5e-5,         # Fine-tuning LR (Ch 24: 5e-5)
    "LORA_RANK": 8,
    "LR_LORA": 2e-4,           # LoRA needs higher LR (adapters initialised near zero)
    "EPOCHS_LORA": 40,         # More epochs for parameter-efficient method
    "MODEL_PRESET": "roberta_base_en",
    "NUM_CLASSES": 6,
}


class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay.

    Warmup avoids large gradient updates at the start of Stage 2,
    which is particularly important when unfreezing pre-trained layers.
    """

    def __init__(self, warmup_steps, base_lr, decay_schedule):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.decay_schedule = decay_schedule

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = self.base_lr * (step / tf.cast(self.warmup_steps, tf.float32))
        decayed = self.decay_schedule(
            step - tf.cast(self.warmup_steps, tf.float32)
        )
        return tf.where(step < tf.cast(self.warmup_steps, tf.float32),
                        warmup, decayed)

    def get_config(self):
        return {
            "warmup_steps": int(self.warmup_steps),
            "base_lr": float(self.base_lr),
        }


def build_model(backbone, num_classes):
    """Attach a classification head to a RoBERTa backbone.

    Architecture (Ch 24):
        [CLS] token output -> Dropout(0.1) -> Dense(256, relu)
        -> Dropout(0.1) -> Dense(num_classes, softmax)

    Parameters
    ----------
    backbone : keras_hub RobertaBackbone
    num_classes : int

    Returns
    -------
    keras.Model
    """
    inputs = backbone.input
    backbone_output = backbone(inputs)
    cls_output = backbone_output[:, 0, :]   # [CLS] token representation

    x = layers.Dropout(0.1)(cls_output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           dtype="float32")(x)

    model = Model(inputs=inputs, outputs=outputs,
                  name="RoBERTa_FaultClassifier")
    return model
