"""
option_a_train.py — Training routines for two-stage fine-tuning and LoRA.
CW2 Option A: Fine-tuning RoBERTa for fault classification.
"""
import numpy as np
import keras
import keras_hub
from keras import layers, Model
from pathlib import Path

from option_a_model import CFG, WarmupCosineDecay, build_model


OUTPUT_DIR = Path("output")


def sanity_check(model, train_ds):
    """Overfit a tiny subset to verify the pipeline is wired correctly.

    Takes 64 samples from the first batch and trains for 50 epochs.
    If loss drops well below random chance, the pipeline is correct.
    """
    print("\nSanity check: overfitting 64 static samples...")
    for batch_x, batch_y in train_ds.take(1):
        mini_x = ({k: v[:64] for k, v in batch_x.items()}
                  if isinstance(batch_x, dict) else batch_x[:64])
        mini_y = batch_y[:64]
        break

    mini_ds = __import__("tensorflow").data.Dataset.from_tensor_slices(
        (mini_x, mini_y)
    ).batch(64)

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
    """Stage 1: freeze backbone, train classification head only.

    Ch 24: 1-2 epochs at lr=1e-3. The head learns to map pre-trained
    representations to the target fault categories.
    """
    import json
    backbone.trainable = False

    trainable_params = sum(p.numpy().size for p in model.trainable_weights)
    total_params = model.count_params()
    print(f"\nStage 1 -- backbone FROZEN")
    print(f"  Trainable: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CFG["LR_STAGE1"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_s1 = [
        keras.callbacks.ModelCheckpoint(
            str(OUTPUT_DIR / "best_roberta_stage1.keras"),
            monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=2,
            restore_best_weights=True, verbose=1,
        ),
    ]

    print(f"Training Stage 1 ({CFG['EPOCHS_STAGE1']} epochs, "
          f"LR={CFG['LR_STAGE1']})...")
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
        json.dump(
            {k: [float(v) for v in vals]
             for k, vals in history_s1.history.items()}, f
        )
    return history_s1, s1_val_acc


def train_stage2(model, backbone, train_ds, val_ds, class_weight_dict):
    """Stage 2: unfreeze top 4 Transformer layers, fine-tune with low LR.

    Ch 24: lr=5e-5, cosine decay with linear warmup over 10% of steps.
    Early stopping with patience 4.
    """
    import json
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False

    # Unfreeze top 4 Transformer layers.
    transformer_layers = [
        l for l in backbone.layers if "transformer_layer" in l.name
    ]
    if transformer_layers:
        for layer in transformer_layers[-4:]:
            layer.trainable = True
        n_unfrozen = sum(1 for l in transformer_layers if l.trainable)
        print(f"  Unfroze top {n_unfrozen} of {len(transformer_layers)} "
              "transformer layers")
    else:
        # Fallback: unfreeze top 30% of all layers.
        n_layers = len(backbone.layers)
        freeze_until = int(n_layers * 0.7)
        for j, layer in enumerate(backbone.layers):
            layer.trainable = (j >= freeze_until)
        print(f"  Fallback: unfroze top 30% of {n_layers} layers")

    trainable_params = sum(p.numpy().size for p in model.trainable_weights)
    total_params = model.count_params()
    print(f"\nStage 2 -- backbone UNFROZEN")
    print(f"  Trainable: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    # Warmup + cosine decay schedule.
    total_steps = len(train_ds) * CFG["EPOCHS_STAGE2"]
    warmup_steps = int(0.1 * total_steps)
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=CFG["LR_STAGE2"],
        decay_steps=total_steps - warmup_steps,
        alpha=1e-7,
    )
    lr_with_warmup = WarmupCosineDecay(
        warmup_steps, CFG["LR_STAGE2"], lr_schedule
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_with_warmup),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks_s2 = [
        keras.callbacks.ModelCheckpoint(
            str(OUTPUT_DIR / "best_roberta_faultclassifier.keras"),
            monitor="val_accuracy", mode="max",
            save_best_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4,
            restore_best_weights=True, verbose=1,
        ),
    ]

    print(f"Training Stage 2 ({CFG['EPOCHS_STAGE2']} epochs, "
          f"LR={CFG['LR_STAGE2']})...")
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
        json.dump(
            {k: [float(v) for v in vals]
             for k, vals in history_s2.history.items()}, f
        )
    return history_s2, s2_val_acc


def train_lora(train_ds, val_ds, class_weight_dict, stage1_path):
    """LoRA fine-tuning: low-rank adapters on attention layers.

    Ch 24 / Ch 16: rank-8 LoRA adapters update <2% of total parameters.
    Classification head is transferred from Stage 1 for aligned initialisation.
    """
    import json
    backbone_lora = keras_hub.models.RobertaBackbone.from_preset(
        CFG["MODEL_PRESET"]
    )
    backbone_lora.enable_lora(rank=CFG["LORA_RANK"])

    inputs_lora = backbone_lora.input
    backbone_output_lora = backbone_lora(inputs_lora)
    cls_output_lora = backbone_output_lora[:, 0, :]
    x = layers.Dropout(0.1)(cls_output_lora)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs_lora = layers.Dense(
        CFG["NUM_CLASSES"], activation="softmax", dtype="float32"
    )(x)
    model_lora = Model(
        inputs=inputs_lora, outputs=outputs_lora, name="RoBERTa_LoRA"
    )

    lora_trainable = sum(
        p.numpy().size for p in model_lora.trainable_weights
    )
    lora_total = model_lora.count_params()
    print(f"\nLoRA model -- Trainable: {lora_trainable:,} / {lora_total:,} "
          f"({100 * lora_trainable / lora_total:.2f}%)")

    # Transfer Stage 1 head weights for aligned starting point.
    stage1_model = keras.saving.load_model(stage1_path)
    for src_layer, dst_layer in zip(
        stage1_model.layers[-4:], model_lora.layers[-4:]
    ):
        if (type(src_layer) == type(dst_layer)
                and src_layer.get_weights()):
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
            save_best_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1,
        ),
    ]

    print(f"Training LoRA ({CFG['EPOCHS_LORA']} epochs, "
          f"rank={CFG['LORA_RANK']}, LR={CFG['LR_LORA']})...")
    history_lora = model_lora.fit(
        train_ds, validation_data=val_ds,
        epochs=CFG["EPOCHS_LORA"],
        callbacks=callbacks_lora,
        class_weight=class_weight_dict,
        verbose=1,
    )

    with open(OUTPUT_DIR / "history_lora.json", "w") as f:
        json.dump(
            {k: [float(v) for v in vals]
             for k, vals in history_lora.history.items()}, f
        )
    return model_lora, history_lora, lora_trainable, lora_total
