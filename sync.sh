#!/bin/bash
# Sync files from the main working directory into the transfer repo.
# Run this before git push to ensure NCC gets the latest versions.
# Usage: bash sync.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT="$(dirname "$SCRIPT_DIR")"

echo "Syncing from $PARENT into $SCRIPT_DIR..."

# CW1
cp "$PARENT/cw1/notebook.ipynb" "$SCRIPT_DIR/cw1/"

# CW2 — modular Python files (Ch 24 structure)
for f in option_a_main.py option_a_data.py option_a_model.py option_a_train.py option_a_evaluate.py; do
    cp "$PARENT/cw2/$f" "$SCRIPT_DIR/cw2/"
done
cp "$PARENT/cw2/jobscript.sh" "$SCRIPT_DIR/cw2/"
cp "$PARENT/cw2/notebook.ipynb" "$SCRIPT_DIR/cw2/"
cp "$PARENT/cw2/environment.yml" "$SCRIPT_DIR/cw2/"
cp "$PARENT/cw2/data/smrt_maintenance_logs.csv" "$SCRIPT_DIR/cw2/data/"

# CW2 tests
mkdir -p "$SCRIPT_DIR/cw2/tests"
cp "$PARENT/cw2/tests/"*.py "$SCRIPT_DIR/cw2/tests/"

# CW2 small output artefacts
for f in baselines.json config.json history_stage1.json history_stage2.json history_lora.json test_predictions.npz; do
    cp "$PARENT/cw2/output/$f" "$SCRIPT_DIR/cw2/output/" 2>/dev/null
done

echo "Done. Now: git add -A && git commit -m 'sync' && git push"
