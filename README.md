# NCC Transfer — Deep Learning Coursework

Transfer repo for running CW1 and CW2 on the Durham NCC cluster.

## Setup on NCC

```bash
# Clone this repo
git clone <repo-url> ~/ncc-transfer
cd ~/ncc-transfer

# The concrete crack dataset must be placed at data/ds/
# with subdirectories img/ (458 JPEGs) and ann/ (458 JSON annotations)
# Upload via scp or rsync separately (too large for git)
```

## CW1: Crack Segmentation (U-Net)

```bash
cd ~/ncc-transfer/cw1
sbatch jobscript.sh
```

Runs the full notebook: data loading, training (~2 hrs), evaluation, visualisation.
Output: `notebook.ipynb` with all cells executed, `best_unet.keras` checkpoint.

## CW2: Fault Classification (RoBERTa)

```bash
# Step 1: Train
cd ~/ncc-transfer/cw2
sbatch jobscript.sh

# Step 2: Evaluate (after training completes)
jupyter nbconvert --execute --to notebook --output notebook.ipynb notebook.ipynb
```

Training produces artifacts in `output/`. The notebook loads them for evaluation only.
