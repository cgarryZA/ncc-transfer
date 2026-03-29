#!/bin/bash
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=32G
#SBATCH -p tpg-gpu-small
#SBATCH -t 0-08:00:00
#SBATCH --job-name=CW1_unet
#SBATCH -o CW1_%j.out
#SBATCH -e CW1_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=szbc46@durham.ac.uk

source /etc/profile
module purge
module load cuda/12.3-cudnn8.9

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CW2

cd ~/ncc-transfer/cw1

stdbuf -oL jupyter nbconvert \
    --execute \
    --to notebook \
    --output notebook.ipynb \
    --ExecutePreprocessor.timeout=-1 \
    notebook.ipynb 2>&1 | tee cw1_run.log

echo "Exit code: $?"
conda deactivate
