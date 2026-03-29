#!/bin/bash
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=16G
#SBATCH -p tpg-gpu-small
#SBATCH --qos=short
#SBATCH -t 0-08:00:00
#SBATCH --job-name=CW2A_train
#SBATCH -o CW2A_%j.out
#SBATCH -e CW2A_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=szbc46@durham.ac.uk

source /etc/profile
module purge
module load cuda/12.3-cudnn8.9

source ~/miniconda3/etc/profile.d/conda.sh
conda activate CW2

cd ~/ncc-transfer/cw2

stdbuf -oL python option_a_main.py 2>&1 | tee training.log

conda deactivate
