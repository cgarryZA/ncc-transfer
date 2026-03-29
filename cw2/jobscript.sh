#!/bin/bash
#SBATCH -N 1                          # 1 node
#SBATCH -c 2                          # 2 CPU cores
#SBATCH --gres=gpu:1                  # 1 GPU (any type)
#SBATCH --mem=16G                     # 16 GB system RAM
#SBATCH -p tpg-gpu-small              # Taught PG partition
#SBATCH --qos=short                   # Up to 2 days walltime
#SBATCH -t 0-08:00:00                 # 8-hour time limit
#SBATCH --job-name=CW2A_train         # Job name
#SBATCH -o CW2A_%j.out                # stdout -> file (%j = job ID)
#SBATCH -e CW2A_%j.err                # stderr -> file
#SBATCH --mail-type=ALL               # Email on all events
#SBATCH --mail-user=szbc46@durham.ac.uk

source /etc/profile
module purge
module load cuda/12.3-cudnn8.9

source ~/anaconda3/etc/profile.d/conda.sh
conda activate CW2

cd "$(dirname "$0")"

stdbuf -oL python option_a_main.py 2>&1 | tee training.log

conda deactivate
