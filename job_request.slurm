#!/bin/bash
# ====================================
#SBATCH --job-name=torch-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=0-05:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
# ====================================


source ~/software/init-conda
conda activate pytorch

python model_code.py > output.txt 2>&1  # Redirect both stdout and stderr

