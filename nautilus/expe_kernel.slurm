#!/bin/bash
#SBATCH --job-name=expe
#SBATCH --qos=short
#SBATCH -p gpu
#SBATCH --output=output/expe_%j.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:03:00

echo "Run main_ppo_eda"

export PYTHONUNBUFFERED=1
PYTHON=/micromamba/$USER/envs/ppo_env/bin/python

srun $PYTHON ../source_code/main_expe.py
