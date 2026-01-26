#!/bin/bash
#SBATCH --job-name=ppo
#SBATCH --qos=short
#SBATCH -p gpu
#SBATCH --output=output/test_%j.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00

echo "Run main_ppo_eda"

PYTHON=/micromamba/$USER/envs/ppo_env/bin/python

srun $PYTHON ../source_code/main_ppo_eda.py -m visualization=false
