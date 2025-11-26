#!/bin/bash
#SBATCH --job-name=ppo
#SBATCH --qos=short
#SBATCH -p gpu
#SBATCH --output=output/ppo_%j.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00

echo "Run main_ppo_eda"

# 1) Chemin direct vers le Python de ton env micromamba
PYTHON=/micromamba/$USER/envs/ppo_env/bin/python

# 2) Lancer ton script avec ce python-là
srun $PYTHON ../source_code/main_ppo_eda.py -m visualization=false
