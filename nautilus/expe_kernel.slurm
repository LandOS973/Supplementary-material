#!/bin/bash
#SBATCH --job-name=expe
#SBATCH --qos=short
#SBATCH -p gpu
#SBATCH --output=/home/t.lan@univ-angers.fr/Supplementary-material/outputs/expe_%j.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00

echo "Job démarré sur $(hostname) à $(date)"
echo "Run main_expe"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/micromamba/$USER/envs/ppo_env/bin/python

srun $PYTHON /home/t.lan@univ-angers.fr/Supplementary-material/source_code/main_expe.py
echo "Job terminé à $(date)"