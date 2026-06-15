#!/bin/bash
#SBATCH --job-name=test_main
#SBATCH --qos=short
#SBATCH -p gpu
#SBATCH --output=/home/t.lan@univ-angers.fr/Supplementary-material/outputs/test_main_%j.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00

echo "Job démarré sur $(hostname) à $(date)"

PYTHON=/micromamba/$USER/envs/ppo_env/bin/python
export PYTHONUNBUFFERED=1

# Passe les args directement à main.py (overrides Hydra via -m ou positionnels)
# ex: sbatch test_main.slurm verbose=false budget=1000 problem=nk3
srun $PYTHON -B /home/t.lan@univ-angers.fr/Supplementary-material/source_code/main.py visualization=false "$@"

echo "Job terminé à $(date)"

# Exemples de lancement :
# sbatch test_main.slurm                                      # config par défaut
# sbatch test_main.slurm budget=1000 verbose=false            # override rapide
# sbatch test_main.slurm agent=ppo problem=nk3 seed=42        # changer agent/problem
