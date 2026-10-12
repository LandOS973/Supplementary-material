#!/bin/bash
#SBATCH --job-name=timing          
#SBATCH --output=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/timing_%j.out 
#SBATCH --error=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/timing_%j.err  
#SBATCH --ntasks=1                  
#SBATCH --gres=gpu:1                
#SBATCH --cpus-per-task=10          
#SBATCH --hint=nomultithread        
#SBATCH --time=04:00:00
#SBATCH --account=ndg@v100

# --- Préparation ---
echo "Job démarré sur $(hostname) à $(date)"

# 1. Se placer dans le bon dossier (Sécurité absolue)
cd $WORK/Supplementary-material

# 2. Créer le dossier s'il n'existe pas
mkdir -p outputs

# 3. Charger l'environnement Conda
module purge
module load miniforge/24.9.0
conda activate ppo_env

# --- Lancement ---
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun python source_code/expe/main_timing.py -g jeanzay/grids/timing.json "$@"
echo "Job terminé à $(date)"
