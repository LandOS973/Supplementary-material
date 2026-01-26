#!/bin/bash
#SBATCH --job-name=expe_jz          # Nom du job
#SBATCH --output=output/expe_%j.out # Fichier de sortie standard
#SBATCH --error=output/expe_%j.err  # Fichier d'erreur (Séparé pour mieux débugger)
#SBATCH --ntasks=1                  # 1 Processus
#SBATCH --gres=gpu:1                # 1 GPU
#SBATCH --cpus-per-task=10          # ⚠️ OBLIGATOIRE JZ (1/4 du noeud V100)
#SBATCH --hint=nomultithread        # ⚠️ OBLIGATOIRE JZ (Pas d'hyperthreading)
#SBATCH --time=08:00:00             # Durée max
#SBATCH --account=ndg@v100          # Ton compte projet (ou @a100 si besoin)

# --- Préparation ---
echo "Job démarré sur $(hostname) à $(date)"

# 1. Se placer dans le bon dossier (Sécurité absolue)
cd $WORK/Supplementary-material

# 2. Créer le dossier d'output s'il n'existe pas (évite le crash au démarrage)
mkdir -p output

# 3. Charger l'environnement Conda
module purge
module load miniforge/24.9.0
conda activate ppo_env

# --- Lancement ---
echo "Lancement de main_expe.py..."

# Force Python à afficher les logs en temps réel (pas de buffer)
export PYTHONUNBUFFERED=1

# srun est essentiel pour que Slurm surveille le processus python
srun python source_code/main_expe.py

echo "Job terminé à $(date)"