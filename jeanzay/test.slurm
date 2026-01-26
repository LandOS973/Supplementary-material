#!/bin/bash
#SBATCH --job-name=ppo_jz           # Nom du job
#SBATCH --output=output/test_%j.out # Fichier de sortie standard (%j = ID du job)
#SBATCH --error=output/test_%j.err  # Fichier d'erreur
#SBATCH --ntasks=1                  # Nombre de tâches (processus)
#SBATCH --gres=gpu:1                # Nombre de GPU 
#SBATCH --cpus-per-task=10          # ⚠️ OBLIGATOIRE sur JZ : 10 CPUs par GPU (ratio 1/4 du noeud)
#SBATCH --hint=nomultithread        # ⚠️ OBLIGATOIRE sur JZ : Désactive l'hyperthreading (performance)
#SBATCH --time=00:10:00             # Temps max (HH:MM:SS)
#SBATCH --account=ndg@v100          # Compte de facturation

# 1. Se placer dans le dossier de travail (sécurité)
# On s'assure d'être dans le dossier racine du projet
cd $WORK/Supplementary-material

# 2. Chargement de l'environnement
module purge
module load miniforge/24.9.0
conda activate ppo_env

# 3. Vérification
echo "Python utilisé : $(which python)"
echo "Dossier actuel : $(pwd)"

# 4. Lancement du script
# srun est important pour que Slurm gère bien les ressources
srun python source_code/main_ppo_eda.py -m visualization=false