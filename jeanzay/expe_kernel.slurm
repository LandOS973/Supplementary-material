#!/bin/bash
#SBATCH --job-name=expe_jz          
#SBATCH --output=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/expe_%j.out 
#SBATCH --error=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/expe_%j.err  
#SBATCH --ntasks=1                  
#SBATCH --gres=gpu:1                
#SBATCH --cpus-per-task=10          
#SBATCH --hint=nomultithread        
#SBATCH --time=08:00:00             
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
echo "Lancement de main_expe.py..."

export PYTHONUNBUFFERED=1

M_VALUES=""
while getopts "m:" opt; do
  case "$opt" in
    m) M_VALUES="$OPTARG" ;;
  esac
done
shift $((OPTIND - 1))

JOB_NAME="expe_jz"
if [ -n "$M_VALUES" ]; then
  JOB_SUFFIX="${M_VALUES//,/_}"
  JOB_SUFFIX="${JOB_SUFFIX// /_}"
  JOB_NAME="expe_${JOB_SUFFIX}"
  scontrol update JobName="$JOB_NAME" JobId="$SLURM_JOB_ID" >/dev/null 2>&1 || true
fi

echo "Job name: $JOB_NAME"

if [ -n "$M_VALUES" ]; then
  srun python source_code/main_expe_overall.py -m "$M_VALUES"
else
  srun python source_code/main_expe_overall.py
fi

echo "Job terminé à $(date)"

# sbatch expe_kernel.slurm -m 10