#!/bin/bash
#SBATCH --job-name=expe_jz          
#SBATCH --output=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/expe_%j.out 
#SBATCH --error=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/expe_%j.err  
#SBATCH --ntasks=1                  
#SBATCH --gres=gpu:1                
#SBATCH --cpus-per-task=10          
#SBATCH --hint=nomultithread        
#SBATCH --time=12:00:00             # augmenter (ex: --time=20:00:00) pour -g ppsn_baseline a n=512 (14 combos QUBO+NK+NK3, ~1.5h/combo observe sur PEDA/PPBIL)
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
GRID="wip"
while getopts "m:g:" opt; do
  case "$opt" in
    m) M_VALUES="$OPTARG" ;;
    g) GRID="$OPTARG" ;;
  esac
done
shift $((OPTIND - 1))

JOB_NAME="expe_${GRID}"
if [ -n "$M_VALUES" ]; then
  JOB_SUFFIX="${M_VALUES//,/_}"
  JOB_SUFFIX="${JOB_SUFFIX// /_}"
  JOB_NAME="${JOB_NAME}_${JOB_SUFFIX}"
fi
scontrol update JobName="$JOB_NAME" JobId="$SLURM_JOB_ID" >/dev/null 2>&1 || true

echo "Job name: $JOB_NAME  Grid: $GRID"

if [ -n "$M_VALUES" ]; then
  srun python source_code/expe/main_expe_overall.py -g "$GRID" -m "$M_VALUES"
else
  srun python source_code/expe/main_expe_overall.py -g "$GRID"
fi

echo "Job terminé à $(date)"

# sbatch expe_kernel.slurm -m 10                       # grid "wip" (defaut), M=10 seulement
# sbatch --time=20:00:00 expe_kernel.slurm -g ppsn_baseline   # reproduit la config PPSN exacte (M7 L13 eps0.08 g0.015), tous dims/problemes decouverts (n=512 nouveau, 64/128/256 skippes car deja fait)