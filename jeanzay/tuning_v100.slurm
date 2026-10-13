#!/bin/bash
#SBATCH --job-name=tuning
#SBATCH --output=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/tuning_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/tuning_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH -C v100-32g
#SBATCH --time=03:00:00             # 1 config par tache sur les instances de reglage (3/classe) ; a recalibrer
#SBATCH --account=ndg@v100
#SBATCH --array=0-99%50             # 100 configs, 50 GPU en parallele

# --- Recherche d'hyperparametres sur les instances de reglage ---
# Une tache = une config : taches 0-49 -> config K de tuning_ppsn.json,
# taches 50-99 -> config K-50 de tuning_evocop.json. Resultats dans results/tuning/<config>/.
# Prerequis : source_code/instances_tuning/ genere sur JZ
#   (python source_code/utils/generate_tuning_instances.py).
# Relancable : les instances deja faites sont sautees.
#
# Usage :
#   sbatch jeanzay/tuning_v100.slurm
#   sbatch --array=3,67 jeanzay/tuning_v100.slurm        # relancer des taches precises (67 = evocop 17)
#   sbatch --qos=qos_gpu-dev --time=00:30:00 --array=0 jeanzay/tuning_v100.slurm   # test rapide

echo "Job démarré sur $(hostname) à $(date)"

cd $WORK/Supplementary-material
mkdir -p outputs

module purge
module load miniforge/24.9.0
conda activate ppo_env

export PYTHONUNBUFFERED=1

NB_PER_GRID=50
if [ "$SLURM_ARRAY_TASK_ID" -lt "$NB_PER_GRID" ]; then
  GRID="jeanzay/grids/tuning_ppsn.json"
else
  GRID="jeanzay/grids/tuning_evocop.json"
fi
K=$((SLURM_ARRAY_TASK_ID % NB_PER_GRID))
echo "Tache ${SLURM_ARRAY_TASK_ID} : config ${K} de ${GRID}"

if [ ! -d source_code/instances_tuning ]; then
  echo "source_code/instances_tuning absent : lancer generate_tuning_instances.py d'abord." >&2
  exit 1
fi

srun python source_code/expe/main_expe_overall.py \
  --instances tuning \
  -g "$GRID" \
  --shard "${K}/${NB_PER_GRID}"

echo "Job terminé à $(date)"
