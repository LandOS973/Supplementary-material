#!/bin/bash
#SBATCH --job-name=baselinestd
#SBATCH --output=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/baselinestd_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/baselinestd_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH -C v100-32g
#SBATCH --time=04:00:00             # ~2h pour les configs les plus longues, marge x2
#SBATCH --account=ndg@v100
#SBATCH --array=0-49%50             # 50 configs, 50 GPU en parallele

# --- Avantage (f(x) - b_i^t) / std^t (peragentbaselineglobalstd) autour d'EvoCOP ---
# Une tache = une config de jeanzay/grids/baselinestd_evocop.json, sur les instances
# test (56 classes, rang contre le pool). Resultats dans results/config/<config>/.
# Relancable : les instances deja faites sont sautees.
#
# Usage :
#   sbatch jeanzay/baselinestd_v100.slurm
#   sbatch --array=3,17 jeanzay/baselinestd_v100.slurm      # relancer des taches precises
#   sbatch --qos=qos_gpu-dev --time=00:30:00 --array=0 jeanzay/baselinestd_v100.slurm   # test rapide

echo "Job démarré sur $(hostname) à $(date)"

cd $WORK/Supplementary-material
mkdir -p outputs

module purge
module load miniforge/24.9.0
conda activate ppo_env

export PYTHONUNBUFFERED=1

NB_CONFIGS=50
echo "Tache ${SLURM_ARRAY_TASK_ID} : config ${SLURM_ARRAY_TASK_ID} de baselinestd_evocop.json"

srun python source_code/expe/main_expe_overall.py \
  --instances test \
  -g jeanzay/grids/baselinestd_evocop.json \
  --shard "${SLURM_ARRAY_TASK_ID}/${NB_CONFIGS}"

echo "Job terminé à $(date)"
