#!/bin/bash
#SBATCH --job-name=nasbench_jz
#SBATCH --output=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/nasbench_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/nasbench_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --account=ndg@v100

echo "Job démarré sur $(hostname) à $(date)"

cd "$WORK/Supplementary-material"
mkdir -p outputs

module purge
module load miniforge/24.9.0
ENV_NAME="${ENV_NAME:-nasbench_env}"
conda activate "$ENV_NAME"

export PYTHONUNBUFFERED=1

DEFAULT_DATA="$WORK/Supplementary-material/source_code/instances/nasbench/nasbench_full.tfrecord"
DATA_FILE=""
if [ -f "$DEFAULT_DATA" ]; then
  DATA_FILE="$DEFAULT_DATA"
fi

JOB_NAME="nasbench_jz"
if [ -n "$DATA_FILE" ]; then
  JOB_SUFFIX="$(basename "$DATA_FILE")"
  JOB_SUFFIX="${JOB_SUFFIX//[^a-zA-Z0-9]/_}"
  JOB_NAME="nasbench_${JOB_SUFFIX}"
  scontrol update JobName="$JOB_NAME" JobId="$SLURM_JOB_ID" >/dev/null 2>&1 || true
fi

echo "Job name: $JOB_NAME"

ARGS=(--verbose)
if [ -n "$DATA_FILE" ]; then ARGS+=(--data_file "$DATA_FILE"); fi

echo "Lancement de main_expe_overall_nasbench.py..."
srun python source_code/main_expe_overall_nasbench.py "${ARGS[@]}"

echo "Job terminé à $(date)"
