#!/bin/bash
#SBATCH --job-name=main
#SBATCH --output=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/main_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/main_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --account=ndg@v100

OUTFILE="/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/main_${SLURM_JOB_ID}.out"

echo "Job démarré sur $(hostname) à $(date)"

M_VALUES=""
while getopts "m:" opt; do
  case "$opt" in
    m) M_VALUES="$OPTARG" ;;
  esac
done
shift $((OPTIND - 1))

JOB_NAME="main"
if [ -n "$M_VALUES" ]; then
  JOB_SUFFIX="${M_VALUES//,/_}"
  JOB_SUFFIX="${JOB_SUFFIX// /_}"
  JOB_NAME="main_${JOB_SUFFIX}"
  scontrol update JobName="$JOB_NAME" JobId="$SLURM_JOB_ID" >/dev/null 2>&1 || true
fi

echo "Job name: $JOB_NAME"

cd $WORK/Supplementary-material
mkdir -p outputs

module purge
module load miniforge/24.9.0
conda activate ppo_env

export PYTHONUNBUFFERED=1

if [ -n "$M_VALUES" ]; then
  srun python source_code/main.py -m "$M_VALUES" "$@"
else
  srun python source_code/main.py "$@"
fi

echo "Job terminé à $(date)"

# Exemples de lancement :
# sbatch jeanzay/main.slurm                                    # config par défaut
# sbatch jeanzay/main.slurm budget=1000 verbose=false          # overrides Hydra
# sbatch jeanzay/main.slurm -m "seed=1,2,3 budget=1000"       # Hydra multirun

#tail -n 100 -f $WORK/Supplementary-material/outputs/main_numjob.out
