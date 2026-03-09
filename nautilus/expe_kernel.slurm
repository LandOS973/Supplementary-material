#!/bin/bash
#SBATCH --job-name=expe
#SBATCH --qos=short
#SBATCH -p gpu
#SBATCH --output=/home/t.lan@univ-angers.fr/Supplementary-material/outputs/expe_%j.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --mail-user=thomas.landais@univ-angers.fr
#SBATCH --mail-type=END,FAIL

OUTFILE="/home/t.lan@univ-angers.fr/Supplementary-material/outputs/expe_${SLURM_JOB_ID}.out"

echo "Job démarré sur $(hostname) à $(date)"
echo "Run main_expe"

M_VALUES=""
while getopts "m:" opt; do
  case "$opt" in
    m) M_VALUES="$OPTARG" ;;
  esac
done
shift $((OPTIND - 1))

JOB_NAME="expe"
if [ -n "$M_VALUES" ]; then
  JOB_SUFFIX="${M_VALUES//,/_}"
  JOB_SUFFIX="${JOB_SUFFIX// /_}"
  JOB_NAME="expe_${JOB_SUFFIX}"
  scontrol update JobName="$JOB_NAME" JobId="$SLURM_JOB_ID" >/dev/null 2>&1 || true
fi

echo "Job name: $JOB_NAME"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON=/micromamba/$USER/envs/ppo_env/bin/python

if [ -n "$M_VALUES" ]; then
  srun $PYTHON -B /home/t.lan@univ-angers.fr/Supplementary-material/source_code/main_expe_overall.py -m "$M_VALUES"
else
  srun $PYTHON -B /home/t.lan@univ-angers.fr/Supplementary-material/source_code/main_expe_overall.py
fi
echo "Job terminé à $(date)"
echo "Notification mail Slurm activée."

# cmd pour lancer : sbatch expe_kernel.slurm -m 10,15,20,25,30