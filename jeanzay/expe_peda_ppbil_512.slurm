#!/bin/bash
#SBATCH --job-name=peda_ppbil512
#SBATCH --output=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/peda_ppbil512_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/peda_ppbil512_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00            # a recalibrer apres le premier run (pas encore mesure sur GPU)
#SBATCH --account=ndg@v100
#SBATCH --array=0-4                # 5 combos : PEDA x {QUBO,NK,NK3} + PPBIL x {QUBO,NK}

# --- PEDA et PPBIL sur n=512, GPU (V100), pour completer le pool de concurrents ---
# Chaque tache = un (script, type_problem), balaie tous les K/t disponibles pour
# dim=512 en un seul appel (les scripts gerent nb_instances x nb_restarts en interne,
# pas besoin de splitter par seed comme pour nevergrad).
#
# main_expe_peda.py / main_expe_ppbil.py gerent deja REPO_ROOT correctement
# (os.chdir(SCRIPT_DIR) + chemins absolus), aucun fix de chemin necessaire.
#
# PPBIL ne supporte pas NK3 (binaire uniquement, cf. docstring du script).
#
# Sortie : results/nevergrad/{PEDA,PPBIL}/{QUBO,NK,NK3}/512/<t>/..._budget_<b>_..._i_<i>_r_<r>.txt
#
# Usage :
#   sbatch jeanzay/expe_peda_ppbil_512.slurm
#   sbatch --array=0 jeanzay/expe_peda_ppbil_512.slurm   # test rapide, un seul combo

echo "Job ${SLURM_ARRAY_JOB_ID} tache ${SLURM_ARRAY_TASK_ID} sur $(hostname) a $(date)"

cd $WORK/Supplementary-material
mkdir -p outputs

module purge
module load miniforge/24.9.0
conda activate ppo_env

export PYTHONUNBUFFERED=1

DIM=512
BUDGET=50000
NB_INSTANCES=10
NB_RESTARTS=10

# (script, type_problem) pour chaque indice de tache
COMBOS=(
  "main_expe_peda.py QUBO"
  "main_expe_peda.py NK"
  "main_expe_peda.py NK3"
  "main_expe_ppbil.py QUBO"
  "main_expe_ppbil.py NK"
)

COMBO="${COMBOS[$SLURM_ARRAY_TASK_ID]}"
if [ -z "$COMBO" ]; then
  echo "Pas de combo a l'index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

read -r SCRIPT TYPE_PROBLEM <<< "$COMBO"

echo "Script=$SCRIPT  TypeProblem=$TYPE_PROBLEM  Dim=$DIM  Budget=$BUDGET"

cd source_code
python "$SCRIPT" "$TYPE_PROBLEM" "$DIM" \
  --budget "$BUDGET" \
  --nb-instances "$NB_INSTANCES" \
  --nb-restarts "$NB_RESTARTS"

echo "Tache ${SLURM_ARRAY_TASK_ID} terminee a $(date)"
