#!/bin/bash
#SBATCH --job-name=edas_qubo512
#SBATCH --output=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/edas_qubo512_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/edas_qubo512_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00             # a recalibrer, jamais mesure pour PBIL/MIMIC/BOA a n=512
#SBATCH --partition=cpu_p1
#SBATCH --account=ndg@cpu
#SBATCH --array=0-179%100           # 3 algos (PBIL,MIMIC,BOA) x 6 t x 10 seeds (0..179)

# --- PBIL/MIMIC/BOA sur QUBO n=512, meme granularite (algo,t,seed) que nevergrad ---
# main_baseline_edas_and_tabu.py n'a pas de --nb_restarts interne (contrairement a
# PEDA/PPBIL), donc un appel = un seed. 3 algos x 6 t x 10 seeds = 180 taches,
# chacune traitant les 10 instances de son (t) en un appel.
#
# Path fix applique (meme piege que main_nevergrad.py avant correction) : chemins
# absolus, sortie dans results/nevergrad/<algo>/QUBO/512/<t>/... (au lieu de
# results/results_EDAs_final/ relatif au CWD).
#
# Usage :
#   sbatch jeanzay/expe_edas_qubo512.slurm
#   sbatch --qos=qos_cpu-dev --time=00:10:00 --array=0 jeanzay/expe_edas_qubo512.slurm -b 500

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
while getopts "d:b:n:" opt; do
  case "$opt" in
    d) DIM="$OPTARG" ;;
    b) BUDGET="$OPTARG" ;;
    n) NB_INSTANCES="$OPTARG" ;;
  esac
done

# Decodage (algo, t, seed) : 3 algos x 6 t x 10 seeds
ALGOS=(PBIL MIMIC BOA)
NB_T=6
NB_SEEDS=10
COMBOS_PER_ALGO=$((NB_T * NB_SEEDS))

ALGO_IDX=$((SLURM_ARRAY_TASK_ID / COMBOS_PER_ALGO))
REMAINDER=$((SLURM_ARRAY_TASK_ID % COMBOS_PER_ALGO))
T=$((REMAINDER / NB_SEEDS))
SEED=$((REMAINDER % NB_SEEDS))

ALGO="${ALGOS[$ALGO_IDX]}"
if [ -z "$ALGO" ]; then
  echo "Pas d'algo a l'index ${ALGO_IDX} (tache ${SLURM_ARRAY_TASK_ID})" >&2
  exit 1
fi

echo "Algo=[$ALGO]  Dim=$DIM  T=$T  Seed=$SEED  Budget=$BUDGET  NbInstances=$NB_INSTANCES"

# Skip si deja fait (meme logique que expe_nevergrad_qubo512.slurm)
RESULT_DIR="results/nevergrad/${ALGO}/QUBO/${DIM}/${T}"
EXISTING=$(ls -t "${RESULT_DIR}"/results_EDAs_final_${ALGO}_QUBO_${DIM}_${T}_${NB_INSTANCES}_budget_${BUDGET}_*_${SEED}.txt 2>/dev/null | head -1)

if [ -n "$EXISTING" ] && [ "$(wc -l < "$EXISTING")" -gt 1 ]; then
  echo "  -> deja fait, on saute : $EXISTING"
else
  python source_code/main_baseline_edas_and_tabu.py QUBO "$ALGO" "$DIM" "$T" \
    --nb_instances "$NB_INSTANCES" \
    --budget "$BUDGET" \
    --seed "$SEED"
fi

echo "Tache ${SLURM_ARRAY_TASK_ID} terminee a $(date)"
