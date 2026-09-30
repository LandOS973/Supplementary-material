#!/bin/bash
#SBATCH --job-name=edas_512
#SBATCH --output=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/edas_512_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/edas_512_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00             # a recalibrer, jamais mesure pour PBIL a n=512
#SBATCH --partition=cpu_p1
#SBATCH --account=ndg@cpu
#SBATCH --array=0-59%100            # QUBO par defaut (PBIL x 6 t x 10). NK/NK3 : 1x4x10=0-39.

# --- PBIL a n=512, meme granularite (algo,t,seed) que nevergrad ---
# main_baseline_edas_and_tabu.py n'a pas de --nb_restarts interne (contrairement a
# PEDA/PPBIL), donc un appel = un seed. QUBO a 6 types d'instance (t=0..5), NK/NK3
# en ont 4 (K=1,2,4,8) -> NB_T s'adapte a -p.
#
# Path fix applique (meme piege que main_nevergrad.py avant correction) : chemins
# absolus, sortie dans results/nevergrad/<algo>/<TYPE_PROBLEM>/512/<t>/... (au lieu
# de results/results_EDAs_final/ relatif au CWD).
#
# MIMIC et BOA EXCLUS a n=512 (QUBO/NK/NK3) : leurs deux implementations
# construisent une structure de dependance par paire de variables, cout O(n^2)
# par construction (calc_bi_frequency dans mimic.py, k2_algorithm dans boa.py).
# A n=512 ca ne termine jamais une seule generation dans un temps raisonnable
# (verifie sur JZ : job budget=500 toujours RUNNING apres plusieurs minutes,
# fichiers resultat reduits a l'en-tete). PBIL est un modele univarie, O(n),
# non concerne. Voir paper/outline.md pour la justification complete.
#
# PBIL supporte NK3 dans ce depot (implementation generique, categories geree
# par D) -- contrairement a PPBIL (voir expe_peda_ppbil_512.slurm) qui est
# binaire seulement. Verifie : results/nevergrad/PBIL/NK3/{64,128,256} existent deja.
#
# Usage :
#   sbatch jeanzay/expe_edas_512.slurm                            # QUBO (defaut)
#   sbatch --array=0-39%100 jeanzay/expe_edas_512.slurm -p NK     # NK  : 1x4x10=40
#   sbatch --array=0-39%100 jeanzay/expe_edas_512.slurm -p NK3    # NK3 : 1x4x10=40
#   sbatch --qos=qos_cpu-dev --time=00:10:00 --array=0 jeanzay/expe_edas_512.slurm -b 500
#
# Options (getopts) :
#   -p  type_problem (QUBO|NK|NK3)   (defaut : QUBO)
#   -d  dimension                    (defaut : 512)
#   -b  budget (evaluations)         (defaut : 50000)
#   -n  nb instances                 (defaut : 10)

echo "Job ${SLURM_ARRAY_JOB_ID} tache ${SLURM_ARRAY_TASK_ID} sur $(hostname) a $(date)"

cd $WORK/Supplementary-material
mkdir -p outputs

module purge
module load miniforge/24.9.0
conda activate ppo_env

export PYTHONUNBUFFERED=1

TYPE_PROBLEM=QUBO
DIM=512
BUDGET=50000
NB_INSTANCES=10
while getopts "p:d:b:n:" opt; do
  case "$opt" in
    p) TYPE_PROBLEM="$OPTARG" ;;
    d) DIM="$OPTARG" ;;
    b) BUDGET="$OPTARG" ;;
    n) NB_INSTANCES="$OPTARG" ;;
  esac
done

if [ "$TYPE_PROBLEM" = "QUBO" ]; then
  NB_T=6
else
  NB_T=4
fi

# Decodage (algo, t_idx, seed) -- MIMIC/BOA exclus a n=512, voir commentaire plus haut
ALGOS=(PBIL)
NB_SEEDS=10
COMBOS_PER_ALGO=$((NB_T * NB_SEEDS))

ALGO_IDX=$((SLURM_ARRAY_TASK_ID / COMBOS_PER_ALGO))
REMAINDER=$((SLURM_ARRAY_TASK_ID % COMBOS_PER_ALGO))
T_IDX=$((REMAINDER / NB_SEEDS))
SEED=$((REMAINDER % NB_SEEDS))

ALGO="${ALGOS[$ALGO_IDX]}"
if [ -z "$ALGO" ]; then
  echo "Pas d'algo a l'index ${ALGO_IDX} (tache ${SLURM_ARRAY_TASK_ID})" >&2
  exit 1
fi

if [ "$TYPE_PROBLEM" = "QUBO" ]; then
  T=$T_IDX
else
  K_VALUES=(1 2 4 8)
  T=${K_VALUES[$T_IDX]}
fi

echo "Algo=[$ALGO]  Pb=$TYPE_PROBLEM  Dim=$DIM  T=$T  Seed=$SEED  Budget=$BUDGET  NbInstances=$NB_INSTANCES"

# Skip si deja fait
RESULT_DIR="results/nevergrad/${ALGO}/${TYPE_PROBLEM}/${DIM}/${T}"
EXISTING=$(ls -t "${RESULT_DIR}"/results_EDAs_final_${ALGO}_${TYPE_PROBLEM}_${DIM}_${T}_${NB_INSTANCES}_budget_${BUDGET}_*_${SEED}.txt 2>/dev/null | head -1)

if [ -n "$EXISTING" ] && [ "$(wc -l < "$EXISTING")" -gt 1 ]; then
  echo "  -> deja fait, on saute : $EXISTING"
else
  python source_code/main_baseline_edas_and_tabu.py "$TYPE_PROBLEM" "$ALGO" "$DIM" "$T" \
    --nb_instances "$NB_INSTANCES" \
    --budget "$BUDGET" \
    --seed "$SEED"
fi

echo "Tache ${SLURM_ARRAY_TASK_ID} terminee a $(date)"
