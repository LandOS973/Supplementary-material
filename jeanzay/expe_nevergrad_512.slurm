#!/bin/bash
#SBATCH --job-name=ng_512
#SBATCH --output=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/ng_512_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/ng_512_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1            # nevergrad est mono-thread
#SBATCH --hint=nomultithread        # OBLIGATOIRE sur JZ
#SBATCH --time=06:00:00             # marge x2 sur le pire cas observe (AdaptiveDiscreteOnePlusOne ~2h45/seed)
#SBATCH --partition=cpu_p1          # noeuds CPU Cascade Lake (alloc "Jean Zay CSL")
#SBATCH --account=ndg@cpu           # /!\ compte CPU du projet ndg -- verifier avec `idracct ndg`
#SBATCH --array=0-4859%200          # QUBO par defaut (81 x 6 x 10). Voir Usage pour NK/NK3 (81 x 4 x 10 = 0-3239).

# --- Une tache = un (algo, t, seed) a n=512 : un seul appel main_nevergrad.py ---
# main_nevergrad.py gere nativement QUBO/NK/NK3. QUBO a 6 types d'instance (t=0..5),
# NK/NK3 en ont 4 (K=1,2,4,8) -> NB_T ci-dessous s'adapte a -p.
# 81 algos x NB_T x 10 seeds taches, chacune faisant UN SEUL appel (10 instances).
#
# DiscreteNoisyInfSplits est saute automatiquement (algo pathologiquement lent a
# n=512, ~15-20x les autres algos, deja constate sur QUBO -- voir paper/outline.md).
#
# Pourquoi ce decoupage : calibrage initial sur un seul algo ne generalise pas,
# certains algos sont jusqu'a 3x plus lents que d'autres -> decouper au niveau du
# seed rend chaque tache courte independamment de la variance inter-algo.
#
# Sortie : results/nevergrad/<algo>/<TYPE_PROBLEM>/512/<t>/..._i_<i>_budget_<b>_..._<seed>.txt
#
# Usage :
#   sbatch jeanzay/expe_nevergrad_512.slurm                              # QUBO (defaut)
#   sbatch --array=0-3239%200 jeanzay/expe_nevergrad_512.slurm -p NK     # NK  : 81x4x10=3240
#   sbatch --array=0-3239%200 jeanzay/expe_nevergrad_512.slurm -p NK3    # NK3 : 81x4x10=3240
#
# Options (getopts) :
#   -p  type_problem (QUBO|NK|NK3)   (defaut : QUBO)
#   -d  dimension                    (defaut : 512)
#   -b  budget (evaluations)         (defaut : 50000)
#   -f  force : relance meme si un resultat existe deja (ex: pour regenerer
#       final_scores_seed*.csv sur le meilleur concurrent)
#   -n  nb instances par (algo,t,seed) (defaut : 10, doit matcher les fichiers generes)

echo "Job ${SLURM_ARRAY_JOB_ID} tache ${SLURM_ARRAY_TASK_ID} sur $(hostname) a $(date)"

# 1. Racine du projet
cd $WORK/Supplementary-material
mkdir -p outputs

# 2. Environnement conda (voir memoire jeanzay-ppo-env-setup pour les pieges connus :
#    ViennaRNA absent du ppo_env d'origine, bayes_opt trop recent casse nevergrad,
#    ne JAMAIS mettre `set -euo pipefail` ici -> casse conda activate)
module purge
module load miniforge/24.9.0
conda activate ppo_env

export PYTHONUNBUFFERED=1

# 3. Parametres
TYPE_PROBLEM=QUBO
DIM=512
BUDGET=50000
NB_INSTANCES=10
FORCE=0
while getopts "p:d:b:n:f" opt; do
  case "$opt" in
    p) TYPE_PROBLEM="$OPTARG" ;;
    d) DIM="$OPTARG" ;;
    b) BUDGET="$OPTARG" ;;
    n) NB_INSTANCES="$OPTARG" ;;
    f) FORCE=1 ;;
  esac
done

if [ "$TYPE_PROBLEM" = "QUBO" ]; then
  NB_T=6
else
  NB_T=4
fi
NB_SEEDS=10
COMBOS_PER_ALGO=$((NB_T * NB_SEEDS))

# 4. Decodage (algo, t_idx, seed) a partir de l'index de tache
ALGO_IDX=$((SLURM_ARRAY_TASK_ID / COMBOS_PER_ALGO))
REMAINDER=$((SLURM_ARRAY_TASK_ID % COMBOS_PER_ALGO))
T_IDX=$((REMAINDER / NB_SEEDS))
SEED=$((REMAINDER % NB_SEEDS))

ALGO_FILE="jeanzay/nevergrad_discrete_algos.txt"
readarray -t ALGOS < "$ALGO_FILE"
ALGO="${ALGOS[$ALGO_IDX]}"

if [ -z "${ALGO:-}" ]; then
  echo "Pas d'algo a l'index ${ALGO_IDX} (tache ${SLURM_ARRAY_TASK_ID}) dans $ALGO_FILE" >&2
  exit 1
fi

if [ "$ALGO" = "DiscreteNoisyInfSplits" ]; then
  echo "Algo=[$ALGO] : exclu (pathologiquement lent a n=512), on saute."
  exit 0
fi

# T (type_instance) : pour QUBO t=T_IDX directement (0..5). Pour NK/NK3, K in {1,2,4,8}.
if [ "$TYPE_PROBLEM" = "QUBO" ]; then
  T=$T_IDX
else
  K_VALUES=(1 2 4 8)
  T=${K_VALUES[$T_IDX]}
fi

echo "Algo=[$ALGO]  Pb=$TYPE_PROBLEM  Dim=$DIM  T=$T  Seed=$SEED  Budget=$BUDGET  NbInstances=$NB_INSTANCES"

# 5. Skip si ce (algo,t,seed) a deja un resultat COMPLET (reprise apres timeout/relance).
#    Un fichier tue en plein calcul ne contient que l'en-tete (1 ligne) -- on verifie
#    donc aussi le nombre de lignes, pas juste l'existence du fichier.
RESULT_DIR="results/nevergrad/${ALGO}/${TYPE_PROBLEM}/${DIM}/${T}"
EXISTING=$(ls -t "${RESULT_DIR}"/results_nevergrad_${ALGO}_${TYPE_PROBLEM}_${DIM}_${T}_${NB_INSTANCES}_budget_${BUDGET}_*_${SEED}.txt 2>/dev/null | head -1)

if [ "$FORCE" -eq 0 ] && [ -n "$EXISTING" ] && [ "$(wc -l < "$EXISTING")" -gt 1 ]; then
  echo "  -> deja fait, on saute : $EXISTING"
else
  python source_code/main_nevergrad.py "$TYPE_PROBLEM" "$ALGO" "$DIM" "$T" \
    --nb_instances "$NB_INSTANCES" \
    --budget "$BUDGET" \
    --seed "$SEED"
fi

echo "Tache ${SLURM_ARRAY_TASK_ID} terminee a $(date)"
