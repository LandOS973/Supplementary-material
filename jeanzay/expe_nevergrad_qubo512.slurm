#!/bin/bash
#SBATCH --job-name=ng_qubo512
#SBATCH --output=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/ng_qubo512_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/ng_qubo512_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1            # nevergrad est mono-thread
#SBATCH --hint=nomultithread        # OBLIGATOIRE sur JZ
#SBATCH --time=06:00:00             # marge x2 sur le pire cas observe (AdaptiveDiscreteOnePlusOne ~2h45/seed)
#SBATCH --partition=cpu_p1          # noeuds CPU Cascade Lake (alloc "Jean Zay CSL")
#SBATCH --account=ndg@cpu           # /!\ compte CPU du projet ndg -- verifier avec `idracct ndg`
#SBATCH --array=0-4859%200          # 81 algos x 6 t x 10 seeds (0..4859). %200 = max 200 taches en meme temps (politesse cluster).

# --- Une tache = un (algo, t, seed) sur QUBO n=512 : un seul appel main_nevergrad.py ---
# 81 algos x 6 types d'instance (t=0..5) x 10 seeds = 4860 taches, chacune faisant
# UN SEUL appel (10 instances) -> 4860x10 = 48600 runs au total, le protocole complet
# du papier (10 instances x 10 restarts par config).
#
# Pourquoi ce decoupage (v2) : la v1 (1 tache = 1 (algo,t), boucle interne sur 10 seeds,
# --time=12h) a donne 312/486 taches en TIMEOUT. Calibrage initial fait sur DiscreteDE
# seul (~55min/appel) ne generalisait pas : AdaptiveDiscreteOnePlusOne mesure ~2h30-2h45
# par appel sur JZ (jusqu'a 3x plus lent), donc 10 seeds dans une tache peut depasser
# largement 12h selon l'algo. Decouper au niveau du seed rend chaque tache courte
# independamment de la variance inter-algo, sans reduire le volume de calcul.
#
# main_nevergrad.py gere nativement QUBO/NK/NK3, aucun script "_batch" requis ici
# (contrairement a ViennaRNA qui avait besoin d'un runner dedie).
#
# Sortie : results/nevergrad/<algo>/QUBO/512/<t>/..._i_<i>_budget_<b>_..._<seed>.txt
#
# Usage :
#   sbatch jeanzay/expe_nevergrad_qubo512.slurm
#   sbatch jeanzay/expe_nevergrad_qubo512.slurm -b 50000 -n 10 -d 512
#
# Options (getopts) :
#   -d  dimension                   (defaut : 512)
#   -b  budget (evaluations)        (defaut : 50000)
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

# 4. Decodage (algo, t, seed) a partir de l'index de tache : 81 algos x 6 t x 10 seeds
NB_T=6
NB_SEEDS=10
COMBOS_PER_ALGO=$((NB_T * NB_SEEDS))

ALGO_IDX=$((SLURM_ARRAY_TASK_ID / COMBOS_PER_ALGO))
REMAINDER=$((SLURM_ARRAY_TASK_ID % COMBOS_PER_ALGO))
T=$((REMAINDER / NB_SEEDS))
SEED=$((REMAINDER % NB_SEEDS))

ALGO_FILE="jeanzay/nevergrad_discrete_algos.txt"
readarray -t ALGOS < "$ALGO_FILE"
ALGO="${ALGOS[$ALGO_IDX]}"

if [ -z "${ALGO:-}" ]; then
  echo "Pas d'algo a l'index ${ALGO_IDX} (tache ${SLURM_ARRAY_TASK_ID}) dans $ALGO_FILE" >&2
  exit 1
fi

echo "Algo=[$ALGO]  Dim=$DIM  T=$T  Seed=$SEED  Budget=$BUDGET  NbInstances=$NB_INSTANCES"

# 5. Skip si ce (algo,t,seed) a deja un resultat COMPLET (evite de refaire les 3223 runs
#    deja produits par la v1 avant son timeout -- meme budget, meme seed, peu importe le timestamp).
#    Attention : un fichier tue en plein calcul ne contient que l'en-tete (1 ligne) -- on
#    verifie donc aussi le nombre de lignes, pas juste l'existence du fichier.
RESULT_DIR="results/nevergrad/${ALGO}/QUBO/${DIM}/${T}"
EXISTING=$(ls -t "${RESULT_DIR}"/results_nevergrad_${ALGO}_QUBO_${DIM}_${T}_${NB_INSTANCES}_budget_${BUDGET}_*_${SEED}.txt 2>/dev/null | head -1)

if [ -n "$EXISTING" ] && [ "$(wc -l < "$EXISTING")" -gt 1 ]; then
  echo "  -> deja fait, on saute : $EXISTING"
else
  python source_code/main_nevergrad.py QUBO "$ALGO" "$DIM" "$T" \
    --nb_instances "$NB_INSTANCES" \
    --budget "$BUDGET" \
    --seed "$SEED"
fi

echo "Tache ${SLURM_ARRAY_TASK_ID} terminee a $(date)"
