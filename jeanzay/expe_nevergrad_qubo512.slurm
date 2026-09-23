#!/bin/bash
#SBATCH --job-name=ng_qubo512
#SBATCH --output=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/ng_qubo512_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/ng_qubo512_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1            # nevergrad est mono-thread
#SBATCH --hint=nomultithread        # OBLIGATOIRE sur JZ
#SBATCH --time=12:00:00             # calibre : ~9.2h/tache (10 seeds x ~55min), marge incluse
#SBATCH --partition=cpu_p1          # noeuds CPU Cascade Lake (alloc "Jean Zay CSL")
#SBATCH --account=ndg@cpu           # /!\ compte CPU du projet ndg -- verifier avec `idracct ndg`
#SBATCH --array=0-485               # 81 algos x 6 t (0..485). Ajouter %60 pour limiter la concurrence.

# --- Une tache = un (algo, t) sur QUBO n=512 : boucle interne sur les 10 seeds ---
# 81 algos x 6 types d'instance (t=0..5) = 486 taches, chacune faisant
# RESTARTS appels (defaut 10) = 10 seeds x 10 instances = 100 runs par (algo, t),
# le protocole complet du papier (10 instances x 10 restarts).
#
# Pourquoi ce decoupage : calibrage local (2 points, budget 2000 et 8000 evals)
# donne un cout ~6.65ms/eval et un overhead de chargement negligeable (~2.8s/appel).
# A budget=50000 x 10 instances, un appel ~55min -> 10 seeds ~9.2h/tache.
# Faire les 6 t dans UNE tache (comme la version precedente) aurait demande ~56h/tache,
# infaisable meme avec qos_cpu-t4. Decouper par (algo,t) rend chaque tache courte
# sans rien retirer : 81x6x10x10 = 48600 runs au total, identique dans les deux cas.
#
# main_nevergrad.py gere nativement QUBO/NK/NK3, aucun script "_batch" requis ici
# (contrairement a ViennaRNA qui avait besoin d'un runner dedie).
#
# Sortie : results/nevergrad/<algo>/QUBO/512/<t>/..._i_<i>_budget_<b>_..._<seed>.txt
#
# Usage :
#   sbatch jeanzay/expe_nevergrad_qubo512.slurm
#   sbatch jeanzay/expe_nevergrad_qubo512.slurm -b 50000 -n 10 -r 10 -d 512
#
# Options (getopts) :
#   -d  dimension                   (defaut : 512)
#   -b  budget (evaluations)        (defaut : 50000)
#   -n  nb instances par t          (defaut : 10, doit matcher les fichiers generes)
#   -r  nombre de restarts (seeds)  (defaut : 10, pour retrouver 10x10=100 runs)

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
RESTARTS=10
while getopts "d:b:n:r:" opt; do
  case "$opt" in
    d) DIM="$OPTARG" ;;
    b) BUDGET="$OPTARG" ;;
    n) NB_INSTANCES="$OPTARG" ;;
    r) RESTARTS="$OPTARG" ;;
  esac
done

# 4. Decodage (algo, t) a partir de l'index de tache : 81 algos x 6 t
NB_T=6
ALGO_IDX=$((SLURM_ARRAY_TASK_ID / NB_T))
T=$((SLURM_ARRAY_TASK_ID % NB_T))

ALGO_FILE="jeanzay/nevergrad_discrete_algos.txt"
readarray -t ALGOS < "$ALGO_FILE"
ALGO="${ALGOS[$ALGO_IDX]}"

if [ -z "${ALGO:-}" ]; then
  echo "Pas d'algo a l'index ${ALGO_IDX} (tache ${SLURM_ARRAY_TASK_ID}) dans $ALGO_FILE" >&2
  exit 1
fi

echo "Algo=[$ALGO]  Dim=$DIM  T=$T  Budget=$BUDGET  NbInstances=$NB_INSTANCES  Restarts=$RESTARTS"

# 5. Boucle interne sur les seeds (main_nevergrad.py fait deja les NB_INSTANCES instances
#    en une seule fois pour un (algo, t, seed) donne -> RESTARTS appels = RESTARTS x NB_INSTANCES runs)
for SEED in $(seq 0 $((RESTARTS - 1))); do
  echo "  -> seed=$SEED"
  python source_code/main_nevergrad.py QUBO "$ALGO" "$DIM" "$T" \
    --nb_instances "$NB_INSTANCES" \
    --budget "$BUDGET" \
    --seed "$SEED"
done

echo "Tache ${SLURM_ARRAY_TASK_ID} terminee a $(date)"
