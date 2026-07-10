#!/bin/bash
#SBATCH --job-name=ng_vrna
#SBATCH --output=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/ng_vrna_%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/ng_vrna_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1            # nevergrad + ViennaRNA (pf) sont mono-thread
#SBATCH --hint=nomultithread        # OBLIGATOIRE sur JZ
#SBATCH --time=40:00:00             # large : cibles longues (~300 nt) ~13h/tache, marge pour ~400 nt
#SBATCH --qos=qos_cpu-t4            # OBLIGATOIRE pour depasser 20h (defaut qos_cpu-t3 = 20h max)
#SBATCH --partition=cpu_p1          # noeuds CPU Cascade Lake (alloc "Jean Zay CSL")
#SBATCH --account=ndg@cpu           # /!\ compte CPU du projet ndg -- verifier avec `idracct ndg`
#SBATCH --array=0-80                # 81 algos discrets (0..80). Ajouter %20 pour limiter la concurrence.

# --- Un algo Nevergrad discret par tache du job array, sur une cible ViennaRNA ---
# Chaque tache fait `nb_restarts` runs et ecrit un fichier par run dans
#   results/nevergrad/<algo>/VIENNARNA/<target>/..._i_0_r_<r>.txt
#
# Usage :
#   sbatch jeanzay/expe_nevergrad_viennarna.slurm
#   sbatch jeanzay/expe_nevergrad_viennarna.slurm -t "Kudzu" -b 50000 -r 10
#
# Options (getopts) :
#   -t  nom de la cible Eterna100   (defaut : "Simple Hairpin")
#   -b  budget (evaluations)        (defaut : 50000)
#   -r  nombre de restarts          (defaut : 10)
#   -s  seed de base                (defaut : 0)

echo "Job ${SLURM_ARRAY_JOB_ID} tache ${SLURM_ARRAY_TASK_ID} sur $(hostname) a $(date)"

# 1. Racine du projet
cd $WORK/Supplementary-material
mkdir -p outputs

# 2. Environnement conda
module purge
module load miniforge/24.9.0
conda activate ppo_env

export PYTHONUNBUFFERED=1

# 3. Parametres
TARGET="Simple Hairpin"
BUDGET=50000
RESTARTS=10
SEED=0
while getopts "t:b:r:s:" opt; do
  case "$opt" in
    t) TARGET="$OPTARG" ;;
    b) BUDGET="$OPTARG" ;;
    r) RESTARTS="$OPTARG" ;;
    s) SEED="$OPTARG" ;;
  esac
done

# 4. Selection de l'algo pour cette tache du array
ALGO_FILE="jeanzay/nevergrad_discrete_algos.txt"
readarray -t ALGOS < "$ALGO_FILE"
ALGO="${ALGOS[$SLURM_ARRAY_TASK_ID]}"

if [ -z "${ALGO:-}" ]; then
  echo "Pas d'algo a l'index ${SLURM_ARRAY_TASK_ID} dans $ALGO_FILE" >&2
  exit 1
fi

echo "Algo=[$ALGO]  Target=[$TARGET]  Budget=$BUDGET  Restarts=$RESTARTS  Seed=$SEED"

# 5. Lancement (srun pour que Slurm gere les ressources)
srun python source_code/main_nevergrad_viennarna_batch.py "$ALGO" \
  --target "$TARGET" \
  --budget "$BUDGET" \
  --nb_restarts "$RESTARTS" \
  --seed "$SEED"

echo "Tache ${SLURM_ARRAY_TASK_ID} terminee a $(date)"
