#!/bin/bash
#SBATCH --job-name=peda_ppbil512
#SBATCH --output=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/peda_ppbil512_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/peda_ppbil512_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00             # non mesure sur GPU pour ces scripts, a recalibrer apres coup
#SBATCH --account=ndg@v100

# --- PEDA et PPBIL a n=512, GPU (V100), un seul job sequentiel (pas d'array) ---
# Chaque script gere nb_instances x nb_restarts en interne et balaie tous les t
# d'un coup (type_instance omis) -> 1 appel par algo suffit a couvrir tous les t.
# Pas de --no-gpu : utilise PEDAGpu/PPBILGpu automatiquement.
#
# PPBIL est binaire seulement (D=2) -- ne supporte PAS NK3 (D=3). Pour
# TYPE_PROBLEM=NK3, seul PEDA tourne (PPBIL est saute).
#
# main_expe_peda.py / main_expe_ppbil.py gerent deja REPO_ROOT correctement
# (os.chdir(SCRIPT_DIR) + chemins absolus), aucun fix de chemin necessaire.
#
# Sortie : results/nevergrad/{PEDA,PPBIL}/<TYPE_PROBLEM>/512/<t>/..._budget_<b>_..._i_<i>_r_<r>.txt
#
# Usage :
#   sbatch jeanzay/expe_peda_ppbil_512.slurm                    # QUBO (defaut) : PEDA + PPBIL
#   sbatch jeanzay/expe_peda_ppbil_512.slurm -p NK               # NK           : PEDA + PPBIL
#   sbatch jeanzay/expe_peda_ppbil_512.slurm -p NK3              # NK3          : PEDA seul
#
# Options (getopts) :
#   -p  type_problem (QUBO|NK|NK3)   (defaut : QUBO)
#   -d  dimension                    (defaut : 512)
#   -b  budget (evaluations)         (defaut : 50000)
#   -n  nb instances                 (defaut : 10)
#   -r  nb restarts                  (defaut : 10)

echo "Job ${SLURM_JOB_ID} sur $(hostname) a $(date)"

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
NB_RESTARTS=10
while getopts "p:d:b:n:r:" opt; do
  case "$opt" in
    p) TYPE_PROBLEM="$OPTARG" ;;
    d) DIM="$OPTARG" ;;
    b) BUDGET="$OPTARG" ;;
    n) NB_INSTANCES="$OPTARG" ;;
    r) NB_RESTARTS="$OPTARG" ;;
  esac
done

cd source_code

echo "=== PEDA $TYPE_PROBLEM dim=$DIM (tous les t), GPU ==="
python main_expe_peda.py "$TYPE_PROBLEM" "$DIM" \
  --budget "$BUDGET" \
  --nb-instances "$NB_INSTANCES" \
  --nb-restarts "$NB_RESTARTS"

if [ "$TYPE_PROBLEM" = "NK3" ]; then
  echo "=== PPBIL $TYPE_PROBLEM : saute (PPBIL est binaire seulement, ne supporte pas NK3) ==="
else
  echo "=== PPBIL $TYPE_PROBLEM dim=$DIM (tous les t), GPU ==="
  python main_expe_ppbil.py "$TYPE_PROBLEM" "$DIM" \
    --budget "$BUDGET" \
    --nb-instances "$NB_INSTANCES" \
    --nb-restarts "$NB_RESTARTS"
fi

echo "Job termine a $(date)"
