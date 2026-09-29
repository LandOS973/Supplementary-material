#!/bin/bash
#SBATCH --job-name=peda_ppbil512
#SBATCH --output=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/peda_ppbil512_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/ndg/uzx53rj/Supplementary-material/outputs/peda_ppbil512_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=70:00:00             # CPU, sequentiel, pas d'array : ~27h30 x 2 algos (calibrage local), marge incluse
#SBATCH --qos=qos_cpu-t4            # necessaire, >20h (limite qos_cpu-t3)
#SBATCH --partition=cpu_p1
#SBATCH --account=ndg@cpu

# --- PEDA et PPBIL sur QUBO n=512, CPU, un seul job sequentiel (pas d'array) ---
# Chaque script gere nb_instances x nb_restarts en interne et balaie tous les t
# d'un coup (type_instance omis) -> 2 appels suffisent pour couvrir les 6 t x 100
# runs de chaque algo. Pas de parallelisme ici (contrairement a PBIL/MIMIC/BOA,
# cf. expe_edas_qubo512.slurm) : c'est un choix delibere pour rester simple, au prix
# d'un temps mur plus long (~27h30/algo d'apres le calibrage local CPU).
#
# main_expe_peda.py / main_expe_ppbil.py gerent deja REPO_ROOT correctement
# (os.chdir(SCRIPT_DIR) + chemins absolus), aucun fix de chemin necessaire.
#
# Sortie : results/nevergrad/{PEDA,PPBIL}/QUBO/512/<t>/..._budget_<b>_..._i_<i>_r_<r>.txt
#
# Usage :
#   sbatch jeanzay/expe_peda_ppbil_512.slurm

echo "Job ${SLURM_JOB_ID} sur $(hostname) a $(date)"

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

cd source_code

echo "=== PEDA QUBO dim=$DIM (tous les t) ==="
python main_expe_peda.py QUBO "$DIM" \
  --budget "$BUDGET" \
  --nb-instances "$NB_INSTANCES" \
  --nb-restarts "$NB_RESTARTS" \
  --no-gpu

echo "=== PPBIL QUBO dim=$DIM (tous les t) ==="
python main_expe_ppbil.py QUBO "$DIM" \
  --budget "$BUDGET" \
  --nb-instances "$NB_INSTANCES" \
  --nb-restarts "$NB_RESTARTS" \
  --no-gpu

echo "Job termine a $(date)"
