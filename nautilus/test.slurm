#!/bin/bash
#SBATCH --job-name=ppo              # Job name
#SBATCH --qos=short                 # qos
#SBATCH -p gpu                      # partition
#SBATCH --output=output/ppo_%j.out
#SBATCH --ntasks=1                  # Number of tasks (1 task in this case)
#SBATCH --gres=gpu:1                # Request 1 GPU for this job
#SBATCH --time=00:05:00             # Set the maximum job runtime to 5 minutes
# Run my first script 
echo "Run main_ppo_eda"
module load python
python ../source_code/main_ppo_eda.py -m visualization=false