#!/usr/bin/env bash

JOB_OUTPUT_DIR=/Midgard/home/martinig/adv-comp-bio/jobs

sbatch << HERE
#!/usr/bin/env bash

#SBATCH --output="${JOB_OUTPUT_DIR}/experiment_9.out"
#SBATCH --error="${JOB_OUTPUT_DIR}/experiment_9.out"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=martinig@kth.se
#SBATCH --constrain="arwen|balrog|eowyn|khazadum|rivendell|shelob"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20GB
#SBATCH -t 0-6:00  # time limit: (D-HH:MM) 

# Check job environment
echo "JOB: \${SLURM_JOB_ID}"
echo "HOST: \$(hostname)"
echo "SUBMITTED: $(date)"
echo "STARTED: \$(date)"
echo ""
nvidia-smi

# Activate conda
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate drugVQA

echo "################################################"
echo "#####				TRAINING              ########"
echo "################################################"

python main.py

exit 0
HERE