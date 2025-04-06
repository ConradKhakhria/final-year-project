#!/bin/bash
#$ -N experiment-1
#$ -cwd
#$ -l h_rt=00:10:00
#$ -l mem=4G
#$ -l gpu=1
#$ -pe smp 1
#$ -o $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.out
#$ -e $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.err

# Load Apptainer
source /etc/profile
module load apptainer

# Set cache/temp just in case
export APPTAINER_TMPDIR=$HOME/Scratch/tmp
export APPTAINER_CACHEDIR=$HOME/Scratch/apptainer-cache

echo "Running on node: $(hostname)"
nvidia-smi || echo "No GPU detected"

# Load Hugging Face token securely
export HF_TOKEN=$(cat $HOME/ACFS/final-year-project/new-hf-access-token.txt)

# Run the experiment from the sandbox container with GPU enabled
apptainer exec --nv \
  --env HF_TOKEN=$HF_TOKEN \
  -B $HOME/ACFS/final-year-project:/project \
  $HOME/ACFS/final-year-project/experiment-container.sif \
  python3 /project/experiment-1/experiment-1.py
