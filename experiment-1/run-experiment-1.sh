#!/bin/bash
#$ -N fyp-gpu-container
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

echo "Running on node: $(hostname)"
nvidia-smi || echo "No GPU detected"

export HF_TOKEN=$(cat $HOME/ACFS/final-year-project/hf-access-token.txt)

# run experiment
apptainer exec --nv -B $HOME/ACFS/final-year-project:/project \
  $HOME/ACFS/final-year-project/experiment-container.sif \
  python3 /project/experiment-1/experiment-1.py