#!/bin/bash
#$ -N fyp-gpu-container
#$ -cwd
#$ -l h_rt=00:05:00
#$ -l mem=4G
#$ -pe smp 1
#$ -o $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.out
#$ -e $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.err

# Load Apptainer
source /etc/profile
module load apptainer

echo "Running on node: $(hostname)"
nvidia-smi || echo "No GPU detected"

# run experiment
apptainer exec --nv -B $HOME/ACFS/final-year-project:/project \
  $HOME/ACFS/final-year-project/experiment-container.sif \
  python3 /project/experiment-1.py