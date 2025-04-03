#!/bin/bash
#$ -N fyp-gpu-container
#$ -cwd
#$ -l h_rt=00:05:00
#$ -l gpu=1
#$ -l mem=4G
#$ -pe smp 4
#$ -o $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.out
#$ -e $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.err

# Fix for "module: command not found"
source /etc/profile
module load apptainer

echo "Running on node: $(hostname)"
nvidia-smi || echo "No GPU detected"

# Test environment and Python inside container
apptainer exec --nv -B $HOME/ACFS/final-year-project:/project \
  $HOME/ACFS/final-year-project/pytorch_container.sif \
  python3 -m pip list

# Run your experiment
apptainer exec --nv -B $HOME/ACFS/final-year-project:/project \
  $HOME/ACFS/final-year-project/pytorch_container.sif \
  python3 /project/experiment-1.py
