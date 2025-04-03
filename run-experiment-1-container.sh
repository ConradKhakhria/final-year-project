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

# comment this out if you're actually using the gpu
module load pytorch/2.1.0/cpu

echo "Running on node: $(hostname)"
nvidia-smi || echo "No GPU detected"

# install numpy inside container if it's missing
apptainer exec --nv -B $HOME/ACFS/final-year-project:/project \
  $HOME/ACFS/final-year-project/pytorch_container.sif \
  python3 -m pip install numpy

# run experiment
apptainer exec --nv -B $HOME/ACFS/final-year-project:/project \
  $HOME/ACFS/final-year-project/pytorch_container.sif \
  python3 /project/experiment-1.py