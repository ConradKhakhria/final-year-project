#!/bin/bash
#$ -N build-apptainer
#$ -cwd
#$ -l h_rt=00:10:00
#$ -l mem=20G
#$ -pe smp 2
#$ -o $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.out
#$ -e $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.err

# Load Apptainer
source /etc/profile
module load apptainer

echo "Running on node: $(hostname)"

# Build the container:
# --fakeroot: builds as if running as root (required for unprivileged builds)
# -F / --force: overwrites any existing SIF file without prompting
# --fix-perms: adjusts file permissions to avoid read/write issues
# --tmpdir: uses your scratch space for temporary build files
apptainer build \
    --fakeroot -F --fix-perms --tmpdir=$HOME/Scratch/tmp \
    $HOME/Scratch/container/experiment-container.sif     \
    $HOME/ACFS/final-year-project/experiment-container.def

echo "done"