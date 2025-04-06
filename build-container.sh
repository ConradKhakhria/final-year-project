#!/bin/bash
#$ -N build-apptainer
#$ -cwd
#$ -l h_rt=00:15:00
#$ -l mem=20G
#$ -pe smp 2
#$ -o $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.out
#$ -e $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.err

# Load Apptainer
source /etc/profile
module load apptainer

# Set cache and temp dirs to avoid quota and /tmp issues
export APPTAINER_TMPDIR=$HOME/Scratch/tmp
export APPTAINER_CACHEDIR=$HOME/Scratch/apptainer-cache

# Clean up any previous sandbox container
rm -rf $HOME/Scratch/container/experiment-container

echo "Running on node: $(hostname)"

# Build the sandbox container with fakeroot
apptainer build --fakeroot --sandbox \
  $HOME/Scratch/container/experiment-container \
  $HOME/ACFS/final-year-project/experiment-container.def

echo "Finished building sandbox container"
