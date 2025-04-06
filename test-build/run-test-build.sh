#!/bin/bash
#$ -N test-apptainer
#$ -cwd
#$ -l h_rt=00:00:30
#$ -l mem=2G
#$ -pe smp 1
#$ -o $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.out
#$ -e $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.err

# Load Apptainer
source /etc/profile
module load apptainer

# Build the diagnostic container with fakeroot
apptainer build --fakeroot --tmpdir=$HOME/Scratch/tmp \
    $HOME/Scratch/container/test/test-apptainer.sif   \
    $HOME/ACFS/final-year-project/test-build/test-apptainer.def

# Execute the container to see the diagnostic output
apptainer exec --fakeroot $HOME/Scratch/container/test.sif
