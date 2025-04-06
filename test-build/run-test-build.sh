#!/bin/bash
#$ -N test-minimal
#$ -cwd
#$ -l h_rt=00:00:30
#$ -l mem=2G
#$ -pe smp 1
#$ -o $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.out
#$ -e $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.err

source /etc/profile
module load apptainer

# Remove any manual setting of LD_PRELOAD or FAKEROOTKEY:
unset LD_PRELOAD
unset FAKEROOTKEY

# Ensure the output directory exists:
mkdir -p $HOME/Scratch/container

# Build the diagnostic container as a sandbox image (directory)

apptainer --tmpdir=$HOME/Scratch/tmp build --fakeroot --sandbox $HOME/Scratch/container/test-apptainer $HOME/ACFS/final-year-project/test-build/test-apptainer.def

# Execute the sandbox container with fakeroot to display environment variables
apptainer exec --fakeroot $HOME/Scratch/container/test-apptainer /bin/bash -c "echo 'Inside container:'; echo 'FAKEROOTKEY = ' \$FAKEROOTKEY; echo 'LD_PRELOAD = ' \$LD_PRELOAD"
