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

export APPTAINER_TMPDIR=$HOME/Scratch/tmp
export APPTAINER_CACHEDIR=$HOME/Scratch/apptainer-cache

# Remove any manual setting of LD_PRELOAD or FAKEROOTKEY:
unset LD_PRELOAD
unset FAKEROOTKEY

# Ensure the output directory exists:
mkdir -p $HOME/Scratch/container
rm -rf $HOME/Scratch/container/test-apptainer

# Build sandbox container (required because fakeroot won't work with SIF on your system)
apptainer build --fakeroot --sandbox \
  $HOME/Scratch/container/test-apptainer \
  $HOME/ACFS/final-year-project/test-build/test-apptainer.def

# Run the sandbox with fakeroot
apptainer exec --fakeroot \
  $HOME/Scratch/container/test-apptainer \
  /bin/bash -c "echo 'Inside container:'; echo 'FAKEROOTKEY = ' \$FAKEROOTKEY; echo 'LD_PRELOAD = ' \$LD_PRELOAD"
