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

apptainer build --fakeroot --tmpdir=$HOME/Scratch/tmp \
    $HOME/Scratch/container/test-apptainer.sif \
    $HOME/ACFS/final-year-project/test-build/test-apptainer.def

apptainer exec --fakeroot \
    $HOME/Scratch/container/test-apptainer.sif /bin/bash -c "echo 'Inside container:'; echo 'FAKEROOTKEY = ' \$FAKEROOTKEY; echo 'LD_PRELOAD = ' \$LD_PRELOAD"
