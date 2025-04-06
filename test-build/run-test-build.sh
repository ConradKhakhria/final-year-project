#!/bin/bash
#$ -N test-fakeroot
#$ -cwd
#$ -l h_rt=00:02:00
#$ -l mem=2G
#$ -pe smp 1
#$ -o $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.out
#$ -e $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.err

source /etc/profile
module load apptainer

unset LD_PRELOAD
unset FAKEROOTKEY

export APPTAINER_TMPDIR=$HOME/Scratch/tmp
export APPTAINER_CACHEDIR=$HOME/Scratch/apptainer-cache

echo "Running on $(hostname)"

# Clean up old test container
rm -rf $HOME/Scratch/container/fakeroot-test

# Try building a basic Ubuntu sandbox with fakeroot
apptainer build --fakeroot --sandbox \
  $HOME/Scratch/container/fakeroot-test \
  docker://ubuntu:20.04
