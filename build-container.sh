#!/bin/bash
#$ -N fyp-gpu-container
#$ -cwd
#$ -l h_rt=00:10:00
#$ -l mem=20G
#$ -pe smp 2
#$ -o $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.out
#$ -e $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.err

apptainer build --tmpdir=$HOME/Scratch/tmp experiment-container.sif experiment-container.def
