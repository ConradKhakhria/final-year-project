#!/bin/bash
#$ -N scratch-job
#$ -cwd
#$ -l h_rt=00:00:05
#$ -l mem=20M
#$ -pe smp 1
#$ -o $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.out
#$ -e $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.err

locate libfakeroot.so
