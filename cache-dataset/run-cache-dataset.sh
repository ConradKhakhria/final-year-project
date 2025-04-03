#!/bin/bash
#$ -N fyp-gpu-container
#$ -cwd
#$ -l h_rt=00:05:00
#$ -l mem=4G
#$ -pe smp 1
#$ -o $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.out
#$ -e $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.err

