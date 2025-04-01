#!/bin/bash
#$ -N fyp-gpu
#$ -cwd
#$ -l h_rt=01:00:00
#$ -l gpu=1
#$ -l mem=8G
#$ -pe smp 4
#$ -o $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.out
#$ -e $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.err


# Load required modules
module -f unload compilers mpi gcc-libs
module load beta-modules
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0
module load cuda/11.3.1/gnu-10.2.0
module load cudnn/8.2.1.32/cuda-11.3
module load pytorch/1.11.0/gpu

# change to temporary directory
cd $TMPDIR

# Optional: activate virtualenv or conda
source $HOME/ACFS/final-year-project/venv/bin/activate

# Run the script
python experiment-1.py

# move output to Scratch
mv $TMPDIR/data $HOME/Scratc/fyp
