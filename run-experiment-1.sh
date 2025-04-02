#!/bin/bash
#$ -N fyp-gpu
#$ -cwd
#$ -l h_rt=00:05:00
#$ -l gpu=0
#$ -l mem=2G
#$ -pe smp 4
#$ -o $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.out
#$ -e $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.err

# Load env
source /etc/profile

# Load required modules
module purge

module load pytorch/2.1.0/gpu
module load cuda/11.3.1/gnu-10.2.0
module load cudnn/8.2.1.32/cuda-11.3
module load gcc-libs/10.2.0        

# change to temporary directory
cd $TMPDIR

# Optional: activate virtualenv or conda
source $HOME/ACFS/final-year-project/venv/bin/activate

# Run the script
python3 $HOME/ACFS/final-year-project/experiment-1.py

# move output to Scratch
# Move output to persistent storage
if [ -d "$TMPDIR/data" ]; then
    mv $TMPDIR/data $HOME/Scratch/fyp/data_$JOB_ID
else
    echo "No data directory created."
fi