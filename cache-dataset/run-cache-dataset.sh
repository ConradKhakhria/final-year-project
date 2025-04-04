#!/bin/bash
#$ -N fyp-cache-dataset
#$ -cwd
#$ -l h_rt=00:05:00
#$ -l mem=1G
#$ -pe smp 1
#$ -o $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.out
#$ -e $HOME/Scratch/fyp/logs/$JOB_NAME_$JOB_ID.err

source /etc/profile
module load apptainer

echo "Running on node: $(hostname)"

# Run the cache_dataset.py script inside the container.
# We bind the entire project directory so that both 'cache-dataset' and 'experiment-1' are available at /project.
# Replace 'blog_authorship_corpus' with the desired dataset name if needed.
apptainer exec -B $HOME/ACFS/final-year-project:/project \
    $HOME/ACFS/final-year-project/pytorch_container.sif \
    python3 /project/cache-dataset/cache_dataset.py blog_authorship_corpus