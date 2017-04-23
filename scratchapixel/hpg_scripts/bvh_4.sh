#!/bin/sh
#SBATCH --job-name=bvh      # Job name
#SBATCH --mail-type=ALL              # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=khaledjhassan@ufl.edu  # Where to send mail	
#SBATCH --nodes=1                    # Use one node
#SBATCH --ntasks=1                   # Run a single task	
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --mem=1gb                  # Total memory limit
#SBATCH --time=00:30:00              # Time limit hrs:min:sec
#SBATCH --output=bvh_4_%j.out     # Standard output and error log
#SBATCH --dependency=singleton       # only run 1 at a time because the program creates files and I don't want them to fight

pwd; hostname; date

export OMP_NUM_THREADS=4

module load gcc/5.2.0

cd /home/khaledjhassan/ray-tracing/scratchapixel/bvh

time ./acceleration

date
