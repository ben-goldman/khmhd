#!/bin/sh
#
#
#SBATCH --account=thea
#SBATCH --job-name=khmhd
#SBATCH -c 32 # of cores per node
#SBATCH -N 4  # of nodes
#SBATCH --mem-per-cpu=5G      # The memory the job will use per cpu core
#SBATCH --time=0-08:00         # The time the job will take to run in D-HH:MM
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bog2101@columbia.edu
 
module load anaconda
# conda init bash
source /burg/home/bog2101/.bashrc
conda activate spectralDNS
date
M=8
# for N in $(seq 1 $M); do echo $N; mpiexec -n 128 python khmhd.py --M $M $M $M --N_Re $N; done  
mpiexec -n 128 python khmhd.py -M $M $M $M --N_Re $M
date

# End of script
