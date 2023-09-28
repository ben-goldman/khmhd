#!/bin/sh
#
#
#SBATCH --account=thea
#SBATCH --job-name=khmhd
#SBATCH -c 32 # of cores per node
#SBATCH -N 4  # of nodes
#SBATCH --mem-per-cpu=5G      # The memory the job will use per cpu core
#SBATCH --time=2-01:00         # The time the job will take to run in D-HH:MM
#SBATCH --mail-type=END
#SBATCH --mail-user=bog2101@columbia.edu
 
module load anaconda
# conda init bash
source /burg/home/bog2101/.bashrc
conda activate spectralDNS
date
M=8
mpiexec -n 128 python khmhd.py M=8
for N in {4..$M}; do mpiexec -n 128 python khmhd.py M=$M N=$N; done  
date

# End of script
