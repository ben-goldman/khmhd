#!/bin/sh
#
#
#SBATCH --account=thea
#SBATCH --job-name=khmhd
#SBATCH -c 32 # of cores per node
#SBATCH -N 8  # of nodes
#SBATCH --mem-per-cpu=5G      # The memory the job will use per cpu core
#SBATCH --time=0-8:00         # The time the job will take to run in D-HH:MM
#SBATCH --mail-type=END
#SBATCH --mail-user=bog2101@columbia.edu
 
module load anaconda
# conda init bash
source /burg/home/bog2101/.bashrc
conda activate spectralDNS
date
mpiexec -n 256 python khmhd1.py
date

# End of script
