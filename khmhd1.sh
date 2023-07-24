#!/bin/sh
#
#
#SBATCH --account=thea        # Replace ACCOUNT with your group account name
#SBATCH --job-name=khmhd   # The job name
#SBATCH -c 32                     # The number of cpu cores to use (up to 32 cores per server)
#SBATCH -N 2
#SBATCH --time=0-5:00            # The time the job will take to run in D-HH:MM
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=5G         # The memory the job will use per cpu core
#SBATCH --mail-user=bog2101@columbia.edu
 
module load anaconda
# conda init bash
source /burg/home/bog2101/.bashrc
conda activate spectralDNS
date
mpirun -n 64 python khmhd1.py
date

# End of script
