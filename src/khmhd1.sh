#!/bin/sh
#
#
#SBATCH --account=thea
#SBATCH --job-name=khmhd
#SBATCH -c 32 # of cores per node
#SBATCH -N 2  # of nodes
#SBATCH --mem-per-cpu=5G      # The memory the job will use per cpu core
#SBATCH --time=0-5:00         # The time the job will take to run in D-HH:MM
#SBATCH --mail-type=END
#SBATCH --mail-user=bog2101@columbia.edu
 
module load anaconda
# conda init bash
source /burg/home/bog2101/.bashrc
conda activate spectralDNS
date
mpiexec -n 64 python khmhd1.py
date
ffmpeg -framerate 15 -pattern_type glob -i 'UEk*.jpg' -c:v libx264 -pix_fmt yuv420p UEk.mp4
ffmpeg -framerate 15 -pattern_type glob -i 'BEk*.png' -c:v libx264 -pix_fmt yuv420p BEk.mp4

# End of script
