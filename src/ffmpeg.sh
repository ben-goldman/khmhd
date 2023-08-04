#!/bin/sh
#
#
#SBATCH --account=thea
#SBATCH --job-name=khmhd_movie
#SBATCH -c 4 # of cores per node
#SBATCH -N 1  # of nodes
#SBATCH --mem-per-cpu=5G      # The memory the job will use per cpu core
#SBATCH --time=0-0:30         # The time the job will take to run in D-HH:MM
 
module load anaconda
# conda init bash
source /burg/home/bog2101/.bashrc
conda activate spectralDNS
ffmpeg -framerate 30 -pattern_type glob -i 'BEk*.jpg' -c:v libx264 -pix_fmt yuv420p BEk.mp4
ffmpeg -framerate 30 -pattern_type glob -i 'UEk*.jpg' -c:v libx264 -pix_fmt yuv420p UEk.mp4

# End of script
