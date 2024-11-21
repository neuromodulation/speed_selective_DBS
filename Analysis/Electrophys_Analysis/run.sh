#!/bin/sh 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=10G
#SBATCH -o outfile
#SBATCH -e errfile 
#SBATCH -t 0-00:01:00
python feature_model_optimization_bys_cluster.py