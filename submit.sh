#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16gb
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH -t 2-00:00:00
#SBATCH --signal=SIGUSR1@90
#SBATCH -o ./slurm_output/%j.out

# activate conda environment
source /ihchomes/peng2000/.bashrc
conda activate python39
cd src
python main.py