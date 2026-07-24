#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --error=.logs/%j.err
#SBATCH --output=.logs/%j.out
#SBATCH --account=AIFAC_P02_648
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal

srun /leonardo_work/AIFAC_P02_648/FAME/.venv/bin/python -u cifar10_cnn.py --eps 0.03