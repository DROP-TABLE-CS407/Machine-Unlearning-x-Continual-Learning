#!/bin/bash
#
#SBATCH --job-name=continual-learning     # Job name for tracking
#SBATCH --partition=falcon      # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --mem=55000            # 60GB RAM
#SBATCH --time=12:00:00        # Job time limit set to 12 hours
#
#SBATCH --output=jupyter.log
#SBATCH --error=jupyter.err

pip3.9 install jupyter

module load NCCL/2.20.3_for_CUDA12.2
module load CUDA/12.2-cudnn9
module load cs413-python

cd /dcs/large/u2145461/cs407/ # You might want to use the cd command here to change the working directory that jupyter notebook will use
pip3.9 install --user quadprog

python3.9 -m notebook --ip=0.0.0.0 --port=11861 --no-browser
