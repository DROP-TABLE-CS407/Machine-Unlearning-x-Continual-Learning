#!/bin/bash
#
#SBATCH --job-name=jupyter     # Job name for tracking
#SBATCH --partition=falcon      # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --mem=60000            # 60GB RAM
#SBATCH --time=12:00:00        # Job time limit set to 12 hours
#
#SBATCH --output=jupyter.log
#SBATCH --error=jupyter.err

module load CUDA/11.6.0
module load python-ml/22-12-21-python3.9

cd /dcs/large/u2145461/cs407/Machine-Unlearning-x-Continual-Learning # You might want to use the cd command here to change the working directory that jupyter notebook will use
pip3 install --user quadprog
jupyter notebook --ip=0.0.0.0 --port=11861 --no-browser