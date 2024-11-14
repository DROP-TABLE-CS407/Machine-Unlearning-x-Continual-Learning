#!/bin/bash
#
#SBATCH --job-name=jupyter     # Job name for tracking
#SBATCH --partition=gecko      # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --mem=60000            # 60GB RAM
#SBATCH --time=12:00:00        # Job time limit set to 12 hours
#
#SBATCH --output=jupyter.log
#SBATCH --error=jupyter.err

source /etc/profile.d/modules.sh
module load CUDA/11.6.0
module load python-ml/22-12-21-python3.9
cd /dcs/21/u2110391/CS407/Machine-Unlearning-x-Continual-Learning
pip3 install --user quadprog
jupyter notebook --ip=0.0.0.0 --port=11891 --no-browser