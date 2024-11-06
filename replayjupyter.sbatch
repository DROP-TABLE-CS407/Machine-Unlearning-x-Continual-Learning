#!/bin/bash
#
#SBATCH --job-name=jupyter     # Job name for tracking
#SBATCH --partition=gecko      # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --mem=60000            # 60GB RAM
#SBATCH --time=24:00:00        # Job time limit set to 12 hours
#
#SBATCH --output=jupyter.log
#SBATCH --error=jupyter.err

source /etc/profile.d/modules.sh
module load CUDA
module load python-ml
cd /dcs/large/u2145461/cs407/Machine-Unlearning-x-Continual-Learning # You might want to use the cd command here to change the working directory that jupyter notebook will use
jupyter notebook --ip=0.0.0.0 --port=11861 --no-browser