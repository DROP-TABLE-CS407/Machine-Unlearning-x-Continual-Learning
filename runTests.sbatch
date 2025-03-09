#!/bin/bash
#
#SBATCH --job-name=continual-learning-2     # Job name for tracking
#SBATCH --partition=falcon      # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --mem=60G            # 60GB RAM
#SBATCH --time=23:00:00        # Job time limit set to 12 hours
#
#SBATCH --output=joboutput_%j.out # Standard out from your job
#SBATCH --error=joboutput_%j.err  # Standard error from your job

pip3.12 install --user pyyaml

pip3.12 install --user setuptools

pip3.12 install --user matplotlib

# You might want to use the cd command here to change the working directory that jupyter notebook will use
pip3.12 install --user quadprog
pip3.12 install --user pyparsing
pip3.12 install --user pandas
pip3.12 install --user numpy

module load NCCL/2.20.3_for_CUDA12.2
module load CUDA/12.2-cudnn9

nvidia-smi

time python3.12 negGemGrad.py 0.2 16 10
time python3.12 negGemGrad.py 0.3 16 10
time python3.12 negGemGrad.py 0.4 16 10
time python3.12 negGemGrad.py 0.5 16 10
time python3.12 negGemGrad.py 0.6 16 10
time python3.12 negGemGrad.py 0.7 16 10
time python3.12 negGemGrad.py 0.8 16 10







