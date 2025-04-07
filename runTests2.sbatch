#!/bin/bash
#
#SBATCH --job-name=continual-learning-2     # Job name for tracking
#SBATCH --partition=eagle      # Partition you wish to use (see above for list)
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

time python3.12 negGemGradSalun.py --unlearn_mem_strength 0.6 --unlearn_batch_size 16 --average_over_n_runs 1 --salun 0 --salun_strength 0.5 --rum 0 --rum_split 0.5 --rum_memorization all

# args are as follows (in respective orders): unlearn_mem_strength, unlearn_batch_size,
# average over_n_runs, salun on/off, salun strength, rum on/off, rum_split, rum_memorization
time python3.12 negGemGradSalun.py --unlearn_mem_strength 0.6 --unlearn_batch_size 16 --average_over_n_runs 1 --salun 1 --salun_strength 0.1 --rum 1 --rum_split 0.5 --rum_memorization most








