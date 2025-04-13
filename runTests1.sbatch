#!/bin/bash
#
#SBATCH --job-name=neggem     # Job name for tracking
#SBATCH --partition=gecko      # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48      # Number of CPU threads used by your job
#SBATCH --gres=gpu:3           # Number of GPUs to use 
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

export CUDA_VISIBLE_DEVICES=0,1,2

nvidia-smi

# args are as follows (in respective orders): unlearn_mem_strength, unlearn_batch_size,
# average over_n_runs, salun on/off, salun strength, rum on/off, rum_split, rum_memorization
time python3.12 negGemGradSalun.py --algorithm neggem --alpha 0 --number_of_gpus 3 --learn_mem_strength 0.5 --learn_batch_size 10 --unlearn_mem_strength 0.6 --unlearn_batch_size 10 --average_over_n_runs 3 --salun 1 --salun_strength 0.2 --rum 1 --rum_split 0.99 --rum_memorization most
time python3.12 negGemGradSalun.py --algorithm negagem --alpha 0 --number_of_gpus 3 --learn_mem_strength 0.5 --learn_batch_size 10 --unlearn_mem_strength 0.6 --unlearn_batch_size 10 --average_over_n_runs 3 --salun 1 --salun_strength 0.2 --rum 1 --rum_split 0.99 --rum_memorization most
time python3.12 negGemGradSalun.py --algorithm RL-GEM --alpha 0 --number_of_gpus 3 --learn_mem_strength 0.5 --learn_batch_size 10 --unlearn_mem_strength 0.6 --unlearn_batch_size 10 --average_over_n_runs 3 --salun 1 --salun_strength 0.2 --rum 1 --rum_split 0.99 --rum_memorization most
time python3.12 negGemGradSalun.py --algorithm RL-AGEM --alpha 0 --number_of_gpus 3 --learn_mem_strength 0.5 --learn_batch_size 10 --unlearn_mem_strength 0.6 --unlearn_batch_size 10 --average_over_n_runs 3 --salun 1 --salun_strength 0.2 --rum 1 --rum_split 0.99 --rum_memorization most
time python3.12 negGemGradSalun.py --algorithm ALT-NEGGEM --alpha 0 --number_of_gpus 3 --learn_mem_strength 0.5 --learn_batch_size 10 --unlearn_mem_strength 0.6 --unlearn_batch_size 10 --average_over_n_runs 3 --salun 1 --salun_strength 0.2 --rum 1 --rum_split 0.99 --rum_memorization most
time python3.12 negGemGradSalun.py --algorithm neggrad --alpha 0.9 --number_of_gpus 3 --learn_mem_strength 0.5 --learn_batch_size 10 --unlearn_mem_strength 0.6 --unlearn_batch_size 10 --average_over_n_runs 3 --salun 1 --salun_strength 0.2 --rum 1 --rum_split 0.99 --rum_memorization most

# move all the results and joboutput files to all_algos_unlearn
#mv joboutput_*.out ./all_algos_unlearn
#mv joboutput_*.err ./all_algos_unlearn
#mv ./Results* ./all_algos_unlearn
