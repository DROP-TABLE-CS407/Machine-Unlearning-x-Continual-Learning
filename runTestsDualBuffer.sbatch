#!/bin/bash
#
#SBATCH --job-name=memGem     # Job name for tracking
#SBATCH --partition=falcon      # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1        # Number of GPUs to use 
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

# nvidia-smi

COMMON_ARGS="--unlearn_mem_strength 0.6 --unlearn_batch_size 10 --average_over_n_runs 3 --salun 1 --salun_strength 0.2"

##############################
# Experiment 1: Baseline
##############################
echo "Running Baseline (No buffers)"
python3.12 negGemGradSalunDualBuffer.py $COMMON_ARGS \
    --mem_learning_buffer 0 \
    --mem_unlearning_buffer 0

mkdir -p Results_Baseline_NoBuffers
mv Results_L* Results_Baseline_NoBuffers/

##############################
# Experiment 2: 0.2 Least / Most
##############################
echo "Running Buffer: 0.2 least (learn), 0.2 most (unlearn)"
python3.12 negGemGradSalunDualBuffer.py $COMMON_ARGS \
    --mem_learning_buffer 1 \
    --learning_buffer_split 0.2 \
    --learning_buffer_type least \
    --mem_unlearning_buffer 1 \
    --unlearning_buffer_split 0.2 \
    --unlearning_buffer_type most

mkdir -p Results_Buffer_0.2_Least_Most
mv Results_L* Results_Buffer_0.2_Least_Most/

##############################
# Experiment 3: 0.99 Least / Most
##############################
echo "Running Buffer: 0.99 least (learn), 0.2 most (unlearn)"
python3.12 negGemGradSalunDualBuffer.py $COMMON_ARGS \
    --mem_learning_buffer 1 \
    --learning_buffer_split 0.99 \
    --learning_buffer_type least \
    --mem_unlearning_buffer 1 \
    --unlearning_buffer_split 0.2 \
    --unlearning_buffer_type most

mkdir -p Results_Buffer_0.99_Least_Most
mv Results_L* Results_Buffer_0.99_Least_Most/
