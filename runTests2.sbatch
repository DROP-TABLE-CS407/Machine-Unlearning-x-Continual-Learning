#!/bin/bash
#
#SBATCH --job-name=neggem     # Job name for tracking
#SBATCH --partition=eagle      # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1           # Number of GPUs to use 
#SBATCH --mem=60G            # 60GB RAM
#SBATCH --time=2:00:00        # Job time limit set to 12 hours
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

export CUDA_VISIBLE_DEVICES=0 #,1,2

nvidia-smi

BUFFER_ARGS="--mem_learning_buffer 1 \
    --learning_buffer_split 0.2 \
    --learning_buffer_type least \
    --mem_unlearning_buffer 1 \
    --unlearning_buffer_split 0.99 \
    --unlearning_buffer_type most"

SALUN_ARGS="--salun 1 \
    --salun_strength 0.5"

MEM_STRENGTH_ARGS="--learn_mem_strength 0.5 \
    --unlearn_mem_strength 0.5"

LEARN_BATCH_SIZE_ARGS="--learn_batch_size 100 \
    --unlearn_batch_size 8"

ALPHA_ARGS="--alpha 0.5"

NUMBER_ARGS="--number_of_gpus 1 \
    --average_over_n_runs 1"

time python3.12 negGemGradSalun.py --algorithm RL-GEM $BUFFER_ARGS $SALUN_ARGS $MEM_STRENGTH_ARGS $LEARN_BATCH_SIZE_ARGS $ALPHA_ARGS $NUMBER_ARGS

# move all the results and joboutput files to finetune
mv joboutput_*.out ./finetune
mv joboutput_*.err ./finetune
mv ./Results* ./finetune
