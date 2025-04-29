#!/bin/bash
#
#SBATCH --job-name=neggemA4     # Job name for tracking
#SBATCH --partition=falcon      # Partition you wish to use (see above for list)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12      # Number of CPU threads used by your job
#SBATCH --gres=gpu:1            # Number of GPUs to use 
#SBATCH --mem=60G               # 60GB RAM
#SBATCH --time=6:00:00          # Job time limit set to 12 hours
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

BUFFER_ARGS="--mem_unlearning_buffer 1 \
    --unlearning_buffer_split 0.99 \
    --unlearning_buffer_type most"

SALUN_ARGS="--salun 1 \
    --salun_strength 0.05"

MEM_STRENGTH_ARGS="--learn_mem_strength 0.5 \
    --unlearn_mem_strength 0.5"

LEARN_BATCH_SIZE_ARGS="--learn_batch_size 10 \
    --unlearn_batch_size 2"

ALPHA_ARGS="--alpha 0"

NUMBER_ARGS="--number_of_gpus 1 \
    --average_over_n_runs 1"


# time python3.12 negGemGradMIA.py --algorithm neggem --task_sequence MIA $BUFFER_ARGS $SALUN_ARGS $MEM_STRENGTH_ARGS $LEARN_BATCH_SIZE_ARGS $ALPHA_ARGS $NUMBER_ARGS

LEARN_BATCH_SIZE_ARGS="--learn_batch_size 10 \
    --unlearn_batch_size 1"

# time python3.12 negGemGradMIA.py --algorithm negagem --task_sequence MIA $BUFFER_ARGS $SALUN_ARGS $MEM_STRENGTH_ARGS $LEARN_BATCH_SIZE_ARGS $ALPHA_ARGS $NUMBER_ARGS
# time python3.12 negGemGradMIA.py --algorithm RL-AGEM --task_sequence MIA $BUFFER_ARGS $SALUN_ARGS $MEM_STRENGTH_ARGS $LEARN_BATCH_SIZE_ARGS $ALPHA_ARGS $NUMBER_ARGS

LEARN_BATCH_SIZE_ARGS="--learn_batch_size 10 \
    --unlearn_batch_size 2"

# time python3.12 negGemGradMIA.py --algorithm RL-GEM --task_sequence MIA $BUFFER_ARGS $SALUN_ARGS $MEM_STRENGTH_ARGS $LEARN_BATCH_SIZE_ARGS $ALPHA_ARGS $NUMBER_ARGS
# time python3.12 negGemGradMIA.py --algorithm ALT-NEGGEM --task_sequence MIA $BUFFER_ARGS $SALUN_ARGS $MEM_STRENGTH_ARGS $LEARN_BATCH_SIZE_ARGS $ALPHA_ARGS $NUMBER_ARGS

LEARN_BATCH_SIZE_ARGS="--learn_batch_size 10 \
    --unlearn_batch_size 4"

time python3.12 negGemGradMIA.py --algorithm neggrad --alpha 0.5 --task_sequence MIA $BUFFER_ARGS $SALUN_ARGS $MEM_STRENGTH_ARGS $LEARN_BATCH_SIZE_ARGS $NUMBER_ARGS

# move all the results and joboutput files to 4tests/A4
mv joboutput_*.out ./MIA3
mv joboutput_*.err ./MIA3
mv ./Results* ./MIA3


# LEARN_BATCH_SIZE_ARGS="--learn_batch_size 10 \
#     --unlearn_batch_size 1"

# time python3.12 negGemGradA4.py --algorithm negagem --task_sequence F4 $BUFFER_ARGS $SALUN_ARGS $MEM_STRENGTH_ARGS $LEARN_BATCH_SIZE_ARGS $ALPHA_ARGS $NUMBER_ARGS
# time python3.12 negGemGradA4.py --algorithm RL-AGEM --task_sequence F4 $BUFFER_ARGS $SALUN_ARGS $MEM_STRENGTH_ARGS $LEARN_BATCH_SIZE_ARGS $ALPHA_ARGS $NUMBER_ARGS

# LEARN_BATCH_SIZE_ARGS="--learn_batch_size 10 \
#     --unlearn_batch_size 2"

# time python3.12 negGemGradA4.py --algorithm RL-GEM --task_sequence F4 $BUFFER_ARGS $SALUN_ARGS $MEM_STRENGTH_ARGS $LEARN_BATCH_SIZE_ARGS $ALPHA_ARGS $NUMBER_ARGS
# time python3.12 negGemGradA4.py --algorithm ALT-NEGGEM --task_sequence F4 $BUFFER_ARGS $SALUN_ARGS $MEM_STRENGTH_ARGS $LEARN_BATCH_SIZE_ARGS $ALPHA_ARGS $NUMBER_ARGS
# time python3.12 negGemGradA4.py --algorithm neggem --task_sequence F4 $BUFFER_ARGS $SALUN_ARGS $MEM_STRENGTH_ARGS $LEARN_BATCH_SIZE_ARGS $ALPHA_ARGS $NUMBER_ARGS

# LEARN_BATCH_SIZE_ARGS="--learn_batch_size 10 \
#     --unlearn_batch_size 4"

# time python3.12 negGemGradA4.py --algorithm neggrad --alpha 0.95 --task_sequence F4 $BUFFER_ARGS $SALUN_ARGS $MEM_STRENGTH_ARGS $LEARN_BATCH_SIZE_ARGS $NUMBER_ARGS

# # move all the results and joboutput files to 4tests/A4
# mv joboutput_*.out ./4tests/F4
# mv joboutput_*.err ./4tests/F4
# mv ./Results* ./4tests/F4