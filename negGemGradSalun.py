import os
import sys
import argparse
import torch.multiprocessing as mp

# args are as follows (in respective orders): unlearn_mem_strength, unlearn_batch_size,
# average over_n_runs, salun on/off, salun strength, rum on/off, rum_split, rum_memorization

parser = argparse.ArgumentParser()
parser.add_argument('--number_of_gpus', dest='number_of_gpus', type=str, help='Add number of GPUs')
parser.add_argument('--algorithm', dest='algorithm', type=str, help='Add algorithm')
parser.add_argument('--alpha', dest='alpha', type=str, help='Add alpha')
parser.add_argument('--learn_mem_strength', dest='learn_mem_strength', type=str, help='Add learn_mem_strength')
parser.add_argument('--learn_batch_size', dest='learn_batch_size', type=str, help='Add learn_batch_size')
parser.add_argument('--unlearn_mem_strength', dest='unlearn_mem_strength', type=str, help='Add unlearn_mem_strength')
parser.add_argument('--unlearn_batch_size', dest='unlearn_batch_size', type=str, help='Add unlearn_batch_size')
parser.add_argument('--average_over_n_runs', dest='average_over_n_runs', type=str, help='Add average_over_n_runs')
parser.add_argument('--salun', dest='salun', type=str, help='Add salun on/off')
parser.add_argument('--salun_strength', dest='salun_strength', type=str, help='Add salun strength')
parser.add_argument('--mem_learning_buffer', type=int, default=1, help='Enable memorization optimised memory buffer for learning')
parser.add_argument('--learning_buffer_split', type=float, default=0.2, help='Proportion of buffer selected by score vs random')
parser.add_argument('--learning_buffer_type', type=str, default="least", choices=["least", "most", "random"], help='Memorization type used for selecting learning buffer samples')

# Unlearning buffer settings
parser.add_argument('--mem_unlearning_buffer', type=int, default=1, help='Enable memorization optimised buffer for unlearning')
parser.add_argument('--unlearning_buffer_split', type=float, default=0.2, help='Proportion of buffer selected by score vs random for unlearning')
parser.add_argument('--unlearning_buffer_type', type=str, default="most", choices=["least", "most", "random"], help='Memorization type used for selecting unlearning buffer samples')
cmd_args = parser.parse_args()

print("Command line arguments:")
print("number_of_gpus:", cmd_args.number_of_gpus)
print("algorithm:", cmd_args.algorithm)
print("alpha:", cmd_args.alpha)
print("learn_mem_strength:", cmd_args.learn_mem_strength)
print("learn_batch_size:", cmd_args.learn_batch_size)
print("unlearn_mem_strength:", cmd_args.unlearn_mem_strength)
print("unlearn_batch_size:", cmd_args.unlearn_batch_size)
print("average_over_n_runs:", cmd_args.average_over_n_runs)
print("salun:", cmd_args.salun)
print("salun_strength:", cmd_args.salun_strength)

# run command to check python version
os.system('python3.12 --version')

# set directory /dcs/large/u2145461/cs407/Machine-Unlearning-x-Continual-Learning
# please change these dependent on your own specific path variable

# output pwd, PRINT IT
print("Current working directory: ", os.getcwd())

# now use the pwd command
os.system('pwd')

os.chdir(os.getcwd())

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import time
from negGem.cifar import load_cifar10_data, split_into_classes, get_class_indexes, load_data
import negGem.cifar
from torch.utils.data import DataLoader
import random

import sys
sys.path.append(os.path.abspath("."))  # Adds the current directory
# from GEM.gem import *

from torch.nn.functional import relu, avg_pool2d
import torch.nn as nn
import quadprog

from negGem.args import *
from negGem.util import *
from negGem.eval import *
from negGem.salun import *
from negGem.net import *

mem_data = np.load("Memorization/cifar100_mem.npz")  # Replace with actual file path
mem_scores = mem_data["tr_mem"]

ngpus = torch.cuda.device_count()
print("Number of GPUs visible: ", ngpus)

# we provide some top level initial parameters depending on if we want to work in cifar-10 or cifar-100

# define main function to run on the cifar dataset
N_TASKS = 20 #[2 tasks [airplane, automobile, etc], [dog , frog, etc]]
SIZE_OF_TASKS = 5
N_OUTPUTS = 100
N_INPUTS = 32 * 32 * 3

AGEM = True
PRETRAIN = 0 # number of initial classes to pretrain on
# Globals 
DATASET = 'cifar-100'
DATASET_PATH = 'cifar-100-python' 
CLASSES = negGem.cifar.CLASSES
CONFIDENCE_SAMPLES = 5
if DATASET == 'cifar-10':
    CLASSES = negGem.cifar.CLASSES
    CLASSES = CLASSES.copy()
elif DATASET == 'cifar-100':
    CLASSES = negGem.cifar.CLASSES_100_UNORDERED
    
NUMBER_OF_GPUS = 1
    
def run_cifar(algorithm, args, n_inputs=N_INPUTS, n_outputs=N_OUTPUTS, n_tasks=N_TASKS, size_of_task=SIZE_OF_TASKS, newclasses = [], mem_split=0, mem_type="a", device = 0, mem_data_local = [], task_sequence=[]):
    # Default to standard sequence if none provided: learn all tasks in order
    if not task_sequence:
        task_sequence = list(range(n_tasks))
    
    SHUFFLEDCLASSES = newclasses
    ALL_TASK_ACCURACIES = []  # Single list to track accuracies for all operations
    ALL_TASK_CONFIDENCES = []  # Single list to track confidence for all operations
    ALL_TASK_ACCURACIES_TRAIN = []  # Single list to track accuracies for all operations
    ALL_TASK_CONFIDENCES_TRAIN = []  # Single list to track confidence for all operations
    
    # Set up the model
    model = Net(n_inputs, n_outputs, n_tasks, args)
    if args.cuda:
        torch.cuda.set_device(device)
        model = model.cuda()
        # output the device we are running on
        print("Running on device: ", torch.cuda.get_device_name(device))
    model.is_cifar = True
    test_bs = 1000
    print("Unlearn Batch Size: ", args.unlearn_batch_size)
    print("Unlearn MEM Strength: ", args.unlearn_memory_strength)
    
    test_accuracies = []

    # Load data
    train_data, train_labels, test_data, test_labels = load_data(DATASET_PATH, DATASET)
    ## NEWCLASSES = shuffled CLASSES
    NEWCLASSES = newclasses
    print("new ordering of classes: ", NEWCLASSES)
    oldClasstoNewClass = {}
    for i in range(len(CLASSES)):
        oldClasstoNewClass[i] = NEWCLASSES.index(CLASSES[i])
    for i in range(len(train_labels)):
        train_labels[i] = oldClasstoNewClass[train_labels[i]]
    for i in range(len(test_labels)):
        test_labels[i] = oldClasstoNewClass[test_labels[i]]
        
    new_classes = [None] * len(CLASSES)
    for original_class, new_index in oldClasstoNewClass.items():
        new_classes[new_index] = original_class
    
    # Create a dictionary that maps each task number to the ordered list of classes for that task.
    task_mapping = {}
    total_classes = len(new_classes)
    classes_per_task = total_classes // n_tasks
    remainder = total_classes % n_tasks

    start_idx = 0
    for task in range(n_tasks):
        # Distribute the remainder: the first 'remainder' tasks get one extra class.
        extra = 1 if task < remainder else 0
        end_idx = start_idx + classes_per_task + extra
        task_mapping[task] = new_classes[start_idx:end_idx]
        start_idx = end_idx

    pretrain_classses = NEWCLASSES[:PRETRAIN]
    pretrain_data, pretrain_labels = split_into_classes(train_data, train_labels, pretrain_classses, NEWCLASSES)
    pretest_data, pretest_labels = split_into_classes(test_data, test_labels, pretrain_classses, NEWCLASSES)
    tasks = []
    tests = []
    if PRETRAIN > 0:
        tasks = [[pretrain_data, pretrain_labels]]
        tests = [pretest_data, pretest_labels]
    else:
        tasks = []
        tests = []
    for i in range(n_tasks):
        if i == 0 and PRETRAIN > 0: ## as task 1 we already grab from 
            continue
        ## Since we have already pretrain on the first x classes, we need to offset our counter by x to learn the next set of classes
        elif PRETRAIN > 0:
            task_data, task_labels = split_into_classes(train_data, train_labels, NEWCLASSES[PRETRAIN + size_of_task * (i-1) : PRETRAIN + size_of_task * (i)], NEWCLASSES)
            tasks.append([task_data, task_labels])
            partition_test_data, partition_test_labels = split_into_classes(test_data, test_labels, NEWCLASSES[PRETRAIN + size_of_task * (i-1) : PRETRAIN + size_of_task * (i)], NEWCLASSES)
            tests.append(partition_test_data)
            tests.append(partition_test_labels)
        ## no pretraining, carry on as normal
        else:
            task_data, task_labels = split_into_classes(train_data, train_labels, NEWCLASSES[size_of_task * i : size_of_task * (i + 1)] , NEWCLASSES)
            tasks.append([task_data, task_labels])
            partition_test_data, partition_test_labels = split_into_classes(test_data, test_labels, NEWCLASSES[size_of_task * i : size_of_task * (i + 1)] , NEWCLASSES)
            tests.append(partition_test_data)
            tests.append(partition_test_labels)

    retain_set = tasks[:1]  # First task as retain set
    forget_set = tasks[1:]  # Rest as forget set (will be used for evaluation only)
    test_set = tests
    
    retain_accuracies = []
    forget_accuracies = []
    testing_accuracies = []
    testing_accuracies_forget = []
    
    unlearning_algo = args.algorithm
    
    # Process the task sequence
    for operation in task_sequence:
        if operation >= 0:  # Learn task
            task = operation
            print("Training task: ", task + 1)
            
            x = torch.Tensor(tasks[task][0].reshape(-1, 32*32*3)).float()
            y = torch.Tensor(tasks[task][1]).long()
            
            if args.cuda:
                x, y = x.cuda(), y.cuda()
        
            max_idx = 0
            for epoch in range(args.n_epochs):
                for j in range(0, len(tasks[task][0]), args.batch_size):
                    current_data = x[j: j + args.batch_size]
                    current_labels = y[j: j + args.batch_size]
                    model.train()
                    model.observe(algorithm, current_data, task, current_labels)
                    max_idx += args.batch_size
                
                # Test the model after each epoch
                correct = 0
                total = len(tasks[task][0])
                for j in range(0, len(tasks[task][0]), test_bs):
                    current_data = x[j: j + test_bs]
                    current_labels = y[j: j + test_bs]
                    output = model.forward(current_data, task)
                    pred = output.data.max(1)[1]

                if correct / total > 0.7:
                    break
                
            if args.mem_learning_buffer:
                print(f"[Learn Buffer] Updating memory for task {task + 1} using {args.learning_buffer_type} memorization.")
                model.update_memory_from_dataset(
                    x[:max_idx], y[:max_idx], task, task_mapping,
                    mem_scores, mem_data_local,
                    mem_split=args.learning_buffer_split,
                    mem_type=args.learning_buffer_type,
                    buffer_type='learn'
                )

            if args.mem_unlearning_buffer:
                print(f"[Unlearn Buffer] Updating memory for task {task + 1} using {args.unlearning_buffer_type} memorization.")
                model.update_memory_from_dataset(
                    x[:max_idx], y[:max_idx], task, task_mapping,
                    mem_scores, mem_data_local,
                    mem_split=args.unlearning_buffer_split,
                    mem_type=args.unlearning_buffer_type,
                    buffer_type='unlearn'
                )
        
        else:  # Unlearn task (negative value)
            task_to_unlearn = abs(operation)
            print(f"Unlearning task: {task_to_unlearn + 1}")
            
            model.train()
            model.opt = torch.optim.SGD(model.parameters(), args.unlearning_rate)
            flag = False
            mask = None
            
            for epoch in range(args.unlearn_epochs):
                if flag:
                    print("broken out of loop")
                    break
                    
                for j in range(0, args.n_memories, args.unlearn_batch_size):
                    # Check if the model has sufficiently unlearned the task
                    memory_data = model.unlearn_memory_data[task_to_unlearn]
                    memory_labels = model.unlearn_memory_labs[task_to_unlearn]
                    correct = 0
                    total = len(memory_data)
                    
                    for k in range(0, len(memory_data), min(test_bs, len(memory_data))):
                        current_data = memory_data[k: k + min(test_bs, len(memory_data))]
                        current_labels = memory_labels[k: k + min(test_bs, len(memory_data))]
                        output = model.forward(current_data, task_to_unlearn)
                        pred = output.data.max(1)[1] 
                        correct += (pred == current_labels).sum().item()
                        
                    if correct / total <= 0.333333333333333333:
                        flag = True
                        break
                        
                    if not flag:
                        mask = model.unlearn(unlearning_algo, task_to_unlearn, x1=j, x2=j + args.unlearn_batch_size, alpha=args.alpha, mask=mask)
            
            # Remove the unlearned task from observed tasks
            if task_to_unlearn in model.observed_tasks:
                model.observed_tasks.remove(task_to_unlearn)
        
        # Evaluate performance after each operation (both learning and unlearning)
        model.eval()
        temp_retain_acc, temp_forget_acc, temp_testing_acc, temp_testing_forget_acc = eval_retain_forget_test(
            model, args, retain_set, forget_set, test_set, [], [], [], [])
        
        retain_accuracies.append(temp_retain_acc)
        forget_accuracies.append(temp_forget_acc)
        testing_accuracies.append(temp_testing_acc)
        testing_accuracies_forget.append(temp_testing_forget_acc)
        
        # Evaluate performance on all individual tasks
        temp_test_accuracies = []
        temp_test_confidences = []
        temp_train_accuracies = []
        temp_train_confidences = []
        for task_idx in range(n_tasks):
            test_acc, confidence = eval_task(model, args,
                                        tests[task_idx * 2], tests[task_idx * 2 + 1],
                                        task_idx)
            temp_test_accuracies.append(test_acc)
            temp_test_confidences.append(confidence)
            train_acc, train_confidence = eval_task(model, args,
                                        tasks[task_idx][0], tasks[task_idx][1],
                                        task_idx)
            temp_train_accuracies.append(train_acc)
            temp_train_confidences.append(train_confidence)
        
        ALL_TASK_ACCURACIES.append(temp_test_accuracies)
        ALL_TASK_CONFIDENCES.append(temp_test_confidences)
        ALL_TASK_ACCURACIES_TRAIN.append(temp_train_accuracies)
        ALL_TASK_CONFIDENCES_TRAIN.append(temp_train_confidences)
        
        # Print current state
        print(f"After {'learning' if operation >= 0 else 'unlearning'} task {abs(operation) + 1}")
        print(f"Task accuracies: {temp_test_accuracies}")
    
    # For compatibility with the existing code
    after_unlearn_accuracies = ALL_TASK_ACCURACIES[-1] if ALL_TASK_ACCURACIES else []
    confidence_after_unlearn = ALL_TASK_CONFIDENCES[-1] if ALL_TASK_CONFIDENCES else []
    
    return (model,
            test_accuracies,
            [],  # average_confidence - now integrated in ALL_TASK_CONFIDENCES
            after_unlearn_accuracies,
            confidence_after_unlearn,
            retain_accuracies,
            forget_accuracies,
            testing_accuracies,
            testing_accuracies_forget,
            ALL_TASK_ACCURACIES,
            ALL_TASK_CONFIDENCES,
            ALL_TASK_ACCURACIES_TRAIN,
            ALL_TASK_CONFIDENCES_TRAIN)  # Empty for ALL_TASK_CONTINUAL_LEARNING_ACCURACIES as it's now integrated

# We move the single run logic into a function:
def single_run(run_idx, SHUFFLEDCLASSES, cmd_args, mem_data_local):
    # make a copy of the mem_data .copy() doesn't work as it's a NpzFile
    
    # Assign GPU in round-robin fashion
    dev = run_idx % NUMBER_OF_GPUS
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(dev)
    torch.cuda.set_device(dev)
    torch.cuda.empty_cache()

    random.shuffle(SHUFFLEDCLASSES)
    args = Args()
    args.memory_strength = float(cmd_args.learn_mem_strength)
    args.batch_size = int(cmd_args.learn_batch_size)
    args.unlearn_memory_strength = float(cmd_args.unlearn_mem_strength)
    args.unlearn_batch_size = int(cmd_args.unlearn_batch_size)
    if int(cmd_args.salun) == 1:
        args.salun = True
        args.salun_threshold = float(cmd_args.salun_strength)
    else:
        args.salun = False
        
    args.algorithm = cmd_args.algorithm
    if args.algorithm == 'RL-GEM':
        args.unlearning_rate = 0.03
    if args.algorithm == 'RL-AGEM':
        args.unlearning_rate = 0.03
    args.alpha = float(cmd_args.alpha)
    
    args.mem_learning_buffer = int(cmd_args.mem_learning_buffer)
    args.learning_buffer_split = float(cmd_args.learning_buffer_split)
    args.learning_buffer_type = cmd_args.learning_buffer_type
    
    args.mem_unlearning_buffer = int(cmd_args.mem_unlearning_buffer)
    args.unlearning_buffer_split = float(cmd_args.unlearning_buffer_split)
    args.unlearning_buffer_type = cmd_args.unlearning_buffer_type
        
    task_sequence = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1]

    # Run your main training/unlearning (the run_cifar function, etc.)
    # Replace "..." with your original code for a single iteration
    # (omitting code details here to keep it concise)
    (model,
    test_accuracies_GEM,
    _,
    after_unlearn_acc,
    _,
    retain_accuracies,
    forget_accuracies,
    testing_accuracies,
    testing_accuracies_forget,
    ALL_TASK_ACCURACIES,
    ALL_TASK_UNLEARN_CONFIDENCES,
    ALL_TASK_ACCURACIES_TRAIN,
    ALL_TASK_CONFIDENCES_TRAIN) = run_cifar('GEM', args, device=dev, mem_data_local=mem_data_local, newclasses=SHUFFLEDCLASSES, task_sequence=task_sequence)

    return (
        test_accuracies_GEM[-20:],
        after_unlearn_acc,
        retain_accuracies,
        forget_accuracies,
        testing_accuracies,
        testing_accuracies_forget,
        ALL_TASK_ACCURACIES,
        ALL_TASK_UNLEARN_CONFIDENCES,
        ALL_TASK_ACCURACIES_TRAIN,
        ALL_TASK_CONFIDENCES_TRAIN
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--number_of_gpus', dest='number_of_gpus', type=str, help='Add number of GPUs')
    parser.add_argument('--algorithm', dest='algorithm', type=str, help='Add algorithm: neggem, negagem, neggrad+')
    parser.add_argument('--alpha', dest='alpha', type=str, help='Add alpha')
    parser.add_argument('--learn_mem_strength', dest='learn_mem_strength', type=str, help='Add learn_mem_strength')
    parser.add_argument('--learn_batch_size', dest='learn_batch_size', type=str, help='Add learn_batch_size')
    parser.add_argument('--unlearn_mem_strength', dest='unlearn_mem_strength', type=str, help='Add unlearn_mem_strength')
    parser.add_argument('--unlearn_batch_size', dest='unlearn_batch_size', type=str, help='Add unlearn_batch_size')
    parser.add_argument('--average_over_n_runs', dest='average_over_n_runs', type=str, help='Add average_over_n_runs')
    parser.add_argument('--salun', dest='salun', type=str, help='Add salun on/off')
    parser.add_argument('--salun_strength', dest='salun_strength', type=str, help='Add salun strength')
    
    # Learning buffer settings
    parser.add_argument('--mem_learning_buffer', type=int, default=1, help='Enable memorization optimised memory buffer for learning')
    parser.add_argument('--learning_buffer_split', type=float, default=0.2, help='Proportion of buffer selected by score vs random')
    parser.add_argument('--learning_buffer_type', type=str, default="least", choices=["least", "most", "random"], help='Memorization type used for selecting learning buffer samples')

    # Unlearning buffer settings
    parser.add_argument('--mem_unlearning_buffer', type=int, default=1, help='Enable memorization optimised buffer for unlearning')
    parser.add_argument('--unlearning_buffer_split', type=float, default=0.2, help='Proportion of buffer selected by score vs random for unlearning')
    parser.add_argument('--unlearning_buffer_type', type=str, default="most", choices=["least", "most", "random"], help='Memorization type used for selecting unlearning buffer samples')
    cmd_args = parser.parse_args()

    # GPU info
    print("Available GPUs: ", torch.cuda.device_count())    
    from negGem.cifar import CLASSES_100_UNORDERED
    SHUFFLEDCLASSES = CLASSES_100_UNORDERED.copy()
    NUMBER_OF_GPUS = int(cmd_args.number_of_gpus)
    
    # print all available GPUs
    os.system('nvidia-smi')
    
    # output
    # Extract and aggregate results from all_runs
    test_accuracies_GEM_all_last_iter = []
    unlearn_accuracies_GEM_all_last_iter = []
    retain_accuracies_all = []
    forget_accuracies_all = []
    testing_accuracies_all = []
    testing_accuracies_forget_all = []
    ALL_TASK_ACCURACIES = []
    ALL_TASK_CONFIDENCES = []
    ALL_TASK_ACCURACIES_TRAIN = []
    ALL_TASK_CONFIDENCES_TRAIN = []

    # Prepare parallel runs
    average_runs = int(cmd_args.average_over_n_runs)
    # spawn a separate process for each run using the spawn method in multiprocessing
    all_results = []
    mem_data_run = {key: np.copy(mem_data[key]) for key in mem_data.files}
    
    all_results = []  # Initialize to collect all results

    for j in range(0, average_runs, NUMBER_OF_GPUS):
        # Calculate how many processes to run in this batch (might be less than NUMBER_OF_GPUS for the last batch)
        current_batch_size = min([NUMBER_OF_GPUS], average_runs - j)
        
        # Create the process pool
        with mp.Pool(processes=current_batch_size) as pool:
            # Calculate the correct indices for this batch
            batch_indices = list(range(j, j + current_batch_size))
            
            # Start the processes and wait for them to complete
            batch_results = pool.starmap(
                single_run, 
                [(i, SHUFFLEDCLASSES, cmd_args, mem_data_run) for i in batch_indices]
            )
            
            # Pool is automatically closed and joined when exiting the with block
            
        # Append results from this batch to our overall results list
        all_results.extend(batch_results)

    # Now process all the collected results after all runs are complete
    for res in all_results:
        test_accuracies_GEM_all_last_iter.append(res[0])
        unlearn_accuracies_GEM_all_last_iter.append(res[1])
        retain_accuracies_all.append(res[2])
        forget_accuracies_all.append(res[3])
        testing_accuracies_all.append(res[4])
        testing_accuracies_forget_all.append(res[5])
        ALL_TASK_ACCURACIES.append(res[6])
        ALL_TASK_CONFIDENCES.append(res[7])
        ALL_TASK_ACCURACIES_TRAIN.append(res[8])
        ALL_TASK_CONFIDENCES_TRAIN.append(res[9])
            
    # calculate the average of the test accuracies over multiple runs on ALL_TASK_ACCURACIES
    ALL_TASK_ACCURACIES_AVERAGED = np.mean(ALL_TASK_ACCURACIES, axis=0)
    ALL_TASK_CONFIDENCES_AVERAGED = np.mean(ALL_TASK_CONFIDENCES, axis=0)
    
    ALL_TRAIN_ACCURACIES_AVERAGED = np.mean(ALL_TASK_ACCURACIES_TRAIN, axis=0)
    ALL_TRAIN_CONFIDENCES_AVERAGED = np.mean(ALL_TASK_CONFIDENCES_TRAIN, axis=0)
    
    # convert to list
    ALL_TASK_ACCURACIES_AVERAGED = ALL_TASK_ACCURACIES_AVERAGED.tolist()
    ALL_TASK_CONFIDENCES_AVERAGED = ALL_TASK_CONFIDENCES_AVERAGED.tolist()
    
    ALL_TRAIN_ACCURACIES_AVERAGED = ALL_TRAIN_ACCURACIES_AVERAGED.tolist()
    ALL_TRAIN_CONFIDENCES_AVERAGED = ALL_TRAIN_CONFIDENCES_AVERAGED.tolist()
    
    # ALL_TASKS... need to be concatenated
    ALL_TASK_ACCURACIES = np.concatenate(ALL_TASK_ACCURACIES, axis=0)
    ALL_TASK_CONFIDENCES = np.concatenate(ALL_TASK_CONFIDENCES, axis=0)
    
    ALL_TASK_ACCURACIES_TRAIN = np.concatenate(ALL_TASK_ACCURACIES_TRAIN, axis=0)
    ALL_TASK_CONFIDENCES_TRAIN = np.concatenate(ALL_TASK_CONFIDENCES_TRAIN, axis=0)
    
    # convert back to lists
    ALL_TASK_ACCURACIES = ALL_TASK_ACCURACIES.tolist()
    ALL_TASK_CONFIDENCES = ALL_TASK_CONFIDENCES.tolist()
    
    ALL_TASK_ACCURACIES_TRAIN = ALL_TASK_ACCURACIES_TRAIN.tolist()
    ALL_TASK_CONFIDENCES_TRAIN = ALL_TASK_CONFIDENCES_TRAIN.tolist()
    
    # make 1 BIG figure which will contain all the plots 4x5 grid, each subplot will be 1 of the different tasks accuracy as it runs
    plt.figure(figsize=(20, 15))
    plt.suptitle('Task test set accuracies for each task as it runs', fontsize=20)
    
    import matplotlib.cm as cm
    
    # Define a colormap to get distinct colors for each task
    colors = cm.rainbow(np.linspace(0, 1, 20))
    
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        # Extract task i accuracy at each iteration (column-wise)
        task_i_accuracies = [iteration[i] for iteration in ALL_TASK_ACCURACIES_AVERAGED]
        plt.plot(task_i_accuracies, color=colors[i])
        plt.title('Task ' + str(i + 1))
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid()
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Adjust the top to make room for the suptitle
    plt.savefig('ALL_TASK_ACCURACIES.png')
    plt.show()
    # save the figure
    plt.savefig('ALL_TASK_ACCURACIES.png')
    plt.close()
    
    # do the same plot for confidence
    plt.figure(figsize=(20, 15))
    plt.suptitle('Task test set confidences for each task as it runs', fontsize=20)
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        # Extract task i confidence at each iteration (column-wise)
        task_i_confidences = [iteration[i] for iteration in ALL_TASK_CONFIDENCES_AVERAGED]
        plt.plot(task_i_confidences, color=colors[i])
        plt.title('Task ' + str(i + 1))
        plt.xlabel('Iterations')
        plt.ylabel('Confidence')
        plt.ylim(0, 1)
        plt.grid()
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Adjust the top to make room for the suptitle
    plt.savefig('ALL_TASK_CONFIDENCES.png')
    plt.show()
    # save the figure
    plt.savefig('ALL_TASK_CONFIDENCES.png')
    plt.close()
    
    # do the same plot for training accuracy
    plt.figure(figsize=(20, 15))
    plt.suptitle('Task training set accuracies for each task as it runs', fontsize=20)
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        # Extract task i accuracy at each iteration (column-wise)
        task_i_accuracies = [iteration[i] for iteration in ALL_TRAIN_ACCURACIES_AVERAGED]
        plt.plot(task_i_accuracies, color=colors[i])
        plt.title('Task ' + str(i + 1))
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid()
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Adjust the top to make room for the suptitle
    plt.savefig('ALL_TRAIN_ACCURACIES.png')
    plt.show()

    # do the same plot for training confidence
    plt.figure(figsize=(20, 15))
    plt.suptitle('Task training set confidences for each task as it runs', fontsize=20)
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        # Extract task i confidence at each iteration (column-wise)
        task_i_confidences = [iteration[i] for iteration in ALL_TRAIN_CONFIDENCES_AVERAGED]
        plt.plot(task_i_confidences, color=colors[i])
        plt.title('Task ' + str(i + 1))
        plt.xlabel('Iterations')
        plt.ylabel('Confidence')
        plt.ylim(0, 1)
        plt.grid()
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Adjust the top to make room for the suptitle
    plt.savefig('ALL_TRAIN_CONFIDENCES.png')
    plt.show()
    # save the figure
    
    import datetime

    cur_date = str(datetime.datetime.now())

    #remove spaces from the date
    cur_date = cur_date.replace(" ", "_")
    cur_date = cur_date.replace(":", "_")
    
    # write the ALL_TASK_ACCURACIES data to a pandas dataframe
    import pandas as pd
    df = pd.DataFrame(ALL_TASK_ACCURACIES)
    df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size) + 'ALL_TASK_ACCURACIES.csv')
    
    df = pd.DataFrame(ALL_TASK_ACCURACIES_AVERAGED)
    df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size) + 'ALL_TASK_ACCURACIES_AVERAGED.csv')
    
    df = pd.DataFrame(ALL_TASK_CONFIDENCES)
    df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size) + 'ALL_TASK_CONFIDENCES.csv')
    
    df = pd.DataFrame(ALL_TASK_CONFIDENCES_AVERAGED)
    df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size) + 'ALL_TASK_CONFIDENCES_AVERAGED.csv')
    
    df = pd.DataFrame(ALL_TRAIN_ACCURACIES_AVERAGED)
    df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size) + 'ALL_TRAIN_ACCURACIES_AVERAGED.csv')
    
    # make a dataframe of all the hyperparameters used
    hyperparameters = {
        'number_of_gpus': cmd_args.number_of_gpus,
        'algorithm': cmd_args.algorithm,
        'alpha': cmd_args.alpha,
        'learn_mem_strength': cmd_args.learn_mem_strength,
        'learn_batch_size': cmd_args.learn_batch_size,
        'unlearn_mem_strength': cmd_args.unlearn_mem_strength,
        'unlearn_batch_size': cmd_args.unlearn_batch_size,
        'average_over_n_runs': cmd_args.average_over_n_runs,
        'salun': cmd_args.salun,
        'salun_strength': cmd_args.salun_strength,
        'mem_learning_buffer': cmd_args.mem_learning_buffer,
        'learning_buffer_split': cmd_args.learning_buffer_split,
        'learning_buffer_type': cmd_args.learning_buffer_type,
        'mem_unlearning_buffer': cmd_args.mem_unlearning_buffer,
        'unlearning_buffer_split': cmd_args.unlearning_buffer_split,
        'unlearning_buffer_type': cmd_args.unlearning_buffer_type,
    }
    
    df = pd.DataFrame(hyperparameters, index=[0])
    df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size) + 'HYPERPARAMETERS.csv')
    
    # change the directory to /dcs/large/u2145461/cs407/Machine-Unlearning-x-Continual-Learning-neggem
    os.chdir(os.getcwd())

    os.mkdir('Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size))

    # save the results in the folder by using the mv command to move all PNG files to the folder
    os.system('mv -f *.png ./Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size))

    # save the results in the folder by using the mv command to move all PNG files to the folder
    os.system('mv -f *.csv ./Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size))

    os.system('mv -f *.pt ./Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size))
