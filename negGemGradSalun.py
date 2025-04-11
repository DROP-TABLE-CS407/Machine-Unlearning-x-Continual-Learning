import os
import sys
import argparse
import torch.multiprocessing as mp

# args are as follows (in respective orders): unlearn_mem_strength, unlearn_batch_size,
# average over_n_runs, salun on/off, salun strength, rum on/off, rum_split, rum_memorization

parser = argparse.ArgumentParser()
parser.add_argument('--unlearn_mem_strength', dest='unlearn_mem_strength', type=str, help='Add unlearn_mem_strength')
parser.add_argument('--unlearn_batch_size', dest='unlearn_batch_size', type=str, help='Add unlearn_batch_size')
parser.add_argument('--average_over_n_runs', dest='average_over_n_runs', type=str, help='Add average_over_n_runs')
parser.add_argument('--salun', dest='salun', type=str, help='Add salun on/off')
parser.add_argument('--salun_strength', dest='salun_strength', type=str, help='Add salun strength')
parser.add_argument('--rum', dest='rum', type=str, help='Add rum on/off')
parser.add_argument('--rum_split', dest='rum_split', type=str, help='Add rum_split')
parser.add_argument('--rum_memorization', dest='rum_memorization', type=str, help='Add rum_memorization')
cmd_args = parser.parse_args()

print (cmd_args.unlearn_mem_strength)
print (cmd_args.unlearn_batch_size)
print (cmd_args.average_over_n_runs)
print (cmd_args.salun)
print (cmd_args.salun_strength)
print (cmd_args.rum)
print (cmd_args.rum_split)
print (cmd_args.rum_memorization)

# run command to check python version
os.system('python3.12 --version')

"""

THERE IS ALSO A DIRECTORY YOU NEED TO CHANGE AT THE BOTTOM OF THE FILE YOU SPERG

simply highlight the directory /dcs/large/u2145461/cs407/Machine-Unlearning-x-Continual-Learning-neggem

hold down ctrl + d then ctrl + v (assuming you already have your directory copied)

if you dont know what directory you're in, use pwd in the cmd line

"""

# set directory /dcs/large/u2145461/cs407/Machine-Unlearning-x-Continual-Learning
# please change these dependent on your own specific path variable

os.chdir('/dcs/large/u2145461/cs407/Machine-Unlearning-x-Continual-Learning-neggem')

save_path_1 = '/dcs/large/u2145461/cs407/Machine-Unlearning-x-Continual-Learning-neggem/GEM/Results4/'

save_path_2 = '/dcs/large/u2145461/cs407/Machine-Unlearning-x-Continual-Learning-neggem/GEM/Results/'

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
print("hi")
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
    
def run_cifar(algorithm, args, n_inputs=N_INPUTS, n_outputs=N_OUTPUTS, n_tasks=N_TASKS, size_of_task=SIZE_OF_TASKS, newclasses = [], mem_split=0, mem_type="a", device = 0, mem_data_local = []):
    SHUFFLEDCLASSES = newclasses
    ALL_TASK_UNLEARN_ACCURACIES = []
    ALL_TASK_UNLEARN_CONFIDENCES = []
    ALL_TASK_CONTINUAL_LEARNING_ACCURACIES = []
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

    # Train the model
    for task in range(n_tasks):
        print("Training task: ", task  + 1)
        
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
            
            #test the model after each epoch
            correct = 0
            total = len(tasks[task][0])
            for j in range(0,len(tasks[task][0]), test_bs):
                current_data = x[j: j + test_bs]
                current_labels = y[j: j + test_bs]
                output = model.forward(current_data, task)
                pred = output.data.max(1)[1]

            if correct / total > 0.80:
                break
            #   output loss only
            
        if args.use_rum:
            # *** Global Memory Update Step ***
            # After training on the task, update the replay buffer using the entire training set.
            print("Updating memory for task", task + 1, "using global selection.")
            # *** Memory Selection Step ***
            mem_split = args.rum_split
            mem_type = args.rum_memorization
            
            # Update the memory for the current task using the selected examples.
            model.update_memory_from_dataset(x[:max_idx], y[:max_idx], task, task_mapping, mem_scores, mem_data_local, mem_split, mem_type)
            
        # Test the model after training
        temp_test_accuracies = []
        for task_test in range(n_tasks):
            test, _ = eval_task(model, args,
                                        tests[task_test * 2], tests[task_test * 2 + 1],
                                        task_test)
            temp_test_accuracies.append(test)
        print(temp_test_accuracies)
        ALL_TASK_CONTINUAL_LEARNING_ACCURACIES.append(temp_test_accuracies)

    # Test the model after training
        average_confidence = []
        for i in range(0, len(tests), 2):
            correct = 0
            total = len(tests[i])     

            # Test the model
            
            x = torch.Tensor(tests[i].reshape(-1, 32*32*3)).float()
            y = torch.Tensor(tests[i+1]).long()
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            model.eval()
            average_confidence_task = []
            # keep track of average confidence score
            for j in range(0,len(tests[i]), test_bs):
                current_data = x[j: j + test_bs]
                current_labels = y[j: j + test_bs]
                output = model.forward(current_data, i // 2)
                # apply softmax to get predictions
                probabilities = torch.nn.functional.softmax(output, dim=1)
                # get the maximum value of the probabilities for each image
                predicted = torch.max(probabilities, 1).values
                # get the average confidence score of the batch
                average_confidence_task.append(torch.mean(predicted).item())
                
                pred = output.data.max(1)[1]
                correct += (pred == current_labels).sum().item()

            test_accuracies.append(correct / total)
            average_confidence.append(sum(average_confidence_task) / len(average_confidence_task))
            
    # if it's the first model, save the model
    #if iternumb == 1:
    #    torch.save(model.state_dict(), 'model.pth')
    
    unlearning_algo = "neggem"
    #unlearning_algo = "neggrad"
    if unlearning_algo == "neggrad":
        args.unlearn_batch_size = args.n_memories
    ## after training lets unlearn the last task and test the model again
    print("Unlearning task: ", n_tasks)
    
    retain_set = tasks[:1]
    forget_set = tasks[-19:]
    test_set = tests
    
    retain_accuracies = []
    
    forget_accuracies = []
    testing_accuracies = []
    
    testing_accuracies_forget = []
    
    # set the optimiser lr to 0.01
    
    model.opt = torch.optim.SGD(model.parameters(), args.unlearning_rate)
    
    retain_accuracies, forget_accuracies, testing_accuracies, testing_accuracies_forget = eval_retain_forget_test(model, args,
                                                        retain_set, forget_set, test_set, retain_accuracies, forget_accuracies,
                                                        testing_accuracies, testing_accuracies_forget)
    
    # eval performance on 20 individual tasks using the eval_task function
    temp_test_accuracies = []
    temp_test_confidences = []
    for task in range(n_tasks):
        
        test, confidence = eval_task(model, args,
                                    tests[task * 2], tests[task * 2 + 1],
                                    task)
        temp_test_accuracies.append(test)
        temp_test_confidences.append(confidence)
        print(temp_test_accuracies)
        
    ALL_TASK_UNLEARN_ACCURACIES.append(temp_test_accuracies)
    ALL_TASK_UNLEARN_CONFIDENCES.append(temp_test_confidences)

    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]: # 
        #torch.cuda.empty_cache()
        model.train()
        flag = False
        for epoch in range(args.unlearn_epochs):
            if flag:
                print("broken out of loop")
                break
            for j in range(0, args.n_memories, args.unlearn_batch_size):
                # test the model on the memories for the task we are unlearning
                # we want to see if the model is unlearning the task
                memory_data = model.memory_data[n_tasks - i]
                memory_labels = model.memory_labs[n_tasks - i]
                correct = 0
                total = len(memory_data)
                for k in range(0, len(memory_data), min(test_bs, len(memory_data))):
                    current_data = memory_data[k: k + min(test_bs, len(memory_data))]
                    current_labels = memory_labels[k: k + min(test_bs, len(memory_data))]
                    output = model.forward(current_data, n_tasks - i)
                    pred = output.data.max(1)[1] 
                    correct += (pred == current_labels).sum().item()
                if correct / total <= 0.35:
                    flag = True
                    break
                if not flag:
                    model.unlearn(unlearning_algo, n_tasks - i, x1 = j, x2 = j + args.unlearn_batch_size)


        model.observed_tasks = model.observed_tasks[:-1]
        
        model.eval()
        retain_accuracies, forget_accuracies, testing_accuracies, testing_accuracies_forget = eval_retain_forget_test(model, args,
                                                            retain_set, forget_set, test_set, retain_accuracies, forget_accuracies,
                                                            testing_accuracies, testing_accuracies_forget)
        
        # eval performance on 20 individual tasks using the eval_task function
        temp_test_accuracies = []
        temp_test_confidences = []
        for task in range(n_tasks):
            
            test, confidence = eval_task(model, args,
                                        tests[task * 2], tests[task * 2 + 1],
                                        task)
            temp_test_accuracies.append(test)
            temp_test_confidences.append(confidence)
            print(temp_test_accuracies)
            
        ALL_TASK_UNLEARN_ACCURACIES.append(temp_test_accuracies)
        ALL_TASK_UNLEARN_CONFIDENCES.append(temp_test_confidences)
    # we want the last item in the list
    print(ALL_TASK_UNLEARN_ACCURACIES)
    after_unlearn_accuracies = ALL_TASK_UNLEARN_ACCURACIES[-1]
    confidence_after_unlearn = ALL_TASK_UNLEARN_CONFIDENCES[-1]
    
    return (model,
            test_accuracies,
            average_confidence ,
            after_unlearn_accuracies,
            confidence_after_unlearn,
            retain_accuracies,
            forget_accuracies,
            testing_accuracies,
            testing_accuracies_forget,
            ALL_TASK_UNLEARN_ACCURACIES,
            ALL_TASK_UNLEARN_CONFIDENCES,
            ALL_TASK_CONTINUAL_LEARNING_ACCURACIES)

# We move the single run logic into a function:
def single_run(run_idx, SHUFFLEDCLASSES, cmd_args, mem_data_local):
    # make a copy of the mem_data .copy() doesn't work as it's a NpzFile
    
    # Assign GPU in round-robin fashion
    dev = run_idx % 3
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(dev)
    torch.cuda.set_device(dev)
    torch.cuda.empty_cache()

    random.shuffle(SHUFFLEDCLASSES)
    args = Args()
    args.unlearn_memory_strength = float(cmd_args.unlearn_mem_strength)
    args.unlearn_batch_size = int(cmd_args.unlearn_batch_size)
    if int(cmd_args.salun) == 1:
        args.salun = True
        args.salun_threshold = float(cmd_args.salun_strength)
    else:
        args.salun = False

    if int(cmd_args.rum) == 1:
        args.use_rum = True
        args.rum_split = float(cmd_args.rum_split)
        args.rum_memorization = cmd_args.rum_memorization
    else:
        args.use_rum = False
        args.rum_split = 0.5
        args.rum_memorization = "a"

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
    ALL_TASK_UNLEARN_ACCURACIES,
    ALL_TASK_UNLEARN_CONFIDENCES,
    ALL_TASK_CONTINUAL_LEARNING_ACCURACIES) = run_cifar('GEM', args, device=dev, mem_data_local=mem_data_local, newclasses=SHUFFLEDCLASSES)

    return (
        test_accuracies_GEM[-20:],
        after_unlearn_acc,
        retain_accuracies,
        forget_accuracies,
        testing_accuracies,
        testing_accuracies_forget,
        ALL_TASK_UNLEARN_ACCURACIES,
        ALL_TASK_UNLEARN_CONFIDENCES,
        ALL_TASK_CONTINUAL_LEARNING_ACCURACIES
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--unlearn_mem_strength', dest='unlearn_mem_strength', type=str)
    parser.add_argument('--unlearn_batch_size', dest='unlearn_batch_size', type=str)
    parser.add_argument('--average_over_n_runs', dest='average_over_n_runs', type=str)
    parser.add_argument('--salun', dest='salun', type=str)
    parser.add_argument('--salun_strength', dest='salun_strength', type=str)
    parser.add_argument('--rum', dest='rum', type=str)
    parser.add_argument('--rum_split', dest='rum_split', type=str)
    parser.add_argument('--rum_memorization', dest='rum_memorization', type=str)
    cmd_args = parser.parse_args()

    # GPU info
    print("Available GPUs: 3. Will distribute runs among them.")
    from negGem.cifar import CLASSES_100_UNORDERED
    SHUFFLEDCLASSES = CLASSES_100_UNORDERED.copy()
    
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
    ALL_TASK_UNLEARN_ACCURACIES = []
    ALL_TASK_UNLEARN_CONFIDENCES = []
    ALL_TASK_CONTINUAL_LEARNING_ACCURACIES = []

    # Prepare parallel runs
    average_runs = int(cmd_args.average_over_n_runs)
    # spawn a separate process for each run using the spawn method in multiprocessing
    all_results = []
    mem_data_run = {key: np.copy(mem_data[key]) for key in mem_data.files}
    
    for j in range(0, average_runs, 3):
        with mp.Pool(processes=3) as pool:
            all_results = pool.starmap(single_run, [(i, SHUFFLEDCLASSES, cmd_args, mem_data_run) for i in range(3)])
        # Close the pool
        pool.close()
        pool.join()

        for res in all_results:
            test_accuracies_GEM_all_last_iter.append(res[0])
            unlearn_accuracies_GEM_all_last_iter.append(res[1])
            retain_accuracies_all.append(res[2])
            forget_accuracies_all.append(res[3])
            testing_accuracies_all.append(res[4])
            testing_accuracies_forget_all.append(res[5])
            ALL_TASK_UNLEARN_ACCURACIES.append(res[6])
            ALL_TASK_UNLEARN_CONFIDENCES.append(res[7])
            ALL_TASK_CONTINUAL_LEARNING_ACCURACIES.append(res[8])
    
    # ALL_TASKS... need to be concatenated
    ALL_TASK_UNLEARN_ACCURACIES = np.concatenate(ALL_TASK_UNLEARN_ACCURACIES, axis=0)
    ALL_TASK_UNLEARN_CONFIDENCES = np.concatenate(ALL_TASK_UNLEARN_CONFIDENCES, axis=0)
    ALL_TASK_CONTINUAL_LEARNING_ACCURACIES = np.concatenate(ALL_TASK_CONTINUAL_LEARNING_ACCURACIES, axis=0)
    
    # convert back to lists
    ALL_TASK_UNLEARN_ACCURACIES = ALL_TASK_UNLEARN_ACCURACIES.tolist()
    ALL_TASK_UNLEARN_CONFIDENCES = ALL_TASK_UNLEARN_CONFIDENCES.tolist()
    ALL_TASK_CONTINUAL_LEARNING_ACCURACIES = ALL_TASK_CONTINUAL_LEARNING_ACCURACIES.tolist()

    # Continue with plotting/saving as in your original code:
    # (omitted for brevity—paste your existing plotting and saving logic here)

    ## average column-wise the test accuracies 
    average_test_accuracies_GEM = np.mean(test_accuracies_GEM_all_last_iter, axis=0)
    average_unlearn_accuracies_GEM = np.mean(unlearn_accuracies_GEM_all_last_iter, axis=0)

    retain_accuracies_avg = np.mean(retain_accuracies_all, axis = 0)
    forget_accuracies_avg = np.mean(forget_accuracies_all, axis = 0)
    testing_accuracies_avg = np.mean(testing_accuracies_all, axis = 0)
    testing_accuracies_forget_avg = np.mean(testing_accuracies_forget_all, axis = 0)

    plt.figure()
    plt.bar(range(1, 21), average_test_accuracies_GEM)
    plt.title('Average Test Set for each task after learning all 20 Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Tasks')
    plt.savefig('average_task_20_accuracy_comparison.png')
    plt.show()

    plt.figure()
    plt.bar(range(1, 21), average_unlearn_accuracies_GEM)
    plt.title('Average Test Set for each task after unlearning the last task Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Tasks')
    plt.savefig('average_task_20_accuracy_comparison_after_unlearning.png')
    plt.show()

    plt.figure()
    difference = average_test_accuracies_GEM - average_unlearn_accuracies_GEM
    plt.bar(range(1, 21), difference)
    plt.title('Difference between average test accuracies and average unlearn accuracies')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Tasks')
    plt.savefig('average_difference_task_20_accuracy_comparison.png')
    plt.show()

    # Define your x values
    x = [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]
    #x = [21, 20]
    # cuda function to flush the memory


    # Convert each x value into a string label
    x_labels = [str(xi) for xi in x]
    #x = [0, 1]
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    plt.figure()
    # Plot your lines
    #plt.plot(x, retain_accuracies_avg, label="Retain Accuracies", linestyle="-")
    #plt.plot(x, forget_accuracies_avg, label="Forget Accuracies", linestyle="-.")
    plt.plot(x, testing_accuracies_avg, label="Retain Test Accuracies", linestyle="-")
    plt.plot(x, testing_accuracies_forget_avg, label="Forget Test Accuracies", linestyle="-")
    # Set y-axis range
    plt.ylim(0, 1)
    # Add axis labels
    plt.xlabel("Task Unlearned")
    plt.ylabel("Accuracy")
    # Use the custom string labels for the x-axis
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    #x = [0, 1]
    plt.xticks(x, x_labels)
    # Display legend and plot 
    plt.legend()
    plt.savefig('RetainForgetTest.png')
    plt.show()

    # reverse the ALL_TASK_CONTINUAL_LEARNING_ACCURACIES list
    ALL_TASK_CONTINUAL_LEARNING_ACCURACIES.reverse()

    print(len(ALL_TASK_CONTINUAL_LEARNING_ACCURACIES))
    print(len(ALL_TASK_UNLEARN_ACCURACIES))

    avg_cont_learning_accuracies = []
    for i in range(20):
        avg_cont_learning_accuracies.append([])

    for i in range(len(ALL_TASK_CONTINUAL_LEARNING_ACCURACIES)):
        avg_cont_learning_accuracies[i%20].append(ALL_TASK_CONTINUAL_LEARNING_ACCURACIES[i])
        
    avg_cont_learning_accuracies = np.mean(avg_cont_learning_accuracies, axis = 1)

    avg_cont_unlearning_accuracies = []
    for i in range(20):
        avg_cont_unlearning_accuracies.append([])
        
    for i in range(len(ALL_TASK_UNLEARN_ACCURACIES)):
        avg_cont_unlearning_accuracies[i%20].append(ALL_TASK_UNLEARN_ACCURACIES[i])

    avg_cont_unlearning_accuracies = np.mean(avg_cont_unlearning_accuracies, axis = 1)

    # get the column 0 of the avg_cont_learning_accuracies
    avg_retain_cont_learning_accuracies = avg_cont_learning_accuracies[:, 0]

    # the forget accuracies are the average of the last 19 columns
    avg_forget_cont_learning_accuracies = np.mean(avg_cont_learning_accuracies[:, 1:], axis = 1)

    # get the column 0 of the avg_cont_unlearning_accuracies
    avg_retain_cont_unlearning_accuracies = avg_cont_unlearning_accuracies[:, 0]

    # the forget accuracies are the average of the last 19 columns
    avg_forget_cont_unlearning_accuracies = np.mean(avg_cont_unlearning_accuracies[:, 1:], axis = 1)

    plt.figure()
    plt.plot(x, avg_retain_cont_unlearning_accuracies, label="Retain Cont. Unlearn Test Accuracies", linestyle="-")
    plt.plot(x, avg_forget_cont_unlearning_accuracies, label="Forget Cont. Unlearn Test Accuracies", linestyle="--")
    plt.plot(x, avg_retain_cont_learning_accuracies, label="Retain Cont. Learn Test Accuracies (Baseline)", linestyle="-")
    plt.plot(x, avg_forget_cont_learning_accuracies, label="Forget Cont. Learn Test Accuracies (Baseline)", linestyle="--")
    plt.xticks(x, x_labels)
    plt.ylim(0, 1)
    plt.xlabel("Task")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig('RetainForgetContLearningUnlearning.png')
    plt.show()

    # now we want to create a plot similar to above but rather than plotting the retain/forget accuracies, we plot and track the accuracies of each individual task
    plt.figure()
    for i in range(avg_cont_unlearning_accuracies.shape[1]):
        plt.plot(x, avg_cont_unlearning_accuracies[:, i], label="Task " + str(i+1), linestyle="-")
    plt.xticks(x, x_labels)
    plt.ylim(0, 1)
    plt.xlabel("Task")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig('ContLearningUnlearningAllTasks.png')
    plt.show()

    plt.figure()
    for i in range(avg_cont_learning_accuracies.shape[1]):
        plt.plot(x, avg_cont_learning_accuracies[:, i], label="Task " + str(i+1), linestyle="-")
    plt.xticks(x, x_labels)
    plt.ylim(0, 1)
    plt.xlabel("Task")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig('ContLearningAllTasks.png')
    plt.show()

    # for each run in retain_accuracies_all, forget_accuracies_all, testing_accuracies_all, testing_accuracies_forget_all, plot the exactly like the cell above
    for i in range(len(retain_accuracies_all)):
        retain_accuracies_avg = retain_accuracies_all[i]
        forget_accuracies_avg = forget_accuracies_all[i]
        testing_accuracies_avg = testing_accuracies_all[i]
        testing_accuracies_forget_avg = testing_accuracies_forget_all[i]
        plt.figure()
        #plt.plot(x, retain_accuracies_avg, label="Retain Accuracies", linestyle="-")
        #plt.plot(x, forget_accuracies_avg, label="Forget Accuracies", linestyle="-.")
        plt.plot(x, testing_accuracies_avg, label="Retain Test Accuracies", linestyle="-")
        plt.plot(x, testing_accuracies_forget_avg, label="Forget Test Accuracies", linestyle="-")
        plt.ylim(0, 1)
        plt.xlabel("Task Unlearned")
        plt.ylabel("Accuracy")
        plt.xticks(x, x_labels)
        plt.legend()
        plt.savefig('RetainForgetTest' + str(i) + '.png')
        plt.show()
        print("Plot ", i, " Done")
        print("Average Retain Accuracies: ", retain_accuracies_avg)
        print("Average Forget Accuracies: ", forget_accuracies_avg)
        print("Average Retain Test Accuracies: ", testing_accuracies_avg)
        print("Average Forget Test Accuracies: ", testing_accuracies_forget_avg)
        print("Done")
        
    # create a new folder to store the results, use the mkdir command combined with the current time and memory strength and batch size

    import datetime

    cur_date = str(datetime.datetime.now())

    #remove spaces from the date
    cur_date = cur_date.replace(" ", "_")
    cur_date = cur_date.replace(":", "_")

    # save all the accuracies using pandas
    import pandas as pd

    df = pd.DataFrame(test_accuracies_GEM_all_last_iter)
    df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size) + 'TestAccuracies.csv')

    df = pd.DataFrame(unlearn_accuracies_GEM_all_last_iter)
    df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size) + 'UnlearnAccuracies.csv')

    df = pd.DataFrame(retain_accuracies_all)
    df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size) + 'RetainAccuracies.csv')

    df = pd.DataFrame(forget_accuracies_all)
    df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size) + 'ForgetAccuracies.csv')

    df = pd.DataFrame(testing_accuracies_all)
    df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size) + 'TestingAccuracies.csv')

    df = pd.DataFrame(testing_accuracies_forget_all)
    df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size) + 'TestingAccuraciesForget.csv')

    df = pd.DataFrame(ALL_TASK_UNLEARN_ACCURACIES)
    df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size) + 'AllTaskUnlearnAccuracies.csv')

    df = pd.DataFrame(ALL_TASK_UNLEARN_CONFIDENCES)
    df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size) + 'AllTaskUnlearnConfidences.csv')

    df = pd.DataFrame(ALL_TASK_CONTINUAL_LEARNING_ACCURACIES)
    df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size) + 'AllTaskContinualLearningAccuracies.csv')

    df = pd.DataFrame(avg_cont_learning_accuracies)
    df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size) + 'AvgContLearningAccuracies.csv')

    df = pd.DataFrame(avg_cont_unlearning_accuracies)
    df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size) + 'AvgContUnlearningAccuracies.csv')

    # save the model
    #torch.save(model, 'Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size) + 'Model.pt')

    # change the directory to /dcs/large/u2145461/cs407/Machine-Unlearning-x-Continual-Learning-neggem
    os.chdir('/dcs/large/u2145461/cs407/Machine-Unlearning-x-Continual-Learning-neggem')

    os.mkdir('Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size))

    # save the results in the folder by using the mv command to move all PNG files to the folder
    os.system('mv -f *.png ./Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size))

    # save the results in the folder by using the mv command to move all PNG files to the folder
    os.system('mv -f *.csv ./Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size))

    os.system('mv -f *.pt ./Results' + str(cur_date) + 'MemoryStrength' + str(cmd_args.unlearn_mem_strength) + 'BatchSize' + str(cmd_args.unlearn_batch_size))
