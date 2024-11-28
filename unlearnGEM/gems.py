import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import time
from cifar import load_cifar10_data, split_into_classes, get_class_indexes 
from torch.utils.data import DataLoader
import random
import os
import sys
sys.path.append(os.path.abspath("."))  # Adds the current directory
# from GEM.gem import *
from GEM.args import *
from torch.nn.functional import relu, avg_pool2d
import torch.nn as nn
import quadprog

AGEM = True
PRETRAIN = 0 # number of initial classes to pretrain on
# Globals 
DATASET_PATH = 'cifar-10-batches-py' 
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
SHUFFLEDCLASSES = CLASSES.copy()


## Scraped code fundamental for GEM this is the code required to create a resnet18 model from scratch 
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(nclasses, nf=20):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf)



### Now lets make the GEM model --  we will scrape this from the GEM code
def compute_offsets(task, nc_per_task, is_cifar):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    val1 = max(PRETRAIN - nc_per_task, 0)
    val2 = max(PRETRAIN - nc_per_task, 0)
    if task == 0:
        val1 = 0
        val2 = max(PRETRAIN - nc_per_task, 0)
    offset1 = task * nc_per_task + val1
    offset2 = (task + 1) * nc_per_task + val2    
    return offset1, offset2


def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))
    #penis
def agemprojection(gradient, gradient_memory, margin=0.5, eps=1e-3):
    """
    Projection of gradients for A-GEM with the memory approach
    Use averaged gradient memory for projection
    
    input:  gradient, g-reference
    output: gradient, g-projected
    """
    """
    # print(gradient.shape)
    gref  = gradient_memory.cpu().t().double().numpy().mean(axis=0)
    # print(gref.shape)
    g = gradient.cpu().contiguous().view(-1).double().numpy()
    # print(g.shape)
    g_transpose = g.transpose()
    # print(g_transpose.shape)
    gref_transpose = gref.transpose()
    # print(gref_transpose.shape)
    # check g_transpose and gref are of the same shape
    
    dot_prod = np.dot(g_transpose, gref.squeeze())
    # print(dot_prod.shape)
    # dot product constraint has already been checked

    # projection
    dot_prod = dot_prod / np.dot(gref, gref_transpose)
    # print(dot_prod.shape)
    g = g - np.dot(dot_prod, gref)
    # print(g.shape)
    gradient.copy_(torch.Tensor(g))
    """
    g = gradient_memory.t().double().mean(axis=0).cuda()
    gref = gradient.contiguous().view(-1).double().cuda()
    g_transpose = g.view(1, -1)
    gref_transpose = gref.view(1, -1)
    dot_prod = torch.dot(g_transpose.squeeze(), gref.squeeze())
    dot_prod = dot_prod / torch.dot(gref.squeeze(), gref_transpose.squeeze())
    # add epsilon to prevent numerical instability  
    epsvector = torch.Tensor([eps]).cuda()
    
    g = g - (gref.squeeze() * dot_prod)
    gradient.copy_(torch.Tensor(g).view(-1, 1))
    
    

class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens
        self.margin = args.memory_strength
        self.net = ResNet18(n_outputs)
     

        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs

        self.opt = torch.optim.SGD(self.parameters(), args.lr)

        self.n_memories = args.n_memories
        self.gpu = args.cuda


        """
        Allocate episodic memory
        n_tasks: number of tasks
        n_memories: number of memories per task
        n_inputs: number of input features
        """

        # allocate episodic memory
        self.memory_data = torch.FloatTensor(
            n_tasks, self.n_memories, n_inputs)
        self.memory_labs = torch.LongTensor(n_tasks, self.n_memories)
        if args.cuda:
            self.memory_data = self.memory_data.cuda()
            self.memory_labs = self.memory_labs.cuda()

        # allocate temporary synaptic memory
        """ This is the memory that stores the gradients of the parameters of the network
            FOR each task. This is used to check for violations of the GEM constraint
            Assume:

            The model has 3 parameters with sizes 100, 200, and 300 elements respectively.
            n_tasks = 5 (number of tasks).
            The allocated tensors would have the following shapes:

            self.grad_dims: [100, 200, 300]
            self.grads: Shape [600, 5] (600 is the sum of 100, 200, and 300).
        """
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        if args.cuda:
            self.grads = self.grads.cuda()

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        minus = 0
        if PRETRAIN > 0:
            minus = 1
        else: 
            minus = 0
        self.nc_per_task = int((n_outputs - PRETRAIN) / (n_tasks - minus))

    def forward(self, x, t=-1):
        output = self.net(x)
        if t == -1:
            return output
        # make sure we predict classes within the current task
        val1 = 0
        val2 = 0
        if t != 0:
            val1 = max(PRETRAIN - self.nc_per_task, 0)
            val2 = val1
        else:
            val1 = 0
            val2 = max(PRETRAIN - self.nc_per_task, 0)                                                 
        offset1 = int(t * self.nc_per_task + val1) #t = 0 0, 5 -----t = 1 5 , 6 ## t = 0 0 ,5 --- t =1 5, 7
        offset2 = int((t + 1) * self.nc_per_task + val2) 
        if offset1 > 0:
            output[:, :offset1].data.fill_(-10e10)
        if offset2 < self.n_outputs:
            output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, t, y):
        # update memory
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t
            
        val = 0
        if t == 0:
            val = max(PRETRAIN,1)
        else:
            val = 1
        # Update ring buffer storing examples from current task
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories) #256
        effbsz = endcnt - self.mem_cnt # 256
        self.memory_data[t, self.mem_cnt: endcnt].copy_(
            x.data[: effbsz])
        if bsz == 1:
            self.memory_labs[t, self.mem_cnt] = y.data[0]
        else:
            self.memory_labs[t, self.mem_cnt: endcnt].copy_(
                y.data[: effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0

        # compute gradient on previous tasks
        # if PRETRAIN == 0:
        #     val = 1
        # else:
        #     val = 0
        if len(self.observed_tasks) > 1: ### CHANGED FROM 1 to 0 SINCE WE PRETRAIN ON FST 5 CLASSES 
            for tt in range(len(self.observed_tasks) -1): ### CHANGED FROM -1 to -0 SINCE WE PRETRAIN ON FST 5 CLASSES 
                self.zero_grad()
                # fwd/bwd on the examples in the memory
                past_task = self.observed_tasks[tt]
                
                offset1, offset2 = compute_offsets(past_task, self.nc_per_task,
                                                   self.is_cifar)
                ptloss = self.ce(
                    self.forward(
                        self.memory_data[past_task],
                        past_task)[:, offset1: offset2],
                    self.memory_labs[past_task] - offset1)
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims,
                           past_task)

        # now compute the grad on the current minibatch
        self.zero_grad()

        offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar) 
        loss = self.ce(self.forward(x, t)[:, offset1: offset2], y - offset1)
        loss.backward()

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1: ### CHANGED FROM 1 to 0 SINCE WE PRETRAIN ON FST 5 CLASSES 
            if AGEM:
                store_grad(self.parameters, self.grads, self.grad_dims, t)
                indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
                    else torch.LongTensor(self.observed_tasks[:-1])
                dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                                self.grads.index_select(1, indx))
                if (dotp < 0).sum() != 0:
                    agemprojection(self.grads[:, t].unsqueeze(1), self.grads.index_select(1, indx), self.margin)
                    # copy gradients back
                    overwrite_grad(self.parameters, self.grads[:, t],
                                self.grad_dims)
            # copy gradient
            else:
                store_grad(self.parameters, self.grads, self.grad_dims, t)
                indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
                    else torch.LongTensor(self.observed_tasks[:-1])
                dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                                self.grads.index_select(1, indx))
                if (dotp < 0).sum() != 0:
                    project2cone2(self.grads[:, t].unsqueeze(1),
                                self.grads.index_select(1, indx), self.margin)
                    # copy gradients back
                    overwrite_grad(self.parameters, self.grads[:, t],
                                self.grad_dims)
        self.opt.step()

# define main function to run on the cifar dataset
N_TASKS = 5 #[2 tasks [airplane, automobile, etc], [dog , frog, etc]]
SIZE_OF_TASKS = 2
N_OUTPUTS = 10
N_INPUTS = 32 * 32 * 3
def run_cifar(args, n_inputs=N_INPUTS, n_outputs=N_OUTPUTS, n_tasks=N_TASKS, size_of_task=SIZE_OF_TASKS, newclasses = SHUFFLEDCLASSES):
    # Set up the model
    model = Net(n_inputs, n_outputs, n_tasks, args)
    if args.cuda:
        model.cuda()
    model.is_cifar = True
    test_bs = 1000
    
    test_accuracies = []

    # Load data
    train_data, train_labels, test_data, test_labels = load_cifar10_data(DATASET_PATH)
    ## NEWCLASSES = suffled CLASSES
    NEWCLASSES = newclasses
    print("new ordering of classes: ", NEWCLASSES)
    oldClasstoNewClass = {}
    for i in range(len(CLASSES)):
        oldClasstoNewClass[i] = NEWCLASSES.index(CLASSES[i])
    for i in range(len(train_labels)):
        train_labels[i] = oldClasstoNewClass[train_labels[i]]
    for i in range(len(test_labels)):
        test_labels[i] = oldClasstoNewClass[test_labels[i]]

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
    for i  in range(n_tasks):
        if i == 0 and PRETRAIN > 0: ## as task 1 we already grab from 
            continue
        elif PRETRAIN > 0:
            task_data, task_labels = split_into_classes(train_data, train_labels, NEWCLASSES[PRETRAIN + size_of_task * (i-1) : PRETRAIN + size_of_task * (i)], NEWCLASSES)
            tasks.append([task_data, task_labels])
            partition_test_data, partition_test_labels = split_into_classes(test_data, test_labels, NEWCLASSES[PRETRAIN + size_of_task * (i-1) : PRETRAIN + size_of_task * (i)], NEWCLASSES)
            tests.append(partition_test_data)
            tests.append(partition_test_labels)
        else:
            task_data, task_labels = split_into_classes(train_data, train_labels, NEWCLASSES[size_of_task * i : size_of_task * (i + 1)] , NEWCLASSES)
            tasks.append([task_data, task_labels])
            partition_test_data, partition_test_labels = split_into_classes(test_data, test_labels, NEWCLASSES[size_of_task * i : size_of_task * (i + 1)] , NEWCLASSES)
            tests.append(partition_test_data)
            tests.append(partition_test_labels)
        
    # task1data, task1labels = split_into_classes(train_data, train_labels, ['airplane', 'automobile', 'bird', 'cat', 'deer']) ## pre train on this 
    # task2data, task2labels = split_into_classes(train_data, train_labels, ['dog', 'frog', 'horse', 'ship', 'truck']) ## split this guy up into 5 tasks 
    # # tasks = [[task1data, task1labels], [task2data,task2labels]]
    # test_data_per_class_1, test_labels_per_class_1 = split_into_classes(test_data, test_labels, ['airplane', 'automobile', 'bird', 'cat', 'deer'])
    # test_data_per_class_2, test_labels_per_class_2 = split_into_classes(test_data, test_labels, ['dog', 'frog', 'horse', 'ship', 'truck'])

    # Train the model
    for task in range(n_tasks):
        print("Training task: ", task  + 1)
        
        x = torch.Tensor(tasks[task][0].reshape(-1, 32*32*3)).float()
        y = torch.Tensor(tasks[task][1]).long()
        
        if args.cuda:
            x, y = x.cuda(), y.cuda()
    
        for epoch in range(args.n_epochs):
            for j in range(0, len(tasks[task][0]), args.batch_size):
                current_data = x[j: j + args.batch_size]
                current_labels = y[j: j + args.batch_size]
                model.train()
                model.observe(current_data, task, current_labels)
            print("Epoch: ", epoch)
            
            #test the model after each epoch
            correct = 0
            total = len(tasks[task][0])
            for j in range(0,len(tasks[task][0]), test_bs):
                current_data = x[j: j + test_bs]
                current_labels = y[j: j + test_bs]
                output = model.forward(current_data, task)
                pred = output.data.max(1)[1]
                correct += (pred == current_labels).sum().item()
            print("Accuracy: ", correct / total)
            if correct / total > 0.85:
                break
            #   output loss only


    # Test the model after training
        for i in range(0, len(tests), 2):
            correct = 0
            total = len(tests[i])     

            # Test the model
            print("Testing task: " , i // 2 + 1)
            
            x = torch.Tensor(tests[i].reshape(-1, 32*32*3)).float()
            y = torch.Tensor(tests[i+1]).long()
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            for j in range(0,len(tests[i]), test_bs):
                current_data = x[j: j + test_bs]
                current_labels = y[j: j + test_bs]
                output = model.forward(current_data, i // 2)
                pred = output.data.max(1)[1]
                correct += (pred == current_labels).sum().item()
            print("Accuracy: ", correct / total)
            test_accuracies.append(correct / total)
            
        # # test task 2
        # correct = 0
        # total = len(test_data_per_class_2)
        
        # x = torch.Tensor(test_data_per_class_2.reshape(-1, 32*32*3)).float()
        # y = torch.Tensor(test_labels_per_class_2).long()
        # if args.cuda:
        #     x, y = x.cuda(), y.cuda()
        
        # print("Testing task: 2")
        # for j in range(0,len(test_data_per_class_2), test_bs):
        #     current_data = x[j: j + test_bs]
        #     current_labels = y[j: j + test_bs]
        #     output = model.forward(current_data, 1)
        #     pred = output.data.max(1)[1]
        #     correct += (pred == current_labels).sum().item()
        # print("Accuracy: ", correct / total)
        # test_accuracies.append(correct / total)
        
    return model, test_accuracies

        
#model, test_accuracies = run_cifar(Args())
def train_next_task(args, model, tasknum, n_inputs=N_INPUTS, n_outputs=N_OUTPUTS, n_tasks=N_TASKS, size_of_task=SIZE_OF_TASKS, newclasses = SHUFFLEDCLASSES):
    
    test_accuracies = []
    test_bs = 1000
    # Load data
    train_data, train_labels, test_data, test_labels = load_cifar10_data(DATASET_PATH)
    ## NEWCLASSES = suffled CLASSES
    NEWCLASSES = newclasses
    print("new ordering of classes: ", NEWCLASSES)
    oldClasstoNewClass = {}
    for i in range(len(CLASSES)):
        oldClasstoNewClass[i] = NEWCLASSES.index(CLASSES[i])
    for i in range(len(train_labels)):
        train_labels[i] = oldClasstoNewClass[train_labels[i]]
    for i in range(len(test_labels)):
        test_labels[i] = oldClasstoNewClass[test_labels[i]]

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
        elif PRETRAIN > 0:
            task_data, task_labels = split_into_classes(train_data, train_labels, NEWCLASSES[PRETRAIN + size_of_task * (i-1) : PRETRAIN + size_of_task * (i)], NEWCLASSES)
            tasks.append([task_data, task_labels])
            partition_test_data, partition_test_labels = split_into_classes(test_data, test_labels, NEWCLASSES[PRETRAIN + size_of_task * (i-1) : PRETRAIN + size_of_task * (i)], NEWCLASSES)
            tests.append(partition_test_data)
            tests.append(partition_test_labels)
        else:
            task_data, task_labels = split_into_classes(train_data, train_labels, NEWCLASSES[size_of_task * i : size_of_task * (i + 1)] , NEWCLASSES)
            tasks.append([task_data, task_labels])
            partition_test_data, partition_test_labels = split_into_classes(test_data, test_labels, NEWCLASSES[size_of_task * i : size_of_task * (i + 1)] , NEWCLASSES)
            tests.append(partition_test_data)
            tests.append(partition_test_labels)
        
    # task1data, task1labels = split_into_classes(train_data, train_labels, ['airplane', 'automobile', 'bird', 'cat', 'deer']) ## pre train on this 
    # task2data, task2labels = split_into_classes(train_data, train_labels, ['dog', 'frog', 'horse', 'ship', 'truck']) ## split this guy up into 5 tasks 
    # # tasks = [[task1data, task1labels], [task2data,task2labels]]
    # test_data_per_class_1, test_labels_per_class_1 = split_into_classes(test_data, test_labels, ['airplane', 'automobile', 'bird', 'cat', 'deer'])
    # test_data_per_class_2, test_labels_per_class_2 = split_into_classes(test_data, test_labels, ['dog', 'frog', 'horse', 'ship', 'truck'])

    # Train the model
    task = tasknum - 1
    print("Training task: ", task  + 1)
    
    x = torch.Tensor(tasks[task][0].reshape(-1, 32*32*3)).float()
    y = torch.Tensor(tasks[task][1]).long()
    
    if args.cuda:
        x, y = x.cuda(), y.cuda()

    for epoch in range(args.n_epochs):
        for j in range(0, len(tasks[task][0]), args.batch_size):
            current_data = x[j: j + args.batch_size]
            current_labels = y[j: j + args.batch_size]
            model.train()
            model.observe(current_data, task, current_labels)
        print("Epoch: ", epoch)
        
        #test the model after each epoch
        correct = 0
        total = len(tasks[task][0])
        for j in range(0,len(tasks[task][0]), test_bs):
            current_data = x[j: j + test_bs]
            current_labels = y[j: j + test_bs]
            output = model.forward(current_data, task)
            pred = output.data.max(1)[1]
            correct += (pred == current_labels).sum().item()
        print("Accuracy: ", correct / total)
        if correct / total > 0.85:
            break
        #   output loss only


# Test the model after training
    for i in range(0, len(tests), 2):
        correct = 0
        total = len(tests[i])     

        # Test the model
        print("Testing task: " , i // 2 + 1)
        
        x = torch.Tensor(tests[i].reshape(-1, 32*32*3)).float()
        y = torch.Tensor(tests[i+1]).long()
        if args.cuda:
            x, y = x.cuda(), y.cuda()
        for j in range(0,len(tests[i]), test_bs):
            current_data = x[j: j + test_bs]
            current_labels = y[j: j + test_bs]
            output = model.forward(current_data, i // 2)
            pred = output.data.max(1)[1]
            correct += (pred == current_labels).sum().item()
        print("Accuracy: ", correct / total)
        test_accuracies.append(correct / total)
            
        
    return model, test_accuracies


def plot_task_accuracies(test_accuracies, num_tasks):
    for t in range(num_tasks):
        # Calculate the indices for the current set of accuracies
        start_idx = t * num_tasks
        end_idx = start_idx + num_tasks
        # Extract accuracies for all tasks after training task t+1
        accuracies = test_accuracies[start_idx:end_idx]
        # Create labels for the tasks
        task_labels = [f'Task {i+1}' for i in range(num_tasks)]
        # Plot the accuracies
        plt.bar(task_labels, [acc * 100 for acc in accuracies])

        # Calculate the y-value for the horizontal line
        y_value = 100 / SIZE_OF_TASKS  # or use SIZE_OF_TASKS if it's defined separately

        # Add horizontal line across the bar chart
        plt.axhline(y=y_value, color='red', linestyle='--', label=f'y = {y_value:.2f}')
        
        plt.title(f'Test Set Accuracy for All Tasks (Post Task {t+1} Training)')
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Tasks')
        plt.legend()
        plt.show()


## Call the function
#plot_task_accuracies(test_accuracies, N_TASKS)
#
#test_accuracies_1 = test_accuracies
#
#task_1_accuracies_AGEM = []
#task_1_accuracies_AGEM.append(0.5)
#for i in range(0, len(test_accuracies_1), 5):
#    task_1_accuracies_AGEM.append(test_accuracies_1[i])


def runbatch(runs):
    max_runs = runs
    test_accuracies_AGEM_all = []
    test_accuracies_GEM_all = []

    for i in range(max_runs):
        print(f"Run {i+1}")
        random.shuffle(SHUFFLEDCLASSES)
        AGEM = True
        model, test_accuracies_AGEM = run_cifar(Args())
        print(test_accuracies_AGEM)
        test_accuracies_AGEM_all.append(test_accuracies_AGEM)
        AGEM = False
        model, test_accuracies_GEM = run_cifar(Args())
        print(test_accuracies_GEM)
        test_accuracies_GEM_all.append(test_accuracies_GEM)
        
    average_GEM_accuracies = np.mean(test_accuracies_GEM_all, axis=0)
    average_AGEM_accuracies = np.mean(test_accuracies_AGEM_all, axis=0)


    task_1_accuracies_GEM = []
    task_1_accuracies_GEM.append(0.5)
    for i in range(0, len(average_GEM_accuracies), N_TASKS):
        task_1_accuracies_GEM.append(average_GEM_accuracies[i])
    task_1_accuracies_AGEM = []
    task_1_accuracies_AGEM.append(0.5)
    for i in range(0, len(average_AGEM_accuracies), N_TASKS):
        task_1_accuracies_AGEM.append(average_AGEM_accuracies[i])
        

    plt.plot(task_1_accuracies_AGEM, label='AGEM')
    plt.plot(task_1_accuracies_GEM, label='GEM')
    plt.title('Task 1 Test Set Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Tasks')
    plt.legend()
    plt.show()
        

    plot_task_accuracies(average_AGEM_accuracies, N_TASKS)
    plot_task_accuracies(average_GEM_accuracies, N_TASKS)