import os
import sys

# run command to check oython version
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
from cifar import load_cifar10_data, split_into_classes, get_class_indexes, load_data
import cifar
from torch.utils.data import DataLoader
import random

import sys
sys.path.append(os.path.abspath("."))  # Adds the current directory
# from GEM.gem import *
from GEM.args import *
from torch.nn.functional import relu, avg_pool2d
import torch.nn as nn
import quadprog



# we provide some top level initial parameters depending on if we want to work in cifar-10 or cifar-100

ALL_TASK_UNLEARN_ACCURACIES = []
ALL_TASK_UNLEARN_CONFIDENCES = []

ALL_TASK_CONTINUAL_LEARNING_ACCURACIES = []

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
CLASSES = cifar.CLASSES
SHUFFLEDCLASSES = CLASSES.copy()
CONFIDENCE_SAMPLES = 5
if DATASET == 'cifar-10':
    CLASSES = cifar.CLASSES
    CLASSES = CLASSES.copy()
elif DATASET == 'cifar-100':
    CLASSES = cifar.CLASSES_100_UNORDERED
    SHUFFLEDCLASSES = CLASSES.copy()

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

def NegAGEM(gradient, memories, margin=0.5, eps=1e-3):
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
    """
    gref = memories.t().double().mean(axis=0).cuda() # * margin
    g = gradient.contiguous().view(-1).double().cuda()
    
    dot_prod = torch.dot(g, gref)
    dot_prod = dot_prod/(torch.dot(gref, gref))
    
    dot_prod = abs(dot_prod.item())
    
    memories_np *= dot_prod
    gradient_np *= 0.5
    """
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))
    
def project2neggrad2(gradient, memories, alpha = 0.9):
    gref = memories.t().double().sum(axis=0).cuda() # * margin
    g = gradient.contiguous().view(-1).double().cuda()
    x = gref*alpha + g * (1-alpha)
    gradient.copy_(torch.Tensor(x).view(-1, 1))
    

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

def agemprojection(gradient, gradient_memory, margin=0.5, eps=1e-5):
    """
    Projection of gradients for A-GEM with the memory approach
    Use averaged gradient memory for projection
    
    input:  gradient, g-reference
    output: gradient, g-projected
    """

    gref = gradient_memory.t().double().mean(axis=0).cuda() # * margin
    g = gradient.contiguous().view(-1).double().cuda()

    dot_prod = torch.dot(g, gref)
    
    #if dot_prod < 0:
    #    x = g
    #    gradient.copy_(torch.Tensor(x).view(-1, 1))
    #    return
    
    # avoid division by zero
    dot_prod = dot_prod/(torch.dot(gref, gref))
    
    # epsvector = torch.Tensor([eps]).cuda()
    
    x = g*0.5 + gref * abs(dot_prod)  # + epsvector
    gradient.copy_(torch.Tensor(x).view(-1, 1))
    
def replay(gradient, gradient_memory):
    """
    Adds the gradients of the current task to the memory 
    
    input:  gradient, g-reference
    output: gradient, g-projected
    """
    g = gradient_memory.t().double().sum(axis=0).cuda()
    gref = gradient.contiguous().view(-1).double().cuda()
    # simply add the gradients
    x = g + gref
    gradient.copy_(torch.Tensor(x).view(-1, 1))
    
def naiveretraining(gradient):
    """
    Naive retraining of the model on the current task
    
    input:  gradient, g-reference
    output: gradient, g-projected
    """
    g = gradient.t().double().mean(axis=0).cuda()
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
        self.unlearn_memory_strength = args.unlearn_mem_strength
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

    def forward(self, x, t):
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

    def observe(self, algorithm, x, t, y):
        # update memory
        if t != self.old_task or t not in self.observed_tasks:
            self.observed_tasks.append(t)
            self.old_task = t
            
        val = 0
        if t == 0:
            val = max(PRETRAIN,1)
        else:
            val = 1
        # Update ring buffer storing examples from current task
        bsz = y.data.size(0)
        if (algorithm == 'NAIVE'):
            self.zero_grad()
            loss = self.ce(self.forward(x, t), y)
            loss.backward()
            self.opt.step()
            return
        
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
        if len(self.observed_tasks) > 0: ### CHANGED FROM 1 to 0 SINCE WE PRETRAIN ON FST 5 CLASSES 
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
        if len(self.observed_tasks) > 0: ### CHANGED FROM 1 to 0 SINCE WE PRETRAIN ON FST 5 CLASSES 
            if algorithm == 'AGEM':
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
            elif algorithm == 'GEM':
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
            elif algorithm == 'REPLAY':
                store_grad(self.parameters, self.grads, self.grad_dims, t)
                indx = torch.cuda.LongTensor(self.observed_tasks[:-1]) if self.gpu \
                    else torch.LongTensor(self.observed_tasks[:-1])
                replay(self.grads[:, t].unsqueeze(1), self.grads.index_select(1, indx))
                # copy gradients back
                overwrite_grad(self.parameters, self.grads[:, t],
                            self.grad_dims)
        self.opt.step()


    def unlearn(self, algorithm, t, x1, x2, alpha = 0.9):
        
        ## first check if task t has been learned
        if t not in self.observed_tasks:
            print("Task , ", t, " has not been learned yet - No change")
            return

        ## now check if the task is not the first task learned 
        if len(self.observed_tasks) == 1:
            print("Only one task has been learned - resetting the model")
            self.reset()
            return
            
        if algorithm == 'neggem':
            ## otherwise we need to unlearn the task 
            ## we compute the gradients of all learnt tasks
            current_grad = []
            
            for param in self.parameters():
                if param.grad is not None:
                    current_grad.append(param.grad.data.view(-1))
            current_grad = torch.cat(current_grad).unsqueeze(1)
            
            # now find the grads of the previous tasks
            for tt in range(t + 1):
                self.zero_grad()
                past_task = self.observed_tasks[tt]
                offset1, offset2 = compute_offsets(past_task, self.nc_per_task,
                                                    self.is_cifar)
                ptloss = self.ce(
                        self.forward(
                            self.memory_data[past_task],
                            past_task)[:, offset1: offset2],
                        self.memory_labs[past_task] - offset1)
                if tt == t:
                    ptloss = self.ce(
                        self.forward(
                            self.memory_data[past_task][x1:x2],
                            past_task)[:, offset1: offset2],
                        self.memory_labs[past_task][x1:x2] - offset1)
                    
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims,
                            past_task)
            ## so now, we have the gradients of all the tasks we can now do our projection,
            ## first we check if it is even neccessary to do so, if not simply do a optimiser.step()
        
            forget_grads = self.grads[:, t].unsqueeze(1)
            retain_indices = torch.tensor([i for i in range(self.grads.size(1)) if i in self.observed_tasks and i < t], device=self.grads.device)
            retain_grads = self.grads.index_select(1, retain_indices)
            
            self.zero_grad()
            loss = self.ce(
                        self.forward(self.memory_data[past_task][x1:x2], past_task)[:, offset1: offset2], self.memory_labs[past_task][x1:x2] - offset1)
            # negate loss for unlearning
            loss = -1 * loss
            loss.backward() 

            dotp = torch.mm(self.grads[:, t].unsqueeze(0) * -1, retain_grads)
            if (dotp < 0).sum() != 0:
                forget_grads *= -1
                NegAGEM(forget_grads, retain_grads, self.unlearn_memory_strength)
                if args.salun:
                    apply_Salun(forget_grads, args.salun_threshold)

                overwrite_grad(self.parameters, forget_grads, self.grad_dims)
            
            self.opt.step()
        elif algorithm == "neggrad":
            """
            use the project2neggrad2 function to project the gradient to unlearn the task
            unlike the previous method, we do not perform this in batches
            """
            # now find the grads of the previous tasks
            for tt in range(t + 1):
                self.zero_grad()
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
            forget_grads = self.grads[:, t].unsqueeze(1)
            retain_indices = torch.tensor([i for i in range(self.grads.size(1)) if i in self.observed_tasks and i < t], device=self.grads.device)
            retain_grads = self.grads.index_select(1, retain_indices)
            
            self.zero_grad()
            
            forget_grads *= -1
            project2neggrad2(forget_grads, retain_grads, alpha)
            overwrite_grad(self.parameters, forget_grads, self.grad_dims)
            self.opt.step()
        else:
            print("Invalid Algorithm")

def apply_Salun(gradient, threshold):
    '''
    Applies "SalUn-like" filtering to a raw gradient tensor by keeping only the largest 
    fraction of gradient values (by absolute value) and setting the others to zero.
    
    For example, threshold=0.1 means keep only the top 10% largest gradient magnitudes.
    '''
    # Flatten the gradient for easier manipulation
    grad = gradient.view(-1)
    grad_abs = torch.abs(grad)
    
    if threshold <= 0.0:
        grad.zero_()
    elif threshold >= 1.0:
        # No filtering, keep all gradients
        return gradient
    else:
        # Find the cutoff for the top threshold fraction
        cutoff = torch.quantile(grad_abs, 1 - threshold)
        # Zero out gradients smaller than the cutoff
        mask = grad_abs >= cutoff
        grad.mul_(mask)
    
    return grad.view(gradient.shape)

def project2cone2_neggrad_dual(gradient, forget_memories, retain_memories,
                                margin=0.5, eps=1e-10):
    """
    Dual QP approach to enforce:
    1) f_i^T x <= -margin for each forget_mem f_i
    2) r_j^T x >=  margin for each retain_mem r_j
    while minimising ||x - g||^2 in L2.

    This solves in O((f+r)^3) time rather than O(p^3), making it much more
    efficient if (f+r) << p.

    Args:
        gradient (torch.Tensor): shape (p,) or (p,1), the original gradient g.
        forget_memories (torch.Tensor): shape (f, p) of "forget" vectors f_i.
        retain_memories (torch.Tensor): shape (r, p) of "retain" vectors r_j.
        margin (float): margin for dot-products.
        eps (float): small constant for numerical stability in the matrix P.

    Returns:
        None. The 'gradient' tensor is updated in place to the projected x.
    """
    
    try:
        # ---- 1) Prepare data as NumPy arrays ----
        g = gradient.detach().cpu().contiguous().view(-1).double().numpy()  # shape (p,)
        F = forget_memories.detach().cpu().double().numpy()  # shape (f, p)
        R = retain_memories.detach().cpu().double().numpy()  # shape (r, p)

        f_count = F.shape[0]
        r_count = R.shape[0]
        

        # ---- 2) Build the matrix A of shape (f+r, p). 
        # Rows 0..f-1 => -F[i], rows f..f+r-1 => R[j].
        A_forget = -1 * F  # shape (f, p)
        A_retain =  R  # shape (r, p)
        A = np.concatenate([A_forget, A_retain], axis=0)  # shape ((f+r), p)

        P = A @ A.T  # shape (f+r, f+r)
        # Add eps * I for numerical stability
        P = 0.5 * (P + P.T) + eps*np.eye(f_count + r_count)

        n_vars = f_count + r_count  # dimension of lambda
        n_cons = 2*(f_count + r_count)

        # G is shape (n_vars, n_cons). We'll fill columns one by one (since quadprog uses G^T x >= h).
        G = np.zeros((n_vars, n_cons), dtype=np.double)
        h = np.zeros(n_cons, dtype=np.double)

        # (a) positivity constraints
        for i in range(n_vars):
            # e_i^T lambda >= 0 => \lambda_i >= 0
            # That means G[:, c] is e_i => G[i, c] = 1
            G[i, i] = 1.0
            h[i] = 0.0

        # (b) dot-product constraints
        for i in range(n_vars):
            # constraint index c in the second block
            c = n_vars + i

            a_i = A[i, :]          # shape (p,)
            b_i = margin - np.dot(a_i, g)

            G[:, c] = A @ a_i      # shape (n_vars,)

            # h[c] = b_i
            h[c] = b_i

        q = np.zeros(n_vars, dtype=np.double)
        sol, f_val, _, _, _, _ = quadprog.solve_qp(P, q, G, h, meq=0)

        # ---- 6) Reconstruct x = g + A^T lambda ----
        x_star = g + A.T @ sol  # shape (p,)

        # ---- 7) Copy back into 'gradient' ----
        x_torch = torch.from_numpy(x_star).reshape(-1, 1).to(gradient.device)
        gradient.copy_(x_torch)
    except:
        margin /= 2

def eval_task(model, args, test_data, test_labels, task, test_bs=1000):
    """
    Evaluate model on a set of tasks and return accuracy and average confidence.
    """
    model.eval()
    total = 0
    correct = 0
    confidence_sum = 0.0
    for i in range(0, len(test_data), test_bs):
        current_data = torch.Tensor(test_data.reshape(-1, 32*32*3)).float()
        current_labels = torch.Tensor(test_labels).long()
        if args.cuda:
            current_data, current_labels = current_data.cuda(), current_labels.cuda()
        output = model.forward(current_data, task)
        pred = output.data.max(1)[1]
        correct += (pred == current_labels).sum().item()
        total += current_labels.size(0)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence_sum += probabilities.max(1)[0].sum().item()
    accuracy = correct / total
    avg_confidence = confidence_sum / total
    return accuracy, avg_confidence

def eval_retain_forget_test(model, args, retain_set, forget_set, test_set, retain_acc, forget_acc, test_acc, test_acc_forget):
    test_bs = 1000
    correct_retain = 0
    total_retain = 0
    for i in range(0, len(retain_set)):
        correct = 0
        total = len(retain_set[i])     

        # Test the model
        
        x = torch.Tensor(retain_set[i][0].reshape(-1, 32*32*3)).float()
        y = torch.Tensor(retain_set[i][1]).long()
        if args.cuda:
            x, y = x.cuda(), y.cuda()
        model.eval()
        correct = 0
        total = len(retain_set[i][0])
        for j in range(0,len(retain_set[i][0]), test_bs):
            current_data = x[j: j + test_bs]
            current_labels = y[j: j + test_bs]
            output = model.forward(current_data, i)
            pred = output.data.max(1)[1]
            correct += (pred == current_labels).sum().item()
        correct_retain += correct
        total_retain += total
    print("Total correct retain: ", correct_retain, " Total retain: ", total_retain)
    
    retain_acc.append(correct_retain / total_retain)
    
    correct_forget = 0
    total_forget = 0
    for i in range(0, len(forget_set)):
        correct = 0
        total = len(forget_set[i])     

        # Test the model
        
        x = torch.Tensor(forget_set[i][0].reshape(-1, 32*32*3)).float()
        y = torch.Tensor(forget_set[i][1]).long()
        if args.cuda:
            x, y = x.cuda(), y.cuda()
        model.eval()
        total = len(forget_set[i][0])
        for j in range(0,len(forget_set[i][0]), test_bs):
            current_data = x[j: j + test_bs]
            current_labels = y[j: j + test_bs]
            output = model.forward(current_data, i + len(retain_set))
            pred = output.data.max(1)[1]
            correct += (pred == current_labels).sum().item()
        correct_forget += correct
        total_forget += total
    print("Total correct forget: ", correct_forget, " Total forget: ", total_forget)
    
    forget_acc.append(correct_forget / total_forget)
    
    correct_test = 0
    total_test = 0
    
    for i in range(0, 2 * len(retain_set), 2):
        correct = 0
        total = len(test_set[i])     

        # Test the model
        
        x = torch.Tensor(test_set[i].reshape(-1, 32*32*3)).float()
        y = torch.Tensor(test_set[i+1]).long()
        if args.cuda:
            x, y = x.cuda(), y.cuda()
        model.eval()
        average_confidence_task = []
        # keep track of average confidence score
        for j in range(0,len(test_set[i]), test_bs):
            current_data = x[j: j + test_bs]
            current_labels = y[j: j + test_bs]
            output = model.forward(current_data, i // 2)
            pred = output.data.max(1)[1]
            correct += (pred == current_labels).sum().item()
        correct_test += correct
        total_test += total
    print("Total correct test: ", correct_test, " Total test: ", total_test)
        
    test_acc.append(correct_test/total_test)
    
    correct_test_forget = 0
    total_test_forget = 0
    for i in range(2 * len(retain_set), 2 * (len(retain_set) + len(forget_set)),  2):
        correct = 0
        total = len(test_set[i])     

        # Test the model
        
        x = torch.Tensor(test_set[i].reshape(-1, 32*32*3)).float()
        y = torch.Tensor(test_set[i+1]).long()
        if args.cuda:
            x, y = x.cuda(), y.cuda()
        model.eval()
        average_confidence_task = []
        # keep track of average confidence score
        for j in range(0,len(test_set[i]), test_bs):
            current_data = x[j: j + test_bs]
            current_labels = y[j: j + test_bs]
            output = model.forward(current_data, i // 2)
            pred = output.data.max(1)[1]
            correct += (pred == current_labels).sum().item()
        correct_test_forget += correct
        total_test_forget += total
        
    test_acc_forget.append(correct_test_forget/total_test_forget)
    
    print("Total correct forget test: ", correct_test_forget, " Total test forget: ", total_test_forget)
    return retain_acc, forget_acc, test_acc, test_acc_forget

def run_cifar(algorithm, args, n_inputs=N_INPUTS, n_outputs=N_OUTPUTS, n_tasks=N_TASKS, size_of_task=SIZE_OF_TASKS, newclasses = SHUFFLEDCLASSES):
    # Set up the model
    model = Net(n_inputs, n_outputs, n_tasks, args)
    if args.cuda:
        model.cuda()
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
    
        for epoch in range(args.n_epochs):
            for j in range(0, len(tasks[task][0]), args.batch_size):
                current_data = x[j: j + args.batch_size]
                current_labels = y[j: j + args.batch_size]
                model.train()
                model.observe(algorithm, current_data, task, current_labels)
            
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
        # Test the model after training
        temp_test_accuracies = []
        for task in range(n_tasks):
            test, _ = eval_task(model, args,
                                        tests[task * 2], tests[task * 2 + 1],
                                        task)
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
    
    return model, test_accuracies, average_confidence , after_unlearn_accuracies, confidence_after_unlearn, retain_accuracies, forget_accuracies, testing_accuracies, testing_accuracies_forget

# sample usage
# model, test_accuracies = run_cifar(Args())

test_accuracies_GEM_all_last_iter = []
unlearn_accuracies_GEM_all_last_iter = []

retain_accuracies_all = []
forget_accuracies_all = []
testing_accuracies_all = []
testing_accuracies_forget_all = []

iternumb = 0
while len(test_accuracies_GEM_all_last_iter) < int(sys.argv[3]):
    iternumb += 1
    torch.cuda.empty_cache()
    random.shuffle(SHUFFLEDCLASSES)
    
    args = Args()
    args.unlearn_memory_strength = float(sys.argv[1])
    args.unlearn_batch_size = int(sys.argv[2])
    
    use_salun = int(sys.argv[4])
    if use_salun == 1:
        args.salun = True
        args.salun_threshold = float(sys.argv[5])
    else:
        args.salun = False

    model, test_accuracies_GEM, confidence , after_unlearn_acc, after_unlearn_conf, retain_accuracies, forget_accuracies, testing_accuracies, testing_accuracies_forget = run_cifar('GEM', args)

    test_accuracies_GEM_all_last_iter.append(test_accuracies_GEM[-20:])
    unlearn_accuracies_GEM_all_last_iter.append(after_unlearn_acc)
    retain_accuracies_all.append(retain_accuracies)
    forget_accuracies_all.append(forget_accuracies)
    testing_accuracies_all.append(testing_accuracies)
    testing_accuracies_forget_all.append(testing_accuracies_forget)

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
plt.plot(x, testing_accuracies_avg, label="Retain Test Accuracies", linestyle=":")
plt.plot(x, testing_accuracies_forget_avg, label="Forget Test Accuracies", linestyle="--")
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
plt.plot(x, avg_forget_cont_unlearning_accuracies, label="Forget Cont. Unlearn Test Accuracies", linestyle="-.")
plt.plot(x, avg_retain_cont_learning_accuracies, label="Retain Cont. Learn Test Accuracies (Baseline)", linestyle=":")
plt.plot(x, avg_forget_cont_learning_accuracies, label="Forget Cont. Learn Test Accuracies (Baseline)", linestyle="--")
plt.xticks(x, x_labels)
plt.ylim(0, 1)
plt.xlabel("Task")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('RetainForgetContLearningUnlearning.png')
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
    plt.plot(x, testing_accuracies_avg, label="Retain Test Accuracies", linestyle=":")
    plt.plot(x, testing_accuracies_forget_avg, label="Forget Test Accuracies", linestyle="--")
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
df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(args.unlearn_memory_strength) + 'BatchSize' + str(args.unlearn_batch_size) + 'TestAccuracies.csv')

df = pd.DataFrame(unlearn_accuracies_GEM_all_last_iter)
df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(args.unlearn_memory_strength) + 'BatchSize' + str(args.unlearn_batch_size) + 'UnlearnAccuracies.csv')

df = pd.DataFrame(retain_accuracies_all)
df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(args.unlearn_memory_strength) + 'BatchSize' + str(args.unlearn_batch_size) + 'RetainAccuracies.csv')

df = pd.DataFrame(forget_accuracies_all)
df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(args.unlearn_memory_strength) + 'BatchSize' + str(args.unlearn_batch_size) + 'ForgetAccuracies.csv')

df = pd.DataFrame(testing_accuracies_all)
df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(args.unlearn_memory_strength) + 'BatchSize' + str(args.unlearn_batch_size) + 'TestingAccuracies.csv')

df = pd.DataFrame(testing_accuracies_forget_all)
df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(args.unlearn_memory_strength) + 'BatchSize' + str(args.unlearn_batch_size) + 'TestingAccuraciesForget.csv')

df = pd.DataFrame(ALL_TASK_UNLEARN_ACCURACIES)
df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(args.unlearn_memory_strength) + 'BatchSize' + str(args.unlearn_batch_size) + 'AllTaskUnlearnAccuracies.csv')

df = pd.DataFrame(ALL_TASK_UNLEARN_CONFIDENCES)
df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(args.unlearn_memory_strength) + 'BatchSize' + str(args.unlearn_batch_size) + 'AllTaskUnlearnConfidences.csv')

df = pd.DataFrame(ALL_TASK_CONTINUAL_LEARNING_ACCURACIES)
df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(args.unlearn_memory_strength) + 'BatchSize' + str(args.unlearn_batch_size) + 'AllTaskContinualLearningAccuracies.csv')

df = pd.DataFrame(avg_cont_learning_accuracies)
df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(args.unlearn_memory_strength) + 'BatchSize' + str(args.unlearn_batch_size) + 'AvgContLearningAccuracies.csv')

df = pd.DataFrame(avg_cont_unlearning_accuracies)
df.to_csv('Results' + str(cur_date) + 'MemoryStrength' + str(args.unlearn_memory_strength) + 'BatchSize' + str(args.unlearn_batch_size) + 'AvgContUnlearningAccuracies.csv')

# save the model
#torch.save(model, 'Results' + str(cur_date) + 'MemoryStrength' + str(args.unlearn_memory_strength) + 'BatchSize' + str(args.unlearn_batch_size) + 'Model.pt')

# change the directory to /dcs/large/u2145461/cs407/Machine-Unlearning-x-Continual-Learning-neggem
os.chdir('/dcs/large/u2145461/cs407/Machine-Unlearning-x-Continual-Learning-neggem')

os.mkdir('Results' + str(cur_date) + 'MemoryStrength' + str(args.unlearn_memory_strength) + 'BatchSize' + str(args.unlearn_batch_size))

# save the results in the folder by using the mv command to move all PNG files to the folder
os.system('mv -f *.png ./Results' + str(cur_date) + 'MemoryStrength' + str(args.unlearn_memory_strength) + 'BatchSize' + str(args.unlearn_batch_size))

# save the results in the folder by using the mv command to move all PNG files to the folder
os.system('mv -f *.csv ./Results' + str(cur_date) + 'MemoryStrength' + str(args.unlearn_memory_strength) + 'BatchSize' + str(args.unlearn_batch_size))

os.system('mv -f *.pt ./Results' + str(cur_date) + 'MemoryStrength' + str(args.unlearn_memory_strength) + 'BatchSize' + str(args.unlearn_batch_size))
