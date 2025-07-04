import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from .eval import *
from .salun import *
from .util import *

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
        self.salun = args.salun
        self.salun_threshold = args.salun_threshold
        self.net = ResNet18(n_outputs)

        self.algorithm = args.algorithm 
        self.alpha = args.alpha

        self.ce = nn.CrossEntropyLoss()
        self.n_outputs = n_outputs

        self.opt = torch.optim.SGD(self.parameters(), args.lr)

        self.gpu = args.cuda
        
        self.n_learn_memories = args.n_learn_memories
        self.n_unlearn_memories = args.n_unlearn_memories
        
        self.constraint_violation_count = 0

        """
        Allocate episodic memory
        n_tasks: number of tasks
        n_memories: number of memories per task
        n_inputs: number of input features
        """

        # allocate episodic memory
        self.learn_memory_data = torch.FloatTensor(n_tasks, self.n_learn_memories, n_inputs)
        self.learn_memory_labs = torch.LongTensor(n_tasks, self.n_learn_memories)
        if args.cuda:
            self.learn_memory_data = self.learn_memory_data.cuda()
            self.learn_memory_labs = self.learn_memory_labs.cuda()

        if args.mem_unlearning_buffer:
            self.unlearn_memory_data = torch.FloatTensor(n_tasks, self.n_unlearn_memories, n_inputs)
            self.unlearn_memory_labs = torch.LongTensor(n_tasks, self.n_unlearn_memories)
            if args.cuda:
                self.unlearn_memory_data = self.unlearn_memory_data.cuda()
                self.unlearn_memory_labs = self.unlearn_memory_labs.cuda()
        else:
            # reuse the learn buffer if it exists
            self.unlearn_memory_data = self.learn_memory_data
            self.unlearn_memory_labs = self.learn_memory_labs

        self.mem_learning_buffer = args.mem_learning_buffer
        self.mem_unlearning_buffer = args.mem_unlearning_buffer

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
        
        if not self.mem_learning_buffer:
            endcnt = min(self.mem_cnt + bsz, self.n_learn_memories) #256
            effbsz = endcnt - self.mem_cnt # 256
            self.learn_memory_data[t, self.mem_cnt: endcnt].copy_(
                x.data[: effbsz])
            if bsz == 1:
                self.learn_memory_labs[t, self.mem_cnt] = y.data[0]
            else:
                self.learn_memory_labs[t, self.mem_cnt: endcnt].copy_(
                    y.data[: effbsz])
            self.mem_cnt += effbsz
            if self.mem_cnt == self.n_learn_memories:
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
                        self.learn_memory_data[past_task],
                        past_task)[:, offset1: offset2],
                    self.learn_memory_labs[past_task] - offset1)
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims,
                        past_task)

        # now compute the grad on the current minibatch
        self.zero_grad()
        
        # for all grads NOT in self.observed tasks, set to 0
        for tt in range(20):
            if tt not in self.observed_tasks:
                self.grads[:, tt].fill_(0.0)

        offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar) 
        loss = self.ce(self.forward(x, t)[:, offset1: offset2], y - offset1)
        loss.backward()

        # check if gradient violates constraints
        if len(self.observed_tasks) > 0: ### CHANGED FROM 1 to 0 SINCE WE PRETRAIN ON FST 5 CLASSES 
            if algorithm == 'AGEM':
                store_grad(self.parameters, self.grads, self.grad_dims, t)
                for tt in range(20):
                    if tt not in self.observed_tasks:
                        self.grads[:, tt].fill_(0.0)
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
                # for all grads NOT in self.observed tasks, set to 0
                for tt in range(20):
                    if tt not in self.observed_tasks:
                        self.grads[:, tt].fill_(0.0)
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
            elif algorithm == "ALT-GEM":
                # enforces that learning the new task is guarenteed to be learnt
                store_grad(self.parameters, self.grads, self.grad_dims, t)
                indx = torch.cuda.LongTensor(self.observed_tasks) if self.gpu \
                    else torch.LongTensor(self.observed_tasks)
                project2cone2(self.grads[:, t].unsqueeze(1),
                            self.grads.index_select(1, indx), self.margin)
                # copy gradients back
                overwrite_grad(self.parameters, self.grads[:, t],
                            self.grad_dims)
        # for all grads NOT in self.observed tasks, set to 0
        for tt in range(20):
            if tt not in self.observed_tasks:
                self.grads[:, tt].fill_(0.0)
        self.opt.step()


    def unlearn(self, algorithm, t, x1, x2, alpha = 0.9, mask = None):
        ## first check if task t has been learned
        if t not in self.observed_tasks:
            print("Task , ", t, " has not been learned yet - No change")
            return

        ## now check if the task is not the first task learned 
        if len(self.observed_tasks) == 1:
            print("Only one task has been learned - resetting the model")
            self.reset()
            return
        
        # for all grads NOT in self.observed tasks, set to 0
        for tt in range(20):
            if tt not in self.observed_tasks:
                self.grads[:, tt].fill_(0.0)
            
        if algorithm == 'neggem':
            
            # now find the grads of the previous tasks
            for tt in self.observed_tasks:
                self.zero_grad()
                past_task = tt
                offset1, offset2 = compute_offsets(past_task, self.nc_per_task,
                                                    self.is_cifar)
                ptloss = self.ce(
                        self.forward(
                            self.learn_memory_data[past_task],
                            past_task)[:, offset1: offset2],
                        self.learn_memory_labs[past_task] - offset1)
                if past_task == t:
                    ptloss = self.ce(
                        self.forward(
                            self.learn_memory_data[past_task][x1:x2],
                            past_task)[:, offset1: offset2],
                        self.learn_memory_labs[past_task][x1:x2] - offset1)
                    
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims,
                            past_task)
            ## so now, we have the gradients of all the tasks we can now do our projection,
            ## first we check if it is even neccessary to do so, if not simply do a optimiser.step()
            
            self.zero_grad()
            offset1,offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
            loss = self.ce(
                        self.forward(self.unlearn_memory_data[t][x1:x2],
                            t)[:, offset1: offset2], self.unlearn_memory_labs[t][x1:x2] - offset1)
            loss += self.ce(
                        self.forward(self.learn_memory_data[t][x1:x2],
                            t)[:, offset1: offset2], self.learn_memory_labs[t][x1:x2] - offset1)
            # negate loss for unlearning
            loss = -5 * loss
            loss.backward()
            
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            indx = torch.cuda.LongTensor([i for i in self.observed_tasks if i != t]) if self.gpu \
                else torch.LongTensor([i for i in self.observed_tasks if i != t])
                
            forget_grads = self.grads[:, t].unsqueeze(1)
            retain_indices = torch.tensor([i for i in range(self.grads.size(1)) if i in self.observed_tasks and i != t], device=self.grads.device)
            retain_grads = self.grads.index_select(1, retain_indices)
            
            if self.salun:
                mask = apply_salun(forget_grads, self.salun_threshold, mask=mask)
                
            self.grads[:, t] = forget_grads.squeeze(1)

            dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                                self.grads.index_select(1, indx))

            if (dotp < 0).sum() != 0:
                self.constraint_violation_count += 1
                project2cone2(forget_grads, retain_grads, self.unlearn_memory_strength)
                self.grads[:, t] = forget_grads.squeeze(1)
                # copy gradients back
                overwrite_grad(self.parameters, self.grads[:, t],
                            self.grad_dims)
            
            self.opt.step()
            ### now we do RL-GEM right after!!!
            for tt in self.observed_tasks:
                self.zero_grad()
                past_task = tt
                offset1, offset2 = compute_offsets(past_task, self.nc_per_task,
                                                    self.is_cifar)
                ptloss = self.ce(
                        self.forward(
                            self.learn_memory_data[past_task],
                            past_task)[:, offset1: offset2],
                        self.learn_memory_labs[past_task] - offset1)
                if past_task == t:
                    continue
                    current_labs = (self.learn_memory_labs[past_task][x1:x2] - offset1)
                    random.shuffle(current_labs)    
                    ptloss = self.ce(
                        self.forward(
                            self.learn_memory_data[past_task][x1:x2],
                            past_task)[:, offset1: offset2],
                        current_labs)
                    
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims,
                            past_task)
            ## so now, we have the gradients of all the tasks we can now do our projection,
            ## first we check if it is even neccessary to do so, if not simply do a optimiser.step()
            
            self.zero_grad()
            offset1,offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
            current_labs = (self.learn_memory_labs[t][x1:x2] - offset1)
            random.shuffle(current_labs)
            loss = self.ce(
                        self.forward(self.learn_memory_data[past_task][x1:x2], t)[:, offset1: offset2], current_labs)
            current_labs = (self.unlearn_memory_labs[t][x1:x2] - offset1)
            random.shuffle(current_labs)
            loss += self.ce(
                        self.forward(self.unlearn_memory_data[past_task][x1:x2], t)[:, offset1: offset2], current_labs)
            # no need to multiply loss, use random labelling
            loss = 5 * loss
            loss.backward()
            
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            indx = torch.cuda.LongTensor([i for i in self.observed_tasks if i != t]) if self.gpu \
                else torch.LongTensor([i for i in self.observed_tasks if i != t])
                
            forget_grads = self.grads[:, t].unsqueeze(1)
            retain_indices = torch.tensor([i for i in range(self.grads.size(1)) if i in self.observed_tasks and i != t], device=self.grads.device)
            retain_grads = self.grads.index_select(1, retain_indices)

            dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                                self.grads.index_select(1, indx))
            if self.salun:
                mask = apply_salun(forget_grads, self.salun_threshold, mask=mask)
            if (dotp < 0).sum() != 0:
                self.constraint_violation_count += 1
                NegGEM(forget_grads, retain_grads, self.unlearn_memory_strength)
                self.grads[:, t] = forget_grads.squeeze(1)
                # copy gradients back
                overwrite_grad(self.parameters, self.grads[:, t],
                            self.grad_dims)
            self.opt.step()
            
        elif algorithm == 'negagem':
            # now find the grads of the previous tasks
            for tt in self.observed_tasks:
                self.zero_grad()
                past_task = tt
                offset1, offset2 = compute_offsets(past_task, self.nc_per_task,
                                                    self.is_cifar)
                ptloss = self.ce(
                        self.forward(
                            self.learn_memory_data[past_task],
                            past_task)[:, offset1: offset2],
                        self.unlearn_memory_labs[past_task] - offset1)
                if past_task == t:
                    continue
                    ptloss = self.ce(
                        self.forward(
                            self.learn_memory_data[past_task][x1:x2],
                            past_task)[:, offset1: offset2],
                        self.unlearn_memory_labs[past_task][x1:x2] - offset1)
                    
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims,
                            past_task)
            ## so now, we have the gradients of all the tasks we can now do our projection,
            ## first we check if it is even neccessary to do so, if not simply do a optimiser.step()
            
            self.zero_grad()
            offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
            loss = self.ce(
                        self.forward(self.unlearn_memory_data[t][x1:x2], t)[:, offset1: offset2], self.unlearn_memory_labs[t][x1:x2] - offset1)
            loss += self.ce(
                        self.forward(self.learn_memory_data[t][x1:x2], t)[:, offset1: offset2], self.learn_memory_labs[t][x1:x2] - offset1)
            # negate loss for unlearning
            loss = -1 * loss
            loss.backward()
            
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            # indx = torch.cuda.LongTensor([i for i in self.observed_tasks if i != t]) if self.gpu \
            #     else torch.LongTensor([i for i in self.observed_tasks if i != t])
                
            forget_grads = self.grads[:, t].unsqueeze(1)
            retain_indices = torch.tensor([i for i in range(self.grads.size(1)) if i in self.observed_tasks and i != t], device=self.grads.device)
            retain_grads = self.grads.index_select(1, retain_indices)

            dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                                self.grads.index_select(1, retain_indices))
            if self.salun:
                mask = apply_salun(forget_grads, self.salun_threshold, mask=mask)
            if (dotp < 0).sum() != 0:
                self.constraint_violation_count += 1
                agemprojection(forget_grads, retain_grads)
                self.grads[:, t] = forget_grads.squeeze(1)
                # copy gradients back
                overwrite_grad(self.parameters, self.grads[:, t],
                            self.grad_dims)
            
            self.opt.step()
            
        elif algorithm == 'RL-GEM':
            # otherwise we need to unlearn the task 
            ## we compute the gradients of all learnt tasks
            
            # now find the grads of the previous tasks
            for tt in self.observed_tasks:
                self.zero_grad()
                past_task = tt
                offset1, offset2 = compute_offsets(past_task, self.nc_per_task,
                                                    self.is_cifar)
                ptloss = self.ce(
                        self.forward(
                            self.learn_memory_data[past_task],
                            past_task)[:, offset1: offset2],
                        self.learn_memory_labs[past_task] - offset1)
                if past_task == t:
                    continue
                    current_labs = (self.learn_memory_labs[past_task][x1:x2] - offset1)
                    random.shuffle(current_labs)    
                    ptloss = self.ce(
                        self.forward(
                            self.learn_memory_data[past_task][x1:x2],
                            past_task)[:, offset1: offset2],
                        current_labs)
                    
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims,
                            past_task)
            ## so now, we have the gradients of all the tasks we can now do our projection,
            ## first we check if it is even neccessary to do so, if not simply do a optimiser.step()
            
            self.zero_grad()
            offset1,offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
            current_labs = (self.learn_memory_labs[t][x1:x2] - offset1)
            random.shuffle(current_labs)
            loss = self.ce(
                        self.forward(self.learn_memory_data[past_task][x1:x2], t)[:, offset1: offset2], current_labs)
            current_labs = (self.unlearn_memory_labs[t][x1:x2] - offset1)
            random.shuffle(current_labs)
            loss += self.ce(
                        self.forward(self.unlearn_memory_data[past_task][x1:x2], t)[:, offset1: offset2], current_labs)
            # negate loss for unlearning
            loss = 5 * loss
            loss.backward()
            
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            indx = torch.cuda.LongTensor([i for i in self.observed_tasks if i != t]) if self.gpu \
                else torch.LongTensor([i for i in self.observed_tasks if i != t])
                
            forget_grads = self.grads[:, t].unsqueeze(1)
            retain_indices = torch.tensor([i for i in range(self.grads.size(1)) if i in self.observed_tasks and i != t], device=self.grads.device)
            retain_grads = self.grads.index_select(1, retain_indices)
            
            if self.salun:
                mask = apply_salun(forget_grads, self.salun_threshold, mask=mask)
                
            self.grads[:, t] = forget_grads.squeeze(1)

            dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                                self.grads.index_select(1, indx))

            if (dotp < 0).sum() != 0:
                self.constraint_violation_count += 1
                NegGEM(forget_grads, retain_grads, self.unlearn_memory_strength)
                self.grads[:, t] = forget_grads.squeeze(1)
                # copy gradients back
                overwrite_grad(self.parameters, self.grads[:, t],
                            self.grad_dims)
            self.opt.step()
        elif algorithm == 'RL-AGEM':
            # now find the grads of the previous tasks
            for tt in self.observed_tasks:
                self.zero_grad()
                past_task = tt
                offset1, offset2 = compute_offsets(past_task, self.nc_per_task,
                                                    self.is_cifar)
                ptloss = self.ce(
                        self.forward(
                            self.learn_memory_data[past_task],
                            past_task)[:, offset1: offset2],
                        self.learn_memory_labs[past_task] - offset1)
                if past_task == t:
                    continue
                    current_labs = (self.learn_memory_labs[past_task][x1:x2] - offset1)
                    random.shuffle(current_labs)
                    ptloss = self.ce(
                        self.forward(
                            self.learn_memory_data[past_task][x1:x2],
                            past_task)[:, offset1: offset2],
                        current_labs)
                    
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims,
                            past_task)
            ## so now, we have the gradients of all the tasks we can now do our projection,
            ## first we check if it is even neccessary to do so, if not simply do a optimiser.step()
            
            self.zero_grad()
            offset1,offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
            current_labs = (self.unlearn_memory_labs[t][x1:x2] - offset1)
            random.shuffle(current_labs)
            loss = self.ce(
                        self.forward(self.unlearn_memory_data[t][x1:x2], t)[:, offset1: offset2], current_labs)
            current_labs = (self.learn_memory_labs[t][x1:x2] - offset1)
            random.shuffle(current_labs)
            loss += self.ce(
                        self.forward(self.learn_memory_data[t][x1:x2], t)[:, offset1: offset2], current_labs)
            # negate loss for unlearning
            loss = 5 * loss
            loss.backward()
            
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            indx = torch.cuda.LongTensor([i for i in self.observed_tasks if i != t]) if self.gpu \
                else torch.LongTensor([i for i in self.observed_tasks if i != t])
                
            forget_grads = self.grads[:, t].unsqueeze(1)
            retain_indices = torch.tensor([i for i in range(self.grads.size(1)) if i in self.observed_tasks and i != t], device=self.grads.device)
            retain_grads = self.grads.index_select(1, retain_indices)

            dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                                self.grads.index_select(1, indx))
            
            if self.salun:
                mask = apply_salun(forget_grads, self.salun_threshold, mask=mask)

            if (dotp < 0).sum() != 0:
                agemprojection(forget_grads, retain_grads)
                self.grads[:, t] = forget_grads.squeeze(1)
                # copy gradients back
                overwrite_grad(self.parameters, self.grads[:, t],
                            self.grad_dims)
            self.opt.step()
        elif algorithm == 'ALT-NEGGEM':
            # now find the grads of the previous tasks
            for tt in self.observed_tasks:
                self.zero_grad()
                past_task = tt
                offset1, offset2 = compute_offsets(past_task, self.nc_per_task,
                                                    self.is_cifar)
                ptloss = self.ce(
                        self.forward(
                            self.learn_memory_data[past_task],
                            past_task)[:, offset1: offset2],
                        self.learn_memory_labs[past_task] - offset1)
                if past_task == t:
                    ptloss = self.ce(
                        self.forward(
                            self.learn_memory_data[past_task][x1:x2],
                            past_task)[:, offset1: offset2],
                        self.learn_memory_labs[past_task][x1:x2] - offset1)
                    
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims,
                            past_task)
            ## so now, we have the gradients of all the tasks we can now do our projection,
            ## first we check if it is even neccessary to do so, if not simply do a optimiser.step()
            
            self.zero_grad()
            offset1,offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
            loss = self.ce(
                        self.forward(self.unlearn_memory_data[t][x1:x2], t)[:, offset1: offset2], self.unlearn_memory_labs[t][x1:x2] - offset1)
            loss += self.ce(
                        self.forward(self.learn_memory_data[t][x1:x2], t)[:, offset1: offset2], self.learn_memory_labs[t][x1:x2] - offset1)
            # negate loss for unlearning
            loss = -1 * loss
            loss.backward()
            
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            indx = torch.cuda.LongTensor([i for i in self.observed_tasks if i != t]) if self.gpu \
                else torch.LongTensor([i for i in self.observed_tasks if i != t])
                
            forget_grads = self.grads[:, t].unsqueeze(1)
            retain_indices = torch.tensor([i for i in range(self.grads.size(1)) if i in self.observed_tasks and i != t], device=self.grads.device)
            retain_grads = self.grads.index_select(1, retain_indices)

            all_indices = torch.tensor([i for i in range(self.grads.size(1)) if i in self.observed_tasks], device=self.grads.device)
            all_grads = self.grads.index_select(1, all_indices)

            average_grad_of_retain = torch.mean(retain_grads, dim=1, keepdim=True)

            dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                                self.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                if self.salun:
                    mask = apply_salun(forget_grads, self.salun_threshold, mask=mask)
                NegGEM(forget_grads, all_grads, self.unlearn_memory_strength)
                self.grads[:, t] = forget_grads.squeeze(1)
                # copy gradients back
                overwrite_grad(self.parameters, self.grads[:, t],
                            self.grad_dims)
            
            self.opt.step()
        elif algorithm == 'neggrad':
            """
            use the project2neggrad2 function to project the gradient to unlearn the task
            unlike the previous method, we should not perform this in batches
            """
            # now find the grads of the previous tasks
            for tt in self.observed_tasks:
                self.zero_grad()
                past_task = tt
                offset1, offset2 = compute_offsets(past_task, self.nc_per_task,
                                                    self.is_cifar)
                ptloss = self.ce(
                        self.forward(
                            self.learn_memory_data[past_task],
                            past_task)[:, offset1: offset2],
                        self.learn_memory_labs[past_task] - offset1)
                if past_task == t:
                    ptloss = self.ce(
                        self.forward(
                            self.learn_memory_data[past_task][x1:x2],
                            past_task)[:, offset1: offset2],
                        self.learn_memory_labs[past_task][x1:x2] - offset1)
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims,
                            past_task)
            
            offset1,offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
            loss = self.ce(
                        self.forward(self.unlearn_memory_data[t][x1:x2], t)[:, offset1: offset2], self.unlearn_memory_labs[t][x1:x2] - offset1)
            loss += self.ce(
                        self.forward(self.learn_memory_data[t][x1:x2], t)[:, offset1: offset2], self.learn_memory_labs[t][x1:x2] - offset1)
            # negate loss for unlearning
            loss = -1 * loss
            loss.backward()
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            indx = torch.cuda.LongTensor([i for i in self.observed_tasks if i != t]) if self.gpu \
                else torch.LongTensor([i for i in self.observed_tasks if i != t])
                
            forget_grads = self.grads[:, t].unsqueeze(1)
            retain_indices = torch.tensor([i for i in range(self.grads.size(1)) if i in self.observed_tasks and i != t], device=self.grads.device)
            retain_grads = self.grads.index_select(1, retain_indices)
            
            if self.salun:
                mask = apply_salun(forget_grads, self.salun_threshold, mask=mask)
            
            project2neggrad2(forget_grads, retain_grads, alpha = self.alpha)
            self.grads[:, t] = forget_grads.squeeze(1)
            overwrite_grad(self.parameters, self.grads[:, t], self.grad_dims)
            self.opt.step()
        else:
            print("Invalid Algorithm")
        return mask
            
    def update_memory_from_dataset(self, x_all, y_all, t, t_mapping, mem_scores, mem_data, mem_split, mem_type, buffer_type):
        """
        Update the episodic memory for task t using the entire training set for that task.
        Computes the memorization score for each example and selects the top (or bottom)
        self.n_{learn/unlearn}_memories examples based on the "selection" criteria.
        """
        print(f"[Memory Update] Task {t}, Buffer: {buffer_type}, Type: {mem_type}, Split: {mem_split}")
        if buffer_type == 'learn':
            target_data = self.learn_memory_data
            target_labs = self.learn_memory_labs
            n_memories = self.n_learn_memories
        elif buffer_type == 'unlearn':
            target_data = self.unlearn_memory_data
            target_labs = self.unlearn_memory_labs
            n_memories = self.n_unlearn_memories

        task_orig_indices = []
        classes_in_task = t_mapping[t]
        for cls in classes_in_task:
            task_orig_indices.extend(mem_data[f"tr_classidx_{cls}"])
        task_orig_indices = task_orig_indices[:len(x_all)]
        task_orig_indices = np.array(task_orig_indices)

        mem_scores = torch.tensor(mem_scores[task_orig_indices])
        N = mem_scores.size(0)
        
        selection_mode = mem_type
        
        # Sort indices by memorization score.
        if selection_mode == "most":
            _, sorted_indices = torch.sort(mem_scores, descending=True)
        elif selection_mode == "least":
            _, sorted_indices = torch.sort(mem_scores, descending=False)
        else:
            arr = [i for i in range(len(mem_scores))]
            random.shuffle(arr)
            sorted_indices = torch.tensor(arr)

        arr = [i for i in range(len(mem_scores))]
        random.shuffle(arr)
        
        num_to_select = min(n_memories, N)

        if mem_split > 1 or mem_split < 0:
            raise ValueError("split must be in the range [0,1]")

        mem_selection = round(num_to_select * mem_split)

        selected_indices = sorted_indices[:mem_selection].tolist()
        selected_indices = arr[:(num_to_select-mem_selection)] + selected_indices
        random.shuffle(selected_indices)
        
        # Update the memory for task t with the selected examples.
        with torch.no_grad():
            target_data[t, :num_to_select].copy_(x_all[selected_indices])
            target_labs[t, :num_to_select].copy_(y_all[selected_indices])
