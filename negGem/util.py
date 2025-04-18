import numpy as np
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d
import torch
import quadprog

############### RESNET ###############

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

############### GEM ###############

PRETRAIN = 0
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

def NegGEM(gradient, memories, margin=0.5, eps=1e-3):
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
    
def NegAGEM(gradient, memories):
    """
    Projection of gradients for A-GEM with the memory approach
    Use averaged gradient memory for projection
    
    input:  gradient, g-reference
    output: gradient, g-projected
    """

    gref = memories.t().double().sum(axis=0).cuda() # * margin
    g = gradient.contiguous().view(-1).double().cuda()

    dot_prod = torch.dot(g, gref)
    
    #if dot_prod < 0:
    #    x = g
    #    gradient.copy_(torch.Tensor(x).view(-1, 1))
    #    return
    
    # avoid division by zero
    dot_prod = dot_prod/(torch.dot(gref, gref))
    
    # epsvector = torch.Tensor([eps]).cuda()
    
    x = g + gref * abs(dot_prod)  # + epsvector
    gradient.copy_(torch.Tensor(x).view(-1, 1))
    
def project2neggrad2(gradient, memories, alpha = 0.5):
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

    gref = gradient_memory.t().double().sum(axis=0).cuda() # * margin
    g = gradient.contiguous().view(-1).double().cuda()

    dot_prod = torch.dot(g, gref)
    
    #if dot_prod < 0:
    #    x = g
    #    gradient.copy_(torch.Tensor(x).view(-1, 1))
    #    return
    
    # avoid division by zero
    dot_prod = dot_prod/(torch.dot(gref, gref))
    
    # epsvector = torch.Tensor([eps]).cuda()
    
    x = g + gref * abs(dot_prod)  # + epsvector
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
