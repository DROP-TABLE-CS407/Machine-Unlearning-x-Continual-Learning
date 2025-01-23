import Unlearning.unlearn
import torch.nn as nn
from copy import deepcopy
# Define the criterion and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

class Args:
    def __init__(self):
        self.unlearn_lr = 0.01         # Learning rate for unlearning
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.dataset = ''      # Change as needed
        self.num_classes = 10         # Number of classes in the dataset
        self.batch_size = 64
        self.print_freq = 10
        self.warmup = 0               # Number of warmup epochs
        self.imagenet_arch = False    # Set to True if using ImageNet architecture
        self.seed = 42       
        
        # SCRUB SPECIFIC
        self.kd_T = 1
        self.msteps = 1
        self.gamma = 10
        self.beta = 1

        # Neggrad SPECIFIC
        self.alpha=2

        # Add the following attributes to ensure compatibility
        self.decreasing_lr = '50,75'  # Comma-separated epochs where LR decays
        self.rewind_epoch = 0         # Epoch to rewind to; set to 0 if not using rewinding
        self.rewind_pth = ''          # Path to the rewind checkpoint
        self.gpu = 0                  # GPU ID to use; adjust as needed
        self.surgical = False         # Whether to use surgical unlearning
        self.unlearn = 'NG'      # Unlearning method, e.g., 'retrain'
        self.choice = []              # Layers to unlearn surgically; list of layer names
        self.unlearn_epochs = 10     # Number of epochs for unlearning
        self.epochs = 100   

def unlearn_wrapper(data_loaders, model, args):
    unlearn_method = Unlearning.unlearn.get_unlearn_method(args.unlearn)
    if args.unlearn == 'SCRUB':
        model_s = deepcopy(model)
        model_t = deepcopy(model)
        module_list = nn.ModuleList([model_s, model_t])
        unlearn_method(data_loaders, module_list, criterion, args)
        model = module_list[0]
    else:
        unlearn_method(data_loaders, model, criterion, args)
    return model
