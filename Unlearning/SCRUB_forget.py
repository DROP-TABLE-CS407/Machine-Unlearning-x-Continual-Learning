import copy
import os
import time
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader

import arg_parser
from surgical_plugins.cluster import get_distance, get_fs_dist_only
from unlearn.impl import wandb_init
import utils
import evaluation
import unlearn
from trainer import validate


def main():
    start_rte = time.time()
    args = arg_parser.parse_args()

    args.wandb_group_name = f"{args.arch}-{args.dataset}-{args.unlearn}"
    logger = wandb_init(args)

    # Device setup
    device = torch.device(f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu")
    args.save_dir = f'assets/unlearn/{args.unlearn}'
    os.makedirs(args.save_dir, exist_ok=True)

    # Dataset and model setup for CIFAR10
    if args.dataset == "cifar10":
        model, train_loader_full, val_loader, test_loader, marked_loader, _ = utils.setup_model_dataset(args)
        model.cuda()

    def replace_loader_dataset(dataset, batch_size=args.batch_size, seed=1, shuffle=True):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True,
                                           shuffle=shuffle)

    # Preparing datasets for SCRUB
    forget_dataset = copy.deepcopy(marked_loader.dataset)
    retain_dataset = copy.deepcopy(marked_loader.dataset)

    # Filtering CIFAR10 dataset indices for SCRUB-specific forget and retain sets
    num_clusters, n_group, n_sample = 10, 15, args.num_indexes_to_replace
    model.eval()

    distances_matrix, _, _ = get_distance(train_loader_full, model, args, num_clusters=num_clusters)
    _, _, l_des_idx, _ = get_fs_dist_only(distances_matrix, train_loader_full, n_group=n_group, n_sample=n_sample,
                                          group_index=3)
    forget_dataset_indices = l_des_idx
    retain_dataset_indices = set(range(len(train_loader_full.dataset))) - set(forget_dataset_indices)

    # Setting up DataLoaders for SCRUB
    forget_dataset = torch.utils.data.Subset(train_loader_full.dataset, list(forget_dataset_indices))
    retain_dataset = torch.utils.data.Subset(train_loader_full.dataset, list(retain_dataset_indices))
    forget_loader = replace_loader_dataset(forget_dataset, seed=args.seed)
    retain_loader = replace_loader_dataset(retain_dataset, seed=args.seed)

    unlearn_data_loaders = OrderedDict(retain=retain_loader, forget=forget_loader, val=val_loader, test=test_loader)

    criterion = nn.CrossEntropyLoss()

    # Load initial model checkpoint
    args.mask = f'assets/checkpoints/0{args.dataset}_original_{args.arch}_bs256_lr0.1_seed{args.seed}_epochs{args.epochs}.pth.tar'
    checkpoint = torch.load(args.mask, map_location=device)
    if "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=False)

    # Applying SCRUB on CIFAR10 dataset
    start_unlearn = time.time()
    unlearn_method = unlearn.get_unlearn_method("SCRUB")
    model_s = copy.deepcopy(model)
    model_t = copy.deepcopy(model)
    module_list = nn.ModuleList([model_s, model_t])
    unlearn_method(unlearn_data_loaders, module_list, criterion, args)
    model = module_list[0]
    end_rte = time.time()


    # Evaluation of results after SCRUB
    logger.log({
        'unlearn_time': end_rte - start_unlearn,
        'overall_time': end_rte - start_rte
    })

    accuracy = {}
    for name, loader in unlearn_data_loaders.items():
        val_acc = validate(loader, model, criterion, args)
        accuracy[name] = val_acc
        print(f"{name} acc: {val_acc}")

    logger.log({'forget acc': accuracy['forget'], 'retain acc': accuracy['retain'], 'val acc': accuracy['val'],
                'test acc': accuracy['test']})


if __name__ == "__main__":
    main()