# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import subprocess
import pickle
import torch
import os

cifar_10_path = "cifar-10-python.tar.gz"

subprocess.call("tar xzfv cifar-10-python.tar.gz", shell=True)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cifar10_train_1 = unpickle('cifar-10-python/data_batch_1')
cifar10_train_2 = unpickle('cifar-10-python/data_batch_2')
cifar10_train_3 = unpickle('cifar-10-python/data_batch_3')
cifar10_train_4 = unpickle('cifar-10-python/data_batch_4')
cifar10_train_5 = unpickle('cifar-10-python/data_batch_5')
cifar10_test = unpickle('cifar-10-python/test_batch')

cifar10_train = cifar10_train_1 + cifar10_train_2 + cifar10_train_3 + cifar10_train_4 + cifar10_train_5

x_tr = torch.from_numpy(cifar10_train[b'data'])
y_tr = torch.LongTensor(cifar10_train[b'fine_labels'])
x_te = torch.from_numpy(cifar10_test[b'data'])
y_te = torch.LongTensor(cifar10_test[b'fine_labels'])

torch.save((x_tr, y_tr, x_te, y_te), 'cifar10.pt')
