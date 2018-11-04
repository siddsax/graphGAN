import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.x = np.array([[1.3,3], [31,23], [11,3], [1,13], [12,3]])

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = self.x[index, :]
        X = torch.from_numpy(x.reshape((1, x.shape[-1])))#torch.load('data/' + ID + '.pt')
        return X