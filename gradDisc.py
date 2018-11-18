from __future__ import division
from __future__ import print_function
# import logging as lg
# lg.basicConfig(level=lg.INFO, format='%(levelname)-8s: %(message)s')

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
from models import *
from gr_dataset import *

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=20000, help='number of epochs of training')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
parser.add_argument('--numV', type=int, default=4, help='number of max vertices per sample')
parser.add_argument('--lm', type=str, default="", help='number of max vertices per sample')
opt = parser.parse_args()
print(opt)


discriminator = DiscriminatorFT2(opt.numV)
discriminator = load_model(discriminator, "saved_models/" + opt.lm + "_D.pt")
for param in discriminator.parameters():
    param.requires_grad = False

## input data
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# xrand = Variable(Tensor(np.random.rand(1, 4, 2)))
# print(xrand)
# exit()
xrand = Variable(Tensor(np.array(
         [[[0.7230, 0.7867],
         [0.4211, 0.5665],
         [0.4227, 0.5379],
         [0.9821, 0.7725]]]
            )))
xrand.requires_grad_(True)

adj = Variable(torch.FloatTensor(
  [0, 1, 1, 0,
   1, 0, 0, 1,
   1, 0, 0, 1,
   0, 1, 1, 0]
)).reshape((1, -1, 4))


for i in range(1000):

    y = discriminator(xrand, adj)

    L = 1 - torch.abs(y.sum(1)).sum() # objective
    # L.backward()

    # setup for updating params
    # params = f.parameters()
    # p_updater = torch.optim.SGD(params, lr=.001)

    ## update params
    # p_updater.step()
    # lg.info(f.w)
    # lg.info(f.b)

    g = torch.autograd.grad(L, xrand, retain_graph=True)[0]
    # import pdb
    # pdb.set_trace()
    xrand = xrand - g

    if i % 100 == 0:
        print(L)
        print(g)
        print(xrand)
drawRec(adj[0].detach().cpu(), xrand[0].detach().cpu(), name="DrawGrad")   
