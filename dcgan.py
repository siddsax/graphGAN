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
# os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=20000, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
parser.add_argument('--numV', type=int, default=4, help='number of max vertices per sample')
parser.add_argument('--lm', type=str, default="", help='number of max vertices per sample')
parser.add_argument('--sm', type=str, default="save", help='number of max vertices per sample')
opt = parser.parse_args()
print(opt)

opt.sm = opt.sm + "_" + str(np.random.randint(low=0, high=10000))
cuda = True if torch.cuda.is_available() else False
typ = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = GeneratorFT2()
discriminator = DiscriminatorFT2(opt.numV)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

dataloader = torch.utils.data.DataLoader(gr_dataset(), batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

init = 0
if len(opt.lm):
    generator, optimizer_G, init = load_model(generator, "saved_models/" + opt.lm + "_G.pt", optimizer=optimizer_G)
    discriminator, optimizer_D, _ = load_model(discriminator, "saved_models/" + opt.lm + "_D.pt", optimizer=optimizer_D)
    opt.sm = opt.lm

# xrandFix = Variable(Tensor(np.random.rand(2, 4, 2)))
xrandFix = Variable(Tensor(np.array(
            [[[0.6094, 0.5296],
            [0.0501, 0.0959],
            [0.2523, 0.1996],
            [0.5192, 0.0039]]]
            )))

for epoch in range(init, opt.n_epochs):
    for i, (adj, x) in enumerate(dataloader):

        # imgs = imgs.type(torch.FloatTensor)[:,0,:]

        valid = Variable(Tensor(x.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(x.shape[0], 1).fill_(0.0), requires_grad=False)

        adj = adj.type(typ)
        x = x.type(typ)
        # # Configure input
        # real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        xrand = Variable(Tensor(np.random.rand(x.shape[0], x.shape[1], x.shape[2])))
        # # Generate a batch of images
        xgen = generator(xrand, adj)
        # # Loss measures generator's ability to fool the discriminator

        g_loss = adversarial_loss(discriminator(xgen, adj).squeeze(), valid.squeeze())

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # # Measure discriminator's ability to classify real from generated samples
        a = discriminator(x, adj).squeeze()
        b = discriminator(xgen.detach(), adj).squeeze()
        
        frac = (a.sum() + b.sum())/(a.shape[0] + b.shape[0])
        
        real_loss = adversarial_loss(a, valid.squeeze())
        fake_loss = adversarial_loss(b, fake.squeeze())
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                            d_loss.item(), g_loss.item()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:

            print(frac.data.numpy())
            generator.eval()
            xgen = generator(xrandFix, adj[:1])
            generator.train()
            print(xgen[0])
            drawRec(adj[0].detach().cpu(), xgen[0].detach().cpu(), name="DrawGN")   
            save_model(generator, optimizer_G, epoch, opt.sm + "_G.pt") 
            save_model(discriminator, optimizer_D, epoch, opt.sm + "_D.pt") 
        # save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True

