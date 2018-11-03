import torch
from generator import *
import numpy as np
gen = Generator(5)
disc = Discriminator(5)

a = np.array([[1.3,3], [31,23], [11,3], [1,13], [12,3]])
x = torch.autograd.Variable(torch.from_numpy(a)).type(torch.FloatTensor)

k = gen(x)
p = disc(x, k)
print(k)

print(p)