import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution
import torch
from torch.distributions.bernoulli import Bernoulli


def predAdj(x, fin=0):

    out = []
    for i in range(x.shape[0]):
        k = torch.mm(x[i].view(1, -1), x.transpose(1,0))
        if fin==1:
            k[0,i] = 0
        k = torch.nn.functional.softmax(k, -1)
        out.append(k)
    
    out = torch.cat(out, dim=0)
    return out

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class Generator(nn.Module):
    def __init__(self, numV):
        super(Generator, self).__init__()

        self.gc1 = GraphConvolution(2, 5)
        self.gc2 = GraphConvolution(5, 1)
        
        # self.predAdj = predAdj()

    def forward(self, x):
        numV = x.shape[0]
        m = Bernoulli(torch.tensor([0.5]))
        adj = m.sample(sample_shape=(numV, numV))[:,:,0]
        # import pdb
        # pdb.set_trace()
        x = F.relu(self.gc1(x, adj))
        adj = predAdj(x)

        x = F.relu(self.gc2(x, adj))

        adj = predAdj(x, fin=1)

        return adj


class Discriminator(nn.Module):
    def __init__(self, numV):
        super(Discriminator, self).__init__()

        self.gc1 = GraphConvolution(2, 5)
        self.gc2 = GraphConvolution(5, 1)
        self.linear = torch.nn.Linear(numV, 1)

    def forward(self, x, adj):

        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj).view(1, -1)
        x = self.linear(x)
        pred = torch.sigmoid(x)
        return pred


