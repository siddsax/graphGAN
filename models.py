import torch.nn as nn
import torch.nn.functional as F
from layers import *
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

class GeneratorAD(nn.Module):
    def __init__(self):
        super(GeneratorAD, self).__init__()

        self.gc1 = GraphConvolution(2, 5)
        self.gc2 = GraphConvolution(5, 1)
        
        # self.predAdj = predAdj()

    def forward(self, x):
        numV = x.shape[1]
        m = Bernoulli(torch.tensor([0.5]))
        adj = m.sample(sample_shape=(x.shape[0], numV, numV))[:,:,:,0]

        x = F.relu(self.gc1(x, adj))
        adj = predAdj(x)

        x = F.relu(self.gc2(x, adj))
        adj = predAdj(x, fin=1)

        return adj

class DiscriminatorAD(nn.Module):
    def __init__(self, numV):
        super(DiscriminatorAD, self).__init__()

        self.gc1 = GraphConvolution(2, 5)
        self.gc2 = GraphConvolution(5, 1)
        self.linear = torch.nn.Linear(numV, 1)
        torch.nn.init.xavier_uniform_(self.linear.weight)


    def forward(self, x, adj):

        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj).view(1, -1)
        x = self.linear(x)
        pred = torch.sigmoid(x)
        return pred

class GeneratorFT(nn.Module):
    def __init__(self):
        super(GeneratorFT, self).__init__()

        self.gc1 = GraphConvolution(2, 5)
        self.gc2 = GraphConvolution(5, 5)
        # self.gc3 = GraphConvolution(8, 5)
        self.gc4 = GraphConvolution(5, 2)
        
        # self.predAdj = predAdj()

    def forward(self, x, adj):
# .forwardP
        # import pdb
        # pdb.set_trace()
        x = F.relu(self.gc1.forward(x, adj))
        # import pdb
        # pdb.set_trace()
        x = F.relu(self.gc2.forward(x, adj))
                
        # import pdb
        # pdb.set_trace()
        x = self.gc4(x, adj)

        # import pdb
        # pdb.set_trace()
        return x

class DiscriminatorFT(nn.Module):
    def __init__(self, numV):
        super(DiscriminatorFT, self).__init__()

        self.gc1 = GraphConvolution(2, 4)
        self.gc2 = GraphConvolution(4, 1)
        # self.gc3 = GraphConvolution(6, 1)
        self.linear = torch.nn.Linear(numV, 1)
        torch.nn.init.xavier_uniform_(self.linear.weight)


    def forward(self, x, adj):

        x = F.relu(self.gc1(x, adj))
        # x = F.relu(self.gc2(x, adj))
        x = self.gc2(x, adj).view(x.shape[0], 1, -1)
        # import pdb
        # pdb.set_trace()
        x = self.linear(x)
        pred = torch.sigmoid(x)
        return pred
