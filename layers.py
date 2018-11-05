import math

import torch

from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, Ed = 0, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.Ed = Ed
        if Ed:
            self.edgeFun = nn.Sequential(nn.Linear(2*in_features, 3*in_features), nn.ReLU(), nn.Linear(3*in_features, out_features))
            for layer in self.edgeFun:
                try:
                    torch.nn.init.xavier_uniform_(layer.weight)
                except:
                    continue        
        else:
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
            if bias:
                self.bias = Parameter(torch.FloatTensor(out_features))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        if self.Ed:
            t = torch.autograd.Variable(torch.Tensor([0.0]))  # threshold
            adj = (adj > t).float() * 1
            output = []
            for i in range(adj.shape[0]):
                out = []
                for j in range(adj.shape[1]):
                    v = (input[i,j] - input[i])[adj[i,j].type(torch.ByteTensor)]
                    o = torch.sum(self.edgeFun(torch.cat([input[i,j].repeat(v.shape[0], 1), v], dim=1)), dim=0)
                    # o = torch.max(torch.cat(o, dim=0), dim=0)[0]
                    out.append(o.view(1, -1))
                out = torch.cat(out, dim=0)
                output.append(out.view(1, out.shape[0], out.shape[1]))
            
            output = torch.cat(output, dim=0)
            # import pdb
            # pdb.set_trace()
        else:

            support = torch.bmm(input, self.weight.unsqueeze(0).expand(input.size(0), *self.weight.size()))
            output = [] 
            for i in range(support.shape[0]):
                out = torch.mm(adj[i], support[i])
                if self.bias is not None:
                    out = out + self.bias
                output.append(out.view(1, out.shape[0], out.shape[1]))
            output = torch.cat(output, dim=0)
        
        return output

    def forwardP(self, input, adj):
        # try:
        support = torch.bmm(input, self.weight.unsqueeze(0).expand(input.size(0), *self.weight.size()))
        # except:

        output = [] 
        for i in range(support.shape[0]):
            out = torch.mm(adj[i], support[i])
            if self.bias is not None:
                out = out + self.bias
            output.append(out.view(1, out.shape[0], out.shape[1]))
        output = torch.cat(output, dim=0)
        
        import pdb
        pdb.set_trace()
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


