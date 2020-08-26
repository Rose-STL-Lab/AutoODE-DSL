import math
from dgl.nn import GraphConv, GatedGraphConv, GATConv
import dgl
import torch
import torch.nn as nn

class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input
    
class GCN_Layer(nn.Module):
    def __init__(self, in_dim, out_dim, is_final_layer = False):
        super(GCN_Layer, self).__init__()
        self.gcn_conv = GraphConv(in_dim, out_dim, norm='both')#
        self.activation = nn.LeakyReLU()
        self.is_final_layer = is_final_layer
    
    def forward(self, g, xx):
        g = dgl.transform.remove_self_loop(g)
        g = dgl.transform.add_self_loop(g)
        if self.is_final_layer:
            return self.gcn_conv(g, xx)
        else:
            out = self.gcn_conv(g, xx)
            out = self.activation(out)
            return g, out
        
class GAT_Layer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads = 4, is_final_layer = False):
        super(GAT_Layer, self).__init__()
        self.gat = GATConv(in_dim, out_dim, num_heads=num_heads)
        self.batchnorm = nn.BatchNorm1d(out_dim * num_heads)
        self.activation = nn.LeakyReLU()
        self.is_final_layer = is_final_layer
    
    def forward(self, g, xx):
        if self.is_final_layer:
            
            return torch.stack([self.gat(g, xx[i]) for i in range(xx.shape[0])])
        else:
            out = torch.stack([self.gat(g, xx[i]) for i in range(xx.shape[0])])
            out = self.activation(self.batchnorm(out.flatten(2).transpose(1,2)).transpose(1,2))
            return g, out

class GAT(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_heads, num_layer = 5):
        super(GAT, self).__init__()
        self.model = [GAT_Layer(in_dim, hidden_dim)]
        self.model += [GAT_Layer(hidden_dim*num_heads, hidden_dim) for i in range(num_layer-2)]
        self.output_layer = nn.Linear(hidden_dim*num_heads, out_dim)
        self.model = mySequential(*self.model)

    def forward(self, g, xx, output_length):
        outputs = []
        for i in range(output_length):
            out = self.output_layer(self.model(g, xx)[1])
            xx = torch.cat([xx[:,:,3:], out], dim = -1)
            outputs.append(out.unsqueeze(2))
        return torch.cat(outputs, dim = 2)

    
class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layer = 5):
        super(GCN, self).__init__()
        self.model = [GCN_Layer(in_dim, hidden_dim)]
        self.model += [GCN_Layer(hidden_dim, hidden_dim) for i in range(num_layer-2)]
        self.model += [GCN_Layer(hidden_dim, out_dim, is_final_layer = True)]
        self.model = mySequential(*self.model)

    def forward(self, g, xx, output_length):
        xx = xx.transpose(0,1)
        outputs = []
        for i in range(output_length):
            out = self.model(g, xx)
            xx = torch.cat([xx[:,:,3:], out], dim = -1)
            outputs.append(out.unsqueeze(2))
        return torch.cat(outputs, dim = 2).transpose(0,1)