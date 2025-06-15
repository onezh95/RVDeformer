import math

import torch
import torch.nn as nn

from torch_geometric.nn import SAGEConv

from RVDeformer.geo import adj_mat2list



class GNNLayer(nn.Module):
    
    def __init__(self, layer_name, in_channels, out_channels, adj_mat, **kwargs):
        super(GNNLayer, self).__init__()
        if layer_name == 'graph_sage':
            self.gcn_conv = SAGEConv(in_channels, out_channels, **kwargs)
        else:
            raise NotImplementedError('Not implemented gnn model: ', layer_name)

        self.layer_name = layer_name
        self.edge_index = adj_mat2list(adj_mat)
        
        
    def forward(self, x):
        
        out_batch = []
        
        for sample in x:
            out = self.gcn_conv(sample, self.edge_index)
            out_batch.append(out)
        
        out_batch = torch.stack(out_batch)

        return out_batch
    