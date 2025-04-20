import torch
import torch.nn as nn
import dgl.nn as dglnn

class GCNEncoder(nn.Module):
    def __init__(self, in_feats=128, hidden_feats=128, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GraphConv(in_feats, hidden_feats))
        for _ in range(1, num_layers - 1):
            self.layers.append(dglnn.GraphConv(hidden_feats, hidden_feats))

    def forward(self, g, node_feats):
        h = node_feats
        for layer in self.layers:
            h = torch.relu(layer(g, h))
        return h