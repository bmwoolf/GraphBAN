import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch

class GAETeacher(nn.Module):
    def __init__(self, in_feats=128, hidden_feats=128):
        super().__init__()
        self.encoder = dglnn.SAGEConv(in_feats, hidden_feats, aggregator_type='mean')
        self.decoder = nn.Linear(hidden_feats, in_feats)

    def forward(self, g, node_feats):
        z = torch.relu(self.encoder(g, node_feats))
        recon = self.decoder(z)
        return z, recon