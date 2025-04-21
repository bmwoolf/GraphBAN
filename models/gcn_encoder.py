import torch
import torch.nn as nn
import dgl.nn as dglnn
from typing import Tuple

class GCNEncoder(nn.Module):
    def __init__(self, in_feats: int = 128, hidden_feats: int = 128, num_layers: int = 3):
        """Graph Convolutional Network encoder for molecular graphs.
        
        Args:
            in_feats: Input feature dimension
            hidden_feats: Hidden layer dimension
            num_layers: Number of GCN layers
            
        Shapes:
            - Input[0] g: DGLGraph with N nodes
            - Input[1] node_feats: (N, in_feats)
            - Output: (batch_size, hidden_feats)
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GraphConv(in_feats, hidden_feats))
        for _ in range(1, num_layers - 1):
            self.layers.append(dglnn.GraphConv(hidden_feats, hidden_feats))
        # Add global pooling
        self.pool = dglnn.AvgPooling()
        
        # Expected shapes for shape checking
        self.expected_node_feats = (-1, in_feats)  # (N, in_feats)
        self.expected_output = (-1, hidden_feats)  # (B, hidden_feats)

    def forward(self, g: 'dgl.DGLGraph', node_feats: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GCN encoder.
        
        Args:
            g: Input graph
            node_feats: Node features tensor of shape (N, in_feats)
            
        Returns:
            torch.Tensor: Graph features tensor of shape (B, hidden_feats)
        """
        # Shape checking
        assert node_feats.shape[1] == self.expected_node_feats[1], \
            f"Expected node features of shape (N, {self.expected_node_feats[1]}), got {node_feats.shape}"
            
        h = node_feats
        for layer in self.layers:
            h = torch.relu(layer(g, h))
            
        # Pool and check output shape
        out = self.pool(g, h)
        assert out.shape[1] == self.expected_output[1], \
            f"Expected output shape (B, {self.expected_output[1]}), got {out.shape}"
            
        return out  # This will give one vector per graph