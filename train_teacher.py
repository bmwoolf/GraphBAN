import torch
import torch.nn.functional as F
from models.gae_teacher import GAETeacher
from utils.device_utils import move_to_device

def train_teacher(g, node_feats, epochs=250):
    """Train teacher model."""
    device = g.device
    model = GAETeacher().to(device)
    node_feats = node_feats.detach()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        z, recon = model(g, node_feats)
        loss = F.binary_cross_entropy_with_logits(recon, node_feats)
        
        loss.backward(retain_graph=True)
        optimizer.step()

    return model