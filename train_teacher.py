import torch
import torch.nn.functional as F
from models.gae_teacher import GAETeacher

def train_teacher(g, node_feats, epochs=250):
    model = GAETeacher()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        z, recon = model(g, node_feats)
        loss = F.binary_cross_entropy_with_logits(recon, node_feats)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model