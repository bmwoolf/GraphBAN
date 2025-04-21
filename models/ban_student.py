import torch
import torch.nn as nn

class BANLayer(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.U = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.q = nn.Parameter(torch.randn(dim))

    def forward(self, h_c, h_p):
        # h_c: (B, N, D), h_p: (B, M, D)
        Uc = torch.relu(self.U(h_c))  # (B, N, D)
        Vp = torch.relu(self.V(h_p))  # (B, M, D)
        attention = torch.einsum('bnd,bmd->bnm', Uc, Vp)  # (B, N, M)
        weights = torch.softmax(attention, dim=-1)
        context = torch.einsum('bnm,bmd->bnd', weights, Vp)  # (B, N, D)
        joint = (Uc + context) * self.q  # element-wise
        return joint.mean(dim=1)  # (B, D)