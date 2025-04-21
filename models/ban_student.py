import torch
import torch.nn as nn

class BANLayer(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.U = nn.Linear(dim, dim)
        self.V = nn.Linear(dim, dim)
        self.q = nn.Parameter(torch.randn(dim))
        self.dim = dim  # Store for shape checking

    def forward(self, h_c, h_p):
        """Bilinear attention network layer.
        
        Args:
            h_c: Compound features (B, N, D)
            h_p: Protein features (B, M, D)
            
        Returns:
            torch.Tensor: Joint features (B, D)
        """
        # Shape validation
        assert h_c.dim() == 3, f"Expected 3D tensor for h_c, got shape {h_c.shape}"
        assert h_p.dim() == 3, f"Expected 3D tensor for h_p, got shape {h_p.shape}"
        assert h_c.shape[2] == self.dim, f"Expected feature dim {self.dim}, got {h_c.shape[2]}"
        assert h_p.shape[2] == self.dim, f"Expected feature dim {self.dim}, got {h_p.shape[2]}"
        assert h_c.shape[0] == h_p.shape[0], f"Batch sizes must match: {h_c.shape[0]} vs {h_p.shape[0]}"

        Uc = torch.relu(self.U(h_c))  # (B, N, D)
        Vp = torch.relu(self.V(h_p))  # (B, M, D)
        attention = torch.einsum('bnd,bmd->bnm', Uc, Vp)  # (B, N, M)
        weights = torch.softmax(attention, dim=-1)
        context = torch.einsum('bnm,bmd->bnd', weights, Vp)  # (B, N, D)
        joint = (Uc + context) * self.q  # element-wise
        return joint.mean(dim=1)  # (B, D)