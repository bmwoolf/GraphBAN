import torch
import torch.nn as nn

class DistillationLoss(nn.Module):
    def __init__(self, alpha: float = 0.5):
        """Knowledge Distillation Loss.
        
        Args:
            alpha: Weight for MSE loss vs Cosine loss
        """
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.cos = nn.CosineEmbeddingLoss()

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_feat: Student predictions (B, hidden_dim)
            teacher_feat: Teacher embeddings (B, hidden_dim)
        """
        # Remove extra dimensions if needed
        if student_feat.dim() > 2:
            student_feat = student_feat.squeeze()
        if teacher_feat.dim() > 2:
            teacher_feat = teacher_feat.squeeze()
            
        # Ensure same batch size
        min_batch = min(student_feat.shape[0], teacher_feat.shape[0])
        student_feat = student_feat[:min_batch]
        teacher_feat = teacher_feat[:min_batch]
        
        target = torch.ones(min_batch, device=student_feat.device)
        
        return self.alpha * self.mse(student_feat, teacher_feat) + \
               (1 - self.alpha) * self.cos(student_feat, teacher_feat, target)
