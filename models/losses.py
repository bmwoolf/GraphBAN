import torch
import torch.nn as nn

class DistillationLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, feature_dim: int = 128):
        """Knowledge Distillation Loss.
        
        Args:
            alpha: Weight for MSE loss vs Cosine loss
            feature_dim: Expected feature dimension
        """
        super().__init__()
        self.alpha = alpha
        self.feature_dim = feature_dim
        self.mse = nn.MSELoss()
        self.cos = nn.CosineEmbeddingLoss()

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_feat: Student predictions (B, feature_dim)
            teacher_feat: Teacher embeddings (B, feature_dim)
        """
        # Shape validation
        assert student_feat.dim() <= 3, f"Too many dimensions in student features: {student_feat.shape}"
        assert teacher_feat.dim() <= 3, f"Too many dimensions in teacher features: {teacher_feat.shape}"
        
        # Remove extra dimensions if needed
        if student_feat.dim() > 2:
            student_feat = student_feat.squeeze()
        if teacher_feat.dim() > 2:
            teacher_feat = teacher_feat.squeeze()
            
        # Validate final shapes
        assert student_feat.shape[1] == self.feature_dim, \
            f"Expected student feature dim {self.feature_dim}, got {student_feat.shape[1]}"
        assert teacher_feat.shape[1] == self.feature_dim, \
            f"Expected teacher feature dim {self.feature_dim}, got {teacher_feat.shape[1]}"
            
        # Ensure same batch size
        min_batch = min(student_feat.shape[0], teacher_feat.shape[0])
        student_feat = student_feat[:min_batch]
        teacher_feat = teacher_feat[:min_batch]
        
        target = torch.ones(min_batch, device=student_feat.device)
        
        return self.alpha * self.mse(student_feat, teacher_feat) + \
               (1 - self.alpha) * self.cos(student_feat, teacher_feat, target)
