import torch
import torch.nn as nn

class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.cos = nn.CosineEmbeddingLoss()

    def forward(self, student_feat, teacher_feat):
        target = torch.ones(student_feat.size(0)).to(student_feat.device)
        return self.alpha * self.mse(student_feat, teacher_feat) + \
               (1 - self.alpha) * self.cos(student_feat, teacher_feat, target)
