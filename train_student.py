import torch
from models.student_model import StudentModel
from models.losses import DistillationLoss
from models.cdan_module import CDAN

def train_student(h_c: torch.Tensor, h_p: torch.Tensor, teacher_emb: torch.Tensor, epochs: int = 50):
    """Train student model with knowledge distillation.
    
    Args:
        h_c: Compound representations (B, 1, hidden_dim)
        h_p: Protein representations (B, 1, hidden_dim)
        teacher_emb: Teacher embeddings (B, hidden_dim)
    """
    # Debug shapes
    print(f"Student input shapes - h_c: {h_c.shape}, h_p: {h_p.shape}, teacher: {teacher_emb.shape}")
    
    # Detach all inputs
    h_c = h_c.detach()
    h_p = h_p.detach()
    teacher_emb = teacher_emb.detach()
    
    # Move models to same device as inputs
    device = h_c.device
    student = StudentModel().to(device)
    cdan = CDAN(feature_dim=128).to(device)
    kd_loss_fn = DistillationLoss()
    optimizer = torch.optim.Adam(list(student.parameters()) + list(cdan.parameters()), lr=1e-4)

    for epoch in range(epochs):
        student.train()
        cdan.train()
        optimizer.zero_grad()

        # Forward pass
        pred = student(h_c, h_p)
        
        # Match batch sizes
        if teacher_emb.shape[0] != pred.shape[0]:
            teacher_emb = teacher_emb[:pred.shape[0]]
        
        # Compute losses
        kd_loss = kd_loss_fn(pred, teacher_emb)
        domain_labels = torch.randint(0, 2, (pred.shape[0],), device=device)
        domain_loss = torch.nn.CrossEntropyLoss()(cdan(pred), domain_labels)
        
        # Backward pass with retain_graph
        loss = kd_loss + domain_loss
        loss.backward(retain_graph=True)
        optimizer.step()

    return student