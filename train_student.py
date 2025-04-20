import torch
from models.student_model import StudentModel
from models.losses import DistillationLoss
from models.cdan_module import CDAN

def train_student(h_c, h_p, teacher_emb, epochs=50):
    student = StudentModel()
    cdan = CDAN(feature_dim=128)
    kd_loss_fn = DistillationLoss()
    optimizer = torch.optim.Adam(list(student.parameters()) + list(cdan.parameters()), lr=1e-4)

    for epoch in range(epochs):
        student.train()
        cdan.train()

        pred = student(h_c, h_p)
        kd_loss = kd_loss_fn(pred, teacher_emb)
        domain_loss = torch.nn.CrossEntropyLoss()(cdan(pred), torch.randint(0, 2, (pred.shape[0],)))
        loss = kd_loss + domain_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return student