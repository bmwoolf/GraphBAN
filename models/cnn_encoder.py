import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, embedding_dim=128, vocab_size=23, seq_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, seq):
        x = self.embed(seq).permute(0, 2, 1)  # B x C x L
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        return x