import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer

class ESMEncoder(nn.Module):
    def __init__(self, model_name='facebook/esm1b_t33_650M_UR50S'):
        super().__init__()
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.model.config.hidden_size, 128)

    def forward(self, protein_seqs):
        tokens = self.tokenizer(protein_seqs, return_tensors='pt', padding=True, truncation=True)
        output = self.model(**tokens)
        pooled = output.last_hidden_state[:, 0]  # CLS token
        return self.linear(pooled)
