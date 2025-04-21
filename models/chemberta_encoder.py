import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class ChemBERTaEncoder(nn.Module):
    def __init__(self, model_name='seyonec/ChemBERTa-zinc-base-v1'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(self.model.config.hidden_size, 128)

    def forward(self, smiles_list):
        device = next(self.parameters()).device  # Get model's device
        encoded = self.tokenizer(smiles_list, return_tensors='pt', padding=True, truncation=True)
        # Move encoded inputs to same device as model
        encoded = {k: v.to(device) for k, v in encoded.items()}
        output = self.model(**encoded)
        pooled = output.last_hidden_state[:, 0]  # CLS token
        return self.linear(pooled)