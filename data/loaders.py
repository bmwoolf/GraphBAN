import torch
import dgl
from torch.utils.data import Dataset, DataLoader
from data.preprocess import smiles_to_graph, protein_to_tensor

class CPIDataset(Dataset):
    def __init__(self, pairs):  # list of (smiles, sequence, label)
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        smiles, seq, label = self.pairs[idx]
        g = smiles_to_graph(smiles)
        s = protein_to_tensor(seq)
        return g, s, torch.tensor(label, dtype=torch.float32)

def collate_fn(batch):
    graphs, seqs, labels = zip(*batch)
    batched_graph = dgl.batch(graphs)
    seq_tensor = torch.stack(seqs)
    label_tensor = torch.stack(labels)
    return batched_graph, seq_tensor, label_tensor

def get_loader(pairs, batch_size=32):
    dataset = CPIDataset(pairs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
