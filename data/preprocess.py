from rdkit import Chem
from rdkit.Chem import rdmolops
import torch
import dgl

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    g = dgl.graph(rdmolops.GetAdjacencyMatrix(mol).nonzero())
    g = dgl.add_self_loop(g)
    return g

def protein_to_tensor(seq, aa_to_idx=None, max_len=512):
    if aa_to_idx is None:
        aa_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    seq_tensor = torch.zeros(max_len, dtype=torch.long)
    for i, aa in enumerate(seq[:max_len]):
        seq_tensor[i] = aa_to_idx.get(aa, 0)
    return seq_tensor
