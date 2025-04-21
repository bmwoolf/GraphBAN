import torch
from models.gcn_encoder import GCNEncoder
from models.chemberta_encoder import ChemBERTaEncoder
from models.cnn_encoder import CNNEncoder
from models.esm_encoder import ESMEncoder
from models.feature_fusion import FeatureFusion
from models.gae_teacher import GAETeacher
from models.train_teacher import train_teacher
from models.train_student import train_student
from data.loaders import get_loader
from utils.device_utils import get_device, move_to_device

# Example dummy dataset
pairs = [
    ("CCO", "MTEITAAMVKELRESTGAGMMDCKNALSETQHEWAY", 1),
    ("CCN", "GAMAGSGAGAVVTGALGRLLVVYPWTQRFFESFGDLST", 0),
    ("CCC", "MVKVGVNGFGRIGRLVTRAAFNSGKVDIVLDSGDGVTH", 1)
]

data_loader = get_loader(pairs, batch_size=2)

# Get best available device
device = get_device()

# Move models to device
compound_gcn = move_to_device(GCNEncoder(), device)
compound_llm = move_to_device(ChemBERTaEncoder(), device)
protein_cnn = move_to_device(CNNEncoder(), device)
protein_llm = move_to_device(ESMEncoder(), device)
fuser = move_to_device(FeatureFusion(), device)

for batched_graph, seqs, labels in data_loader:
    # Move data to device
    batched_graph = batched_graph.to(device)
    seqs = seqs.to(device)
    labels = labels.to(device)
    smiles_batch = [pairs[i][0] for i in range(len(labels))]  # match batch size
    prots_batch = [pairs[i][1] for i in range(len(labels))]   # match batch size

    # Placeholder node features
    g_feats = torch.randn(batched_graph.num_nodes(), 128).to(device)

    # Encode
    c_struct = compound_gcn(batched_graph, g_feats)
    c_llm = compound_llm(smiles_batch)
    compound_repr = fuser(c_struct, c_llm)

    p_seq = protein_cnn(seqs)
    p_llm = protein_llm(prots_batch)
    protein_repr = fuser(p_seq, p_llm)

    # Train teacher
    teacher = train_teacher(batched_graph, g_feats)
    teacher_emb, _ = teacher(batched_graph, g_feats)

    # Train student
    student = train_student(compound_repr.unsqueeze(1), protein_repr.unsqueeze(1), teacher_emb)
    print("Training complete")
    break
