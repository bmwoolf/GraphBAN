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

# Example dummy dataset
pairs = [
    ("CCO", "MTEITAAMVKELRESTGAGMMDCKNALSETQHEWAY", 1), # ethanol
    ("CCN", "GAMAGSGAGAVVTGALGRLLVVYPWTQRFFESFGDLST", 0),
    ("CCC", "MVKVGVNGFGRIGRLVTRAAFNSGKVDIVLDSGDGVTH", 1)
]

data_loader = get_loader(pairs, batch_size=2)

# Instantiate encoders
compound_gcn = GCNEncoder()
compound_llm = ChemBERTaEncoder()
protein_cnn = CNNEncoder()
protein_llm = ESMEncoder()
fuser = FeatureFusion()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

for batched_graph, seqs, labels in data_loader:
    # Encode inputs
    g_feats = torch.randn(batched_graph.num_nodes(), 128).to(device)  # placeholder node features
    compound_repr = fuser(compound_gcn(batched_graph, g_feats), compound_llm([p[0] for p in pairs]))
    protein_repr = fuser(protein_cnn(seqs), protein_llm([p[1] for p in pairs]))

    # Train teacher on compound graph only (simplified)
    teacher = train_teacher(batched_graph, g_feats)
    teacher_emb, _ = teacher(batched_graph, g_feats)

    # Train student model
    student = train_student(compound_repr.unsqueeze(1), protein_repr.unsqueeze(1), teacher_emb)
    print("Training complete")
    break