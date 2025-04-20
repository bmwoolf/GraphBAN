import torch
from models.feature_fusion import FeatureFusion
from models.gcn_encoder import GCNEncoder

def main():
    # Initialize models
    feature_fusion = FeatureFusion(dim=128)
    gcn_encoder = GCNEncoder(in_feats=128, hidden_feats=128, num_layers=3)
    
    # Move models to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_fusion = feature_fusion.to(device)
    gcn_encoder = gcn_encoder.to(device)
    
    print("Models initialized successfully!")

if __name__ == "__main__":
    main() 