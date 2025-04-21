![GraphBAN Banner](assets/github_banner.png)

# GraphBAN
A deep learning model for predicting compound-protein interactions using a bilinear attention network with knowledge distillation and domain adaptation. It combines GCNs, pretrained LLMs (ChemBERTa & ESM), and a student-teacher architecture to generalize across unseen compounds and proteins.

Please note, this is TWO neural networks- 
1. Teacher (Graph Autoencoder)
2. Student (compound + protein encoders → BAN → classifier)

They form a knowledge-distilled dual-network, trained in two stages:
1. GAE teaches neighborhood structure (unsupervised)
2. Student learns to predict binding (supervised)

# Running locally 
```bash
python3.12 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

# Architecture
## Data Pipeline
- Parse SMILES → molecular graph  (SMILES = Simplified Molecular Input Line Entry System)
- Parse protein sequence → amino acid embedding  
- Build CPI bipartite graph  
- Create train/test splits (in-domain + cross-domain)  

## Model Implementation
- Compound encoder: GCN + ChemBERTa + Fusion  
- Protein encoder: CNN + ESM + Fusion  
- Teacher: GAE (GraphSAGE encoder + linear decoder)  
- Student: BAN + CDAN + FC layer  
- Loss: KD loss (MSE + cosine), BCE loss, adversarial domain loss  

## Training Logic
- Train teacher first (unsupervised GAE)  
- Train student with KD + BAN + CDAN  
- Evaluate using AUROC, AUPRC, F1  

## Definitions
- CPI: Compound-Protein Interaction
- GAE: Graph Autoencoder
- BAN: Bilinear Attention Network
- CDAN: Conditional Domain Adversarial Network
- AUROC: Area Under the Receiver Operating Characteristic Curve
- AUPRC: Area Under the Precision-Recall Curve
- F1: F1 Score
- SMILES: Simplified Molecular Input Line Entry System
    - A text encoding of a molecule’s 2D structure that is used to represent small molecules, aka the binders 
- Negative example: A compound that does not bind to the target protein, you need:
    1. Protein sequence for target protein 
    2. Binder that is known to bind to the target protein
    3. Non-binder that is known to not bind to the target protein
- Binary classifier: We are using this to train a binary classifier, which segments the binder and protein into bind vs non-bind
