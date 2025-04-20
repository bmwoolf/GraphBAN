![GraphBAN Banner](assets/github_banner.png)

# GraphBAN
A deep learning model for predicting compound-protein interactions using a bilinear attention network with knowledge distillation and domain adaptation. It combines GCNs, pretrained LLMs (ChemBERTa & ESM), and a student-teacher architecture to generalize across unseen compounds and proteins.


# Running locally 
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

# Architecture
## Data Pipeline:
- Parse SMILES → molecular graph  
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