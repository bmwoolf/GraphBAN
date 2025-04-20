![GraphBAN Banner](assets/github_banner.png)

# GraphBAN
A deep learning model for predicting compound-protein interactions using a bilinear attention network with knowledge distillation and domain adaptation. It combines GCNs, pretrained LLMs (ChemBERTa & ESM), and a student-teacher architecture to generalize across unseen compounds and proteins.


# Running locally 
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
