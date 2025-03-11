### Running in Kernel EPACT_env2 
import torch
import pandas as pd
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm as notebook_tqdm ## Jupyter-Notebook friendly progress bar
from transformers import BertTokenizer, BertModel  ### Trained on large protein dataset.

#### Using Pretrained protein language model
## https://huggingface.co/Rostlab/prot_bert
# The feature extracted from this model revealed that the LM-embeddings from unlabeled data (only protein sequences) captured important biophysical properties governing protein shape. This implied learning some
# of the grammar of the language of life realized in protein sequences.
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert')
model = BertModel.from_pretrained('Rostlab/prot_bert')

# Example peptide
# peptide = ["T T D P S F L G R Y", "Y L Q P R T F L L"]

# Tokenize the peptide
# inputs = tokenizer(peptide, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Converts the peptide sequence into a format the BERT model understands.
# This includes tokenizing the sequence into BERT-compatible input tokens.
# return_tensors="pt" → Converts the tokenized output into a PyTorch tensor (for model compatibility).
# padding=True → Ensures that all sequences have the same length by adding padding (if needed).
# truncation=True → If the sequence is too long, it will be truncated to fit within the model's limit.
# max_length=512 → Specifies the maximum sequence length (ProtBERT supports up to 512 tokens).

# Forward pass through the model
# outputs = model(**inputs)

# Get the embeddings for the peptide
# peptide_embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling of token embeddings
# print(peptide_embedding)

#### Loading the dataset
savedir = "/diazlab/data3/.abhinav/.learning/pMHC_TCR_specificity/"
iedb = pd.read_csv(savedir + "data/iedb_positives.csv")
vdj = pd.read_csv(savedir + "data/vdjdb_positives.csv")
iedb.columns == vdj.columns ### Checking if the column are matching
TCRdb = pd.concat([iedb, vdj], axis = 0)
TCRdb_nodups = TCRdb.drop_duplicates() ## No duplicates

### Using BeautifulSoul from IMGT/HLA
import requests
from bs4 import BeautifulSoup

hla_list = ['HLA-B*38:01', 'HLA-A*30:02', 'HLA-B*08:01', 'HLA-A*02:01']
base_url = "https://www.ebi.ac.uk/cgi-bin/ipd/api/allele?limit=20&project=HLA"

for hla in hla_list:
    hla_formatted = hla.replace("*", "").replace(":", "")
    url = f"{base_url}{hla_formatted}&type=protein"
    
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        sequence_tag = soup.find("pre")  # Sequence is usually in a <pre> tag
        if sequence_tag:
            sequence = sequence_tag.text.strip()
            print(f">{hla}\n{sequence}\n")
        else:
            print(f"Sequence not found for {hla}")
    else:
        print(f"Failed to retrieve {hla}")


