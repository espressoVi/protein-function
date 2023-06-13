#!/usr/bin/env python
import torch,toml
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

config_dict = toml.load("config.toml")
model_param = config_dict['model']

class ProtBERT(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        hidden_dim = model_param['HIDDEN_DIM']
        self.protbert = BertModel.from_pretrained("Rostlab/prot_bert")
        self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, class_num)
        self.activation = nn.GELU()
    def average_pool(self, embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    def forward(self, input_ids, attention_masks, labels = None):
        with torch.no_grad():
            embeddings = self.protbert(input_ids, attention_mask = attention_masks)['last_hidden_state'].detach()
        #embeddings = self.protbert(input_ids, attention_mask = attention_masks)['last_hidden_state']   #Uncomment to retrain BERT
        x = self.average_pool(embeddings, attention_masks)
        x = self.activation(self.fc1(x))
        predicts = torch.sigmoid(self.fc2(x))
        if self.training:
            return self.ClassificationLoss(labels, predicts), predicts
        return predicts
    def ClassificationLoss(self, targets, predicts):
        bce = targets * torch.log(predicts + 1e-10) + (1-targets.int()) * torch.log(1-predicts+1e-10)
        return -torch.mean(bce)
    def send(self, device):
        self.protbert.to(device)
        self.to(device)

