#!/usr/bin/env python
import torch,toml
import torch.nn as nn
import torch.nn.functional as F
from models.parts import average_pool, EMBNet, ContrastiveLoss

config_dict = toml.load("config.toml")
model_param = config_dict['model']

class Similarity(nn.Module):
    def __init__(self, node_number):
        super().__init__()
        hidden_dim = model_param['HIDDEN_DIM']
        self.node_number = node_number
        self.emb = EMBNet(hidden_dim)
        self.margin = 1.0
        self.loss = ContrastiveLoss(self.margin)
    def forward(self, query_features, key_features = None, sims = None, query_labels = None, key_labels = None):
        x = self.emb(query_features)
        y = self.emb(key_features)
        predicts = torch.sqrt(torch.sum(torch.square(x - y), dim = -1))
        if not self.training:
            return predicts, x
        loss = self.loss(x,y, sims)
        return loss, predicts

class Embeddings(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.protbert = BertModel.from_pretrained(model_param['BERT'])
        self.average_pool = average_pool()
    def forward(self, input_ids, attention_masks):
        with torch.no_grad():
            embeddings = self.protbert(input_ids, attention_mask = attention_masks)['last_hidden_state'].detach()
        return self.average_pool(embeddings, attention_masks)
