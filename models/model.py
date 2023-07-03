#!/usr/bin/env python
import torch,toml
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from models.parts import average_pool

config_dict = toml.load("config.toml")
model_param = config_dict['model']

class Top(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        hidden_dim = model_param['HIDDEN_DIM']
        self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, class_num)
        self.activation = nn.GELU()
    def forward(self, embeddings, labels = None):
        x = self.activation(self.fc1(embeddings))
        predicts = torch.sigmoid(self.fc2(x))
        if self.training:
            return self.ClassificationLoss(labels, predicts), predicts
        return predicts
    def ClassificationLoss(self, targets, predicts):
        bce = targets * torch.log(predicts + 1e-10) + (1-targets.int()) * torch.log(1-predicts+1e-10)
        return -torch.mean(bce)

class Finetune(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        hidden_dim = model_param['HIDDEN_DIM']
        self.protbert = BertModel.from_pretrained(model_param['BERT'])
        self.average_pool = average_pool()
        self.fc1 = nn.Linear(hidden_dim, 2*hidden_dim)
        #self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc2 = nn.Linear(2*hidden_dim, class_num)
        #self.fc2 = nn.Linear(hidden_dim//2, class_num)
        self.activation = nn.GELU()
    def forward(self, input_ids, attention_masks, labels = None):
        embeddings = self.protbert(input_ids, attention_mask = attention_masks)['last_hidden_state']
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

class Embeddings(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.protbert = BertModel.from_pretrained(model_param['BERT'])
        self.average_pool = average_pool()
    def forward(self, input_ids, attention_masks):
        with torch.no_grad():
            embeddings = self.protbert(input_ids, attention_mask = attention_masks)['last_hidden_state'].detach()
        return self.average_pool(embeddings, attention_masks)
