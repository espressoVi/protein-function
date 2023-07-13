#!/usr/bin/env python
import torch,toml
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from models.parts import average_pool, ClassificationLoss, SmoothLoss, PositiveLoss

config_dict = toml.load("config.toml")
model_param = config_dict['model']

class TopMLP(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        hidden_dim = model_param['HIDDEN_DIM']
        self.fc1 = nn.Linear(hidden_dim, hidden_dim*2)
        self.ac1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.ac2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_dim, class_num)
        self.loss = ClassificationLoss()
    def forward(self, embeddings, labels = None):
        embeddings = embeddings.float()
        x = self.ac1(self.fc1(embeddings))
        x = self.ac2(self.fc2(x))
        predicts = torch.sigmoid(self.fc3(x))
        if self.training:
            return self.loss(labels, predicts), predicts
        return predicts

class Top(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        hidden_dim = model_param['HIDDEN_DIM']
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, dilation=1, padding=1, stride=1) # (, 3, embed_size)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2) # (, 3,512)
        self.conv2 = nn.Conv1d(in_channels=3, out_channels=8, kernel_size=3, dilation=1, padding=1, stride=1) # (, 8,512)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2) # (, 8,256)
        self.fc1 = nn.Linear(in_features=int(8 * hidden_dim/4), out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=class_num)
        self.loss = PositiveLoss()
    def forward(self, embeddings, labels = None):
        x = embeddings.float()
        x = x.reshape(x.shape[0], 1, x.shape[1])
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        predicts = torch.sigmoid(self.fc2(x))
        if self.training:
            return self.loss(labels, predicts), predicts
        return predicts

class Finetune(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        hidden_dim = model_param['HIDDEN_DIM']
        self.protbert = BertModel.from_pretrained(model_param['BERT'])
        self.average_pool = average_pool()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, class_num)
        self.activation = nn.GELU()
        self.loss = ClassificationLoss()
    def forward(self, input_ids, attention_masks, labels = None):
        embeddings = self.protbert(input_ids, attention_mask = attention_masks)['last_hidden_state']
        x = self.average_pool(embeddings, attention_masks)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        predicts = torch.sigmoid(self.fc3(x))
        if self.training:
            return self.loss(labels, predicts), predicts
        return predicts
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
