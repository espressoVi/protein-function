#!/usr/bin/env python
from math import sqrt
import torch,toml
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class average_pool(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class EMBNet(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
    def forward(self, embeddings):
        y = self.fc1(embeddings.float())
        y1 = self.bn1(y)
        y = self.fc2(F.relu(y1))
        y = self.fc3(F.relu(y))
        y = self.bn2(y)
        y = y+y1
        y = F.relu(self.fc4(y))
        y = self.bn3(y)
        return y

class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, targets, predicts):
        bce = targets * torch.log(predicts + 1e-10) + (1-targets) * torch.log(1-predicts+1e-10)
        return -torch.mean(bce)

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    def forward(self, x0, x1, y):
        dist = torch.sum(torch.square(x0-x1), 1)
        loss = (1-y)*dist + y*torch.square(torch.clamp(self.margin - torch.sqrt(dist), min=0.0))
        return torch.mean(loss)
