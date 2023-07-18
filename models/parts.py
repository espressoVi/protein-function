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

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Linear(in_features = in_features, out_features = out_features, bias = False)
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1.0/sqrt(self.out_features)
        self.weight.reset_parameters()
        self.bias.data.uniform_(-stdv, stdv)
    def forward(self, input, adj):
        support = self.weight(input)
        output = torch.matmul(adj, support)
        return output + self.bias

class ResBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1x1 = nn.Conv1d(self.in_features, self.out_features, 1, stride = 1, padding = 0, )
        self.maxPool = nn.MaxPool1d(7, stride=2, padding=3)
        self.bn1 = nn.Sequential(nn.BatchNorm1d(self.in_features), nn.ReLU(), nn.Dropout(0.1))
        self.conv1 = nn.Conv1d(self.in_features, self.out_features, 7, stride=2, padding=3, )
        self.bn2 = nn.Sequential(nn.BatchNorm1d(self.out_features), nn.ReLU(), nn.Dropout(0.1))
        self.conv2 = nn.Conv1d(self.out_features, self.out_features, 7, stride = 1, padding = 3,)
    def forward(self, layer):
        output = layer
        shortcut = self.conv1x1(layer)
        shortcut = self.maxPool(shortcut)
        output = self.bn1(output)
        output = self.conv1(output)
        output = self.bn2(output)
        output = self.conv2(output)
        output += shortcut
        return output

class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, targets, predicts):
        bce = targets * torch.log(predicts + 1e-10) + (1-targets.int()) * torch.log(1-predicts+1e-10)
        return -torch.mean(bce)
