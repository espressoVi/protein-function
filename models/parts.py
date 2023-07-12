#!/usr/bin/env python
import torch,toml
import torch.nn as nn
import torch.nn.functional as F

class average_pool(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, targets, predicts):
        bce = targets * torch.log(predicts + 1e-10) + 0.1*(1-targets.int()) * torch.log(1-predicts+1e-10)
        return -torch.mean(bce)
