#!/usr/bin/env python
import torch,toml
import torch.nn as nn
import torch.nn.functional as F
from models.parts import average_pool, GraphConv, EMBNet

config_dict = toml.load("config.toml")
model_param = config_dict['model']

class Node(nn.Module):
    def __init__(self, node_number):
        super().__init__()
        hidden_dim = model_param['HIDDEN_DIM']
        self.node_number = node_number
        #self.node_embed = nn.Embedding(node_number, 256)
        #self.conv1 = GraphConv(in_features = 256, out_features = 256)
        #self.ac1 = nn.ReLU()
        #self.conv2 = GraphConv(in_features = 256, out_features = 128,)
        #self.ac2 = nn.ReLU()
        #self.conv3 = GraphConv(in_features = 128, out_features = 128,)
        self.emb = EMBNet(hidden_dim)
        self.fc = nn.Linear(128, node_number)

        #self.cos = nn.CosineSimilarity(dim = -1)
        self.loss = nn.CrossEntropyLoss(label_smoothing = 0.0)
    def forward(self, edges, embeddings, parent_mask, label_child = None, labels = None):
        #x = self.node_embed(torch.arange(self.node_number).to(embeddings.device)).unsqueeze(0).repeat(embeddings.shape[0],1,1)
        #x = self.conv1(x, edges)
        #x = self.ac1(x)
        #x = self.conv2(x, edges)
        #x = self.ac2(x)
        #x = self.conv3(x, edges)

        y = self.emb(embeddings)
        sim = self.fc(y)
        #sim = self.cos(x, y.unsqueeze(1).repeat(1,edges.shape[-1],1))
        masked_sim = sim.masked_fill((1-parent_mask).bool(), float("-inf"))
        #masked_sim = y.masked_fill((1-parent_mask).bool(), float("-inf"))
        _,preds = torch.topk(masked_sim, 1)
        if self.training:
            loss = self.loss(masked_sim, label_child)
            return loss, preds.squeeze()
        return preds.squeeze()


class Embeddings(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.protbert = BertModel.from_pretrained(model_param['BERT'])
        self.average_pool = average_pool()
    def forward(self, input_ids, attention_masks):
        with torch.no_grad():
            embeddings = self.protbert(input_ids, attention_mask = attention_masks)['last_hidden_state'].detach()
        return self.average_pool(embeddings, attention_masks)
