#!/usr/bin/env python
import torch,toml
import torch.nn as nn
import torch.nn.functional as F
from models.parts import average_pool, GraphConv, ResBlock

config_dict = toml.load("config.toml")
model_param = config_dict['model']

class Node(nn.Module):
    def __init__(self, node_number):
        super().__init__()
        hidden_dim = model_param['HIDDEN_DIM']
        self.node_number = node_number
        self.node_embed = nn.Embedding(node_number, 512)
        self.conv1 = GraphConv(in_features = 512, out_features = 256)
        self.ac1 = nn.ReLU()
        self.conv2 = GraphConv(in_features = 256, out_features = 257,)
        self.ac2 = nn.ReLU()
        self.pool = nn.MaxPool1d(5, stride = 2, padding = 1)
        self.conv3 = GraphConv(in_features = 128, out_features = 64,)
        self.protein = nn.Sequential(ResBlock(1,3), ResBlock(3,5), ResBlock(5,7), ResBlock(7,9),)
        self.fc1 = nn.Linear(9,1)
        self.cos = nn.CosineSimilarity(dim = -1)
    def forward(self, edges, embeddings, probe_nodes, labels = None):
        x = self.node_embed(torch.arange(self.node_number).to(embeddings.device)).unsqueeze(0).repeat(embeddings.shape[0],1,1)
        x = self.conv1(x, edges)
        x = self.ac1(x)
        x = self.conv2(x, edges)
        x = self.ac2(x)
        x = self.pool(x)
        x = self.conv3(x, edges)

        y = self.protein(embeddings.float().unsqueeze(1))
        y = self.fc1(y.view(y.shape[0],y.shape[-1], y.shape[1])).squeeze(-1)
        probe_node_embeddings = x[torch.arange(x.shape[0]).to(x.device),probe_nodes,:].squeeze(1)
        probe_node_sim = self.cos(probe_node_embeddings, y)
        if self.training:
            positives = labels*probe_node_sim
            unk = (1-labels)*probe_node_sim
            loss = torch.mean( torch.square(1-positives) + torch.square(unk))
            #loss = torch.mean( (1-positives) + 0.1*torch.square(unk))
            return loss, probe_node_sim
        return probe_node_sim


class Embeddings(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.protbert = BertModel.from_pretrained(model_param['BERT'])
        self.average_pool = average_pool()
    def forward(self, input_ids, attention_masks):
        with torch.no_grad():
            embeddings = self.protbert(input_ids, attention_mask = attention_masks)['last_hidden_state'].detach()
        return self.average_pool(embeddings, attention_masks)
