import torch
import torch.nn as nn
from model_utils import *

class simpliest_GNN(nn.Module):
    
    def __init__(self, nb_nodes, nb_features, nb_classes):
        super(simpliest_GNN, self).__init__()
        self.W = nn.Linear(in_features=nb_features, out_features=nb_features)
        self.fc = nn.Linear(in_features=nb_nodes, out_features=nb_classes)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        
    def forward(self, image, adj_mat):
        batch_size = image.shape[0]

        # batching adjacency matrix (nb_nodes x nb_nodes) --> (batch size x nb_nodes x nb_nodes)
        # (nb_nodes x nb_nodes) --> (1 x nb_nodes x nb_nodes)
        tens_adj_mat = torch.from_numpy(adj_mat).float().unsqueeze(0)
        # (1 x nb_nodes x nb_nodes) --> (batch size x nb_nodes x nb_nodes)
        batched_tens_adj_mat = tens_adj_mat.expand(batch_size, tens_adj_mat.shape[1], tens_adj_mat.shape[2])

        # flattening image (batch size x width x height) --> (batch size x width*height)
        image = torch.flatten(image.squeeze(), 1, 2).unsqueeze(2)#.permute(2,1,0)
        
        AX = torch.bmm(batched_tens_adj_mat, image)
        Y = self.W(AX)
        Y = Y.view(-1, Y.shape[1] * Y.shape[2])
        Y = self.tanh(Y)
        output = self.fc(Y)
        prob = self.softmax(output)
        return output