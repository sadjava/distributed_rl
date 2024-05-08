from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, state_dim: int, action_n: int, sizes=List[int], dropout=True, act_last=None):
        super(MLP, self).__init__()
        self.state_dim = state_dim
        self.action_n = action_n
        self.dropout = dropout
        self.sizes = sizes + [action_n]
        self.model = nn.ModuleList(
            [nn.Linear(state_dim, sizes[0])]
        )
        if self.dropout:
            self.model.append(nn.Dropout(p=0.2))
        for in_features, out_features in zip(self.sizes[:-1], self.sizes[1:]):
            self.model.append(nn.Linear(in_features, out_features))
            if self.dropout:
                self.model.append(nn.Dropout(p=0.2))
        self.relu = nn.ReLU()
        self.act_last = act_last
        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            if isinstance(param, nn.Linear):
                nn.init.kaiming_normal_(param.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model[:-1]:
            x = self.relu(layer(x))
        x = self.model[-1](x)
        if self.act_last is not None:
            x = self.act_last(x)
        return x