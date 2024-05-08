from typing import List
import math

import torch 
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F

from actors import MLP

class ActorCritic(nn.Module):
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: List[int] = [200],
                 action_scale: float = 1.,
                 alpha=5e-3
    ):
        super(ActorCritic, self).__init__()
        self.pi_model = MLP(state_dim, 2 * action_dim, hidden_sizes, dropout=False)
        self.v_model = MLP(state_dim, 1, hidden_sizes, dropout=False)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.hidden_sizes = hidden_sizes

        self.alpha = alpha

        self.init_weights()
    
    def init_weights(self):
        for layer in self.pi_model.parameters():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0., std=0.1)
                nn.init.constant_(layer.bias, 0.)
        for layer in self.v_model.parameters():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0., std=0.1)
                nn.init.constant_(layer.bias, 0.)
    
    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        m = Normal(mu.view(1, ).data, sigma.view(1, ).data)
        return m.sample().numpy()
    
    def forward(self, x):
        mu, sigma = torch.split(self.pi_model(x), self.action_dim, dim=1)
        mu = self.action_scale * F.tanh(mu)
        sigma = F.softplus(sigma) + 0.001
        values = self.v_model(x)
        return mu, sigma, values
        
    def loss_func(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
        td_eror = v_t - values
        v_loss = td_eror.pow(2)

        m = Normal(mu, sigma)
        log_prob = m.log_prob(a)
        entropy = 0.5 * (1 + math.log(2 * math.pi) + torch.log(m.scale))
        pi_loss = -(log_prob * td_eror.detach() + self.alpha * entropy)
        total_loss = (pi_loss + v_loss).mean()
        return total_loss
