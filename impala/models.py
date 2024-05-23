import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, device: torch.device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.model = (
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Dropout(p=0.8),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            ).to(device)
            .to(torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        logits = self.forward(obs)
        if deterministic:
            action = torch.argmax(logits)
        else:
            action = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

        return action, logits


class ValueFn(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, device: torch.device=torch.device('cpu')):
        super(ValueFn, self).__init__()
        self.model = (
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Dropout(p=0.8),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            ).to(device)
            .to(torch.float32)
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.model(observation)