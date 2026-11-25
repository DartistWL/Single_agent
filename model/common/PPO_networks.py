"""PPO networks."""
import torch
from torch import nn

class SharedFeatureExtractor(nn.Module):

    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        for layer in self.net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, x):
        return self.net(x)

class ActionBranch(nn.Module):

    def __init__(self, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim))
        for layer in self.net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, x):
        return self.net(x)

class ValueNet(nn.Module):

    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden_dim * 2), nn.LayerNorm(hidden_dim * 2), nn.ReLU(), nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        for layer in self.net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()

    def forward(self, x):
        return self.net(x).squeeze(-1)
