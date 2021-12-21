import torch
from torch import nn


class MLP(nn.Module):
    activation_classes = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'sigmoid': nn.Sigmoid,
        'tanh': nn.Tanh,
    }

    def __init__(self, in_features, out_features, hidden_dims, coefficient=1.0, activation="leaky_relu"):
        super().__init__()
        modules = []
        hidden_dims = [in_features] + hidden_dims
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    # nn.BatchNorm1d(hidden_dims[i + 1]),
                    self.activation_classes[activation](),
                )
            )
        self.network = nn.Sequential(
            *modules,
            *([nn.Linear(hidden_dims[-1], out_features)] if out_features else []),
        )
        self.coefficient = coefficient

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coefficient * self.network(x)
