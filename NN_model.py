import torch
import torch.nn as nn
import torch.optim as optim

class StockNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),             
            nn.Linear(60, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
            )

    def forward(self, x):
        return self.model(x)
