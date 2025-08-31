import torch
import torch.nn as nn
import torch.optim as optim

class StockNN(nn.Module):
    def __init__(self, seq_len=10, n_features=6, hidden1=64, hidden2=128, horizon=10):
        super().__init__()
        input_dim = seq_len * n_features   # flatten window
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, horizon)
        )

    def forward(self, x):
        return self.model(x)


def train_model(model, train_loader, epochs=100, learning_rate=0.0001):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    avg_loss = total_loss / (len(train_loader) * train_loader.batch_size)
    return avg_loss


@torch.no_grad()
def test_model(model, test_loader):
    loss_fn = nn.MSELoss()
    model.eval()
    total_loss = 0
    for batch_x, batch_y in test_loader:
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)
        total_loss += loss.item()
    avg_loss = total_loss / (len(test_loader) * test_loader.batch_size)
    return avg_loss
