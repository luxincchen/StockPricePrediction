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

def train_model(model, train_loader, epochs=10, learning_rate=0.001):
    # an epoch is one full pass through the entire training dataset
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        total_loss = 0

        for batch_x, batch_y in train_loader: #
            optimizer.zero_grad() #clears previous gradients. PyTorch accumulates gradients by default, so we must reset them before each new batch
            pred = model(batch_x) 
            loss = loss_fn(pred, batch_y)
            loss.backward() #backpropagation step - calculates the gradient of the loss wrt every parameter in the model (weights & biases)
            optimizer.step() #updates the model weights using the gradients
            total_loss += loss.item()
