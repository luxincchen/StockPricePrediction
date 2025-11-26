# LSTMModel.py
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=50, num_layers=2, output_size=10):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)       # (batch, seq_len, hidden_size)
        out = out[:, -1, :]         # take last hidden state
        out = self.fc(out)          # (batch, output_size=10)
        return out
