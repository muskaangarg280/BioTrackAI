import torch
import torch.nn as nn

class InputLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.seq(x)


class HiddenBlock(nn.Module):
    def __init__(self, size, dropout=0.4):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.seq(x)


class OutputLayer(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)  # logits (used with BCEWithLogitsLoss)


class HeartAttackDNN(nn.Module):
    def __init__(self, input_size, hidden_size=256, depth=5, dropout=0.5):
        super().__init__()
        self.input_layer = InputLayer(input_size, hidden_size)

        self.hidden_layers = nn.Sequential(
            *[HiddenBlock(hidden_size, dropout) for _ in range(depth)]
        )

        self.output_layer = OutputLayer(hidden_size)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        return self.output_layer(x)
