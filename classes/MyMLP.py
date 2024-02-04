import torch
import torch.nn as nn
import torch.nn.functional as F


class MyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(32 * 32 * 3, 512)
        self.h1 = nn.Linear(512, 128)
        self.h2 = nn.Linear(128, 32)
        self.output_layer = nn.Linear(32, 2)

    def forward(self, x):
        out = torch.flatten(x, 1)
        out = F.relu(self.input_layer(out))
        out = F.relu(self.h1(out))
        out = F.relu(self.h2(out))
        out = self.output_layer(out)
        return out

    def __str__(self) -> str:
        return "MyMLP"
