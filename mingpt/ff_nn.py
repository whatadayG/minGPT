import torch
import torch.nn as nn
from torch.nn import functional as F

class Config:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

config = Config(input_size=64, hidden_size=128, output_size=1)

class FFNN(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.input_size, cfg.hidden_size)
        self.fc2 = nn.Linear(cfg.hidden_size, cfg.output_size)

    def forward(self, x):
        hidden_output = F.relu(self.fc1(x))
        output = self.fc2(hidden_output)
        return output

# TODO: anything involving training or training configuration

# test
ffnn = FFNN(config)
inp = torch.randn(128, 64)
out = ffnn.forward(inp)
print(out.size())
