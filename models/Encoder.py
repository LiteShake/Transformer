import torch
import torch.nn as nn
from .Attention import *
from .PosEncoding import *
import torch.nn.functional as F

class Encoder(nn.Module):
    
    def __init__(self, in_size):
        super().__init__()
        
        self.mha1 = MHA(in_size)
        self.FF1 = nn.Linear(in_size, in_size)
        self.FF2 = nn.Linear(in_size, in_size)
        self.layernorm1 = nn.LayerNorm(in_size)
        self.layernorm2 = nn.LayerNorm(in_size)
        
    def forward(self, x):
        
        # pos_vector = GetPositionalEncoding(pos, x.shape, x.shape)
        
        # positioned_input = x + pos_vector
        
        att_vals = self.mha1(x)
        
        residual_1 = x + att_vals
        normalized_1 = self.layernorm1(residual_1)
        
        dense_1 = F.relu(self.FF1(normalized_1))
        dense_2 = F.relu(self.FF2(dense_1))
        
        residual_2 = residual_1 + dense_2
        normalized_2 = self.layernorm2(residual_2)
        
        return normalized_2
