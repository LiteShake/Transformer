
import torch
import torch.nn as nn
import torch.nn.functional as F

class SDPA(nn.Module) :
    
    def __init__(self, in_size):
        super().__init__()
        
        self.in_size = in_size
        
        self.Q = nn.Linear(in_size, in_size)
        self.K = nn.Linear(in_size, in_size)
        self.V = nn.Linear(in_size, in_size)

    def forward(self, x):
        queries = self.Q(x)
        keys = self.K(x)
        values = self.V(x)

        scores = torch.bmm(queries, keys.transpose(1, 2)/ self.in_size ** 0.5)
        attn = F.softmax(scores)
        weighted = torch.bmm(attn, values)
        
        return weighted

class MHA(nn.Module):
    
    def __init__(self, in_size):
        
        self.head1 = SDPA(in_size)
        self.head2 = SDPA(in_size)
        self.head3 = SDPA(in_size)
        
        self.MHALinear = nn.Linear(in_size, in_size)
        
    def forward(self, x):
        
        h1 = self.head1(x)
        h2 = self.head2(x)
        h3 = self.head3(x)
        
        concatted = torch.cat(h1, h2, h3)
        
        outs = self.MHALinear(concatted)
        
        return outs