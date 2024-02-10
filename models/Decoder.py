
import torch
import torch.nn as nn
from .Attention import *
from .PosEncoding import *
import torch.nn.functional as F

class Decoder(nn.Module):
    
    
    def __init__(self, in_size):
        super().__init__()
        
        self.mha1 = MHA(in_size)
        
        self.sa_1 = SDPA(in_size)
        self.sa_2 = SDPA(in_size)
        self.sa_3 = SDPA(in_size)
        
        self.att_linear_1 = nn.Linear(3 * in_size, in_size)
        
        self.FF1 = nn.Linear(in_size, in_size)
        self.FF2 = nn.Linear(in_size, in_size)
        
        self.Linear1 = nn.Linear(in_size, in_size)
        
        self.layernorm1 = nn.LayerNorm(in_size)
        self.layernorm2 = nn.LayerNorm(in_size)
        self.layernorm3 = nn.LayerNorm(in_size)
        
    def forward(self, x, encoder_in):
        
        # pos_vector = GetPositionalEncoding(pos, x.shape, x.shape)
        
        # positioned_input = x + pos_vector
        
        # region SELF ATTENTION + NORM
        att_1 = self.mha1(x)
        
        residual_1 = x + att_1
        normalized_1 = self.layernorm1(residual_1)
        # endregion
        
        # region CROSS ATTENTION
        head1 = self.sa_1(encoder_in)
        head2 = self.sa_2(encoder_in)
        head3 = self.sa_3(normalized_1)
        # endregion
        
        cat_1 = torch.cat((head1, head2, head3), 0)
        flt = torch.flatten(cat_1)
        att_2 = self.att_linear_1(flt)
        
        residual_2 = att_2 + normalized_1
        normalized_2 = self.layernorm2(residual_2)
        
        dense_1 = F.relu(self.FF1(normalized_2))
        dense_2 = F.relu(self.FF2(dense_1))
        
        residual_3 = residual_1 + dense_2
        normalized_3 = self.layernorm3(residual_3)
        
        # region CLASSIFIER
        """ 
        clf_out = self.Linear1(normalized_3)
        soft = F.softmax(clf_out)
        """
        # endregion 
        return normalized_3
