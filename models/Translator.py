import torch
import torch.nn as nn
import torch.nn.functional as F
from .Encoder import *
from .Decoder import *

class Translator(nn.Module):
    
    def __init__(self, ):
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        
        output = []
        
        for i in x:
            
            if( i != "<LFLIP>" ):
                
                