import torch
import math

def GetPositionalEncoding(pos, dim, d_model):
    
    embedding_lst = []
    
    for i in range(0, dim , 2):
        
        embedding_lst.append(torch.sin( pos/(10_000)**(2*dim / d_model) ))
        embedding_lst.append(torch.cos( pos/(10_000)**(2*dim / d_model) ))
        
    return torch.Tensor(embedding_lst)
    
class PositionEncoding(nn.Module):
    def __init__(self,max_len,d_model):
        super(PositionEncoding,self).__init__()
        self.max_len = max_len+5
        self.register_buffer('pos_table', self.tensor_pos_encoding(self.max_len, d_model))

    def pos_encoding(self,pos, k):
        """taking an vocab index and generating a a geometric progression with k dimensions """
        f = lambda i,k: pos / 10000**(2 * (i // 2) / k)
        return [torch.sin(f(i,k)) if i%2==0 else torch.cos(f(i,k)) for i in range(0,k)]

    def tensor_pos_encoding(self,max_len,dim):
        return torch.tensor([self.pos_encoding(i,dim) for i in range(max_len)],device=device).view( max_len,dim)

    def forward(self,x):
        return x+ self.pos_table[:x.size(1),:].detach().clone().unsqueeze(0)