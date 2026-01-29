import torch

#pos_enc = torch.arange(0,10,1).unsqueeze(-1).repeat(1,2).flatten()

#print(pos_enc)


L,p = 10,torch.tensor([3,2,1])

sin = torch.sin(2**torch.arange(0,L).unsqueeze(-1) * torch.pi * p)
cos = torch.cos(2**torch.arange(0,L).unsqueeze(-1) * torch.pi * p)

print(sin)
print(cos)

pos_enc = torch.cat([sin,cos],dim=-1).flatten(-2,-1)

print(pos_enc)
"""

lst = [1,2,3]
lst.append(*[4,5])

print(lst)
"""

a = torch.randn(10,3,1)
b = torch.randn(3,1)

print((a*b).shape)