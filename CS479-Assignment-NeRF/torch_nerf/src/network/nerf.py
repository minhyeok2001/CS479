"""
Pytorch implementation of MLP used in NeRF (ECCV 2020).
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
import torch.nn as nn
import torch.nn.functional as F


## 아.... 이미 만들어져있음...

class PE(nn.Module):
    def __init__(self,L):
        super().__init__()
        ## 00 11 22 33 .. .이런식으로 만들어놓으면 될듯
        ## 아 근데 그러면 그냥 [10,2] 이런식으로 되게 짠다음에  flatten 시키는게 더 좋을듯
        self.L = L
    
    def forward(self,p):
        sin = torch.sin(2**torch.arange(0,self.L,device=p.device) * torch.pi * p.unsqueeze(-1)) ## [B,L,3]
        cos = torch.cos(2**torch.arange(0,self.L,device=p.device) * torch.pi * p.unsqueeze(-1))
        
        ## 아하 명심하기 !!! 브로드캐스팅 기준은 제일 뒷차원부터 !!
        pos_enc = torch.cat([p.unsqueeze(-1),sin,cos],dim=-1).flatten(-2,-1) ## B,L,6 -> B,L*6
        return pos_enc


class NeRF(nn.Module):
    """
    A multi-layer perceptron (MLP) used for learning neural radiance fields.

    For architecture details, please refer to 'NeRF: Representing Scenes as
    Neural Radiance Fields for View Synthesis (ECCV 2020, Best paper honorable mention)'.

    Attributes:
        pos_dim (int): Dimensionality of coordinate vectors of sample points.
        view_dir_dim (int): Dimensionality of view direction vectors.
        feat_dim (int): Dimensionality of feature vector within forward propagation.
    """

    def __init__(
        self,
        pos_dim: int,
        view_dir_dim: int,
        feat_dim: int = 256,
    ) -> None:
        """
        Constructor of class 'NeRF'.
        """
        super().__init__()
        
        ## 일단, L에 따라서 pos encoding은 고정이니까 이걸 좀 만들어놔야 할듯 ???
        #self.pe1 = PE(L=10)
        #self.pe2 = PE(L=4)
        
        ## 일단 네트워크 인풋은 5D임

        temp = []
        for i in range(4):
            temp.extend([nn.Linear(feat_dim,feat_dim),nn.ReLU()])
            
        self.block1 = nn.Sequential(
            nn.Linear(pos_dim,feat_dim),
            nn.ReLU(),
            *temp
        )
        
        self.block2 = nn.Sequential(
            nn.Linear(feat_dim+pos_dim,feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim,feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim,feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim,feat_dim),         
        )
        
        self.block3 = nn.Sequential(
            nn.Linear(feat_dim+view_dir_dim,feat_dim//2),
            nn.ReLU(),
            nn.Linear(feat_dim//2,3),
            nn.Sigmoid()
        )
        
        self.intercept = nn.Linear(feat_dim,1)
        
    #@jaxtyped
    #@typechecked
    def forward(
        self,
        pos: Float[torch.Tensor, "num_sample pos_dim"],
        view_dir: Float[torch.Tensor, "num_sample view_dir_dim"],
    ) -> Tuple[Float[torch.Tensor, "num_sample 1"], Float[torch.Tensor, "num_sample 3"]]:
        """
        Predicts color and density.

        Given sample point coordinates and view directions,
        predict the corresponding radiance (RGB) and density (sigma).

        Args:
            pos: The positional encodings of sample points coordinates on rays.
            view_dir: The positional encodings of ray directions.

        Returns:
            sigma: The density predictions evaluated at the given sample points.
            radiance: The radiance predictions evaluated at the given sample points.
        """
        
        
        #pe1 = self.pe1(pos)
        #pe2 = self.pe2(view_dir)
        
        #print(pe1.shape)
        #print(pe2.shape)

        #print("hjahahahahahahaahahahahahahah")
        
        x = self.block1(pos)
        x = torch.cat([x,pos],dim=-1)
        x = self.block2(x)
        opacity = F.relu(self.intercept(x))

        x = torch.cat([x,view_dir],dim=-1)
        x = self.block3(x)
        rgb = x 

        #print(opacity.shape)
        #print(rgb.shape)
        
        return opacity,rgb



def test():
    pos = torch.randn(3,63)
    view_dir = torch.randn(3,27)
    
    model = NeRF(pos_dim=63,view_dir_dim=27)
    print(model(pos,view_dir))


#test()