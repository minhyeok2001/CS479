import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    ## 근데 나는 뭔가 matrix를 네트워크로 학습하도록 하고 그걸 곱하는 건줄 알았는데, 그런게 아니라 그냥 NN으로 하네? -> 그게 아니라,... identity matrix가 있어서 헷갈렸는데, 여기서는 곱해주는 matrix만 만드는거 맞음 !!
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,k,N]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.fc(x)
        
        # Followed the original implementation to initialize a matrix as I.
        identity = (
            Variable(torch.eye(self.k, dtype=torch.float))
            .reshape(1, self.k * self.k)
            .expand(B, -1)
            .to(device)
        )
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        ## 결국 out shape은 B,3,3꼴 
        return x


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """
    def __init__(
        self,
        input_transform: bool = False,
        feature_transform: bool = False,
        return_mid = False
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform
        self.return_mid = return_mid

        if self.input_transform:
            self.stn3 = STNKd(k=3)
        if self.feature_transform:
            self.stn64 = STNKd(k=64)

        # point-wise mlp
        
        self.pointwise_mlp_1 = nn.Sequential(nn.Conv1d(3,64,1), nn.Conv1d(64,64,1) ,nn.ReLU(inplace=True))
        self.pointwise_mlp_2 = nn.Sequential(nn.Conv1d(64,128,1),nn.ReLU(inplace=True))
        self.pointwise_mlp_3 = nn.Sequential(nn.Conv1d(128,1024,1),nn.ReLU(inplace=True))
        
        self.pooling = nn.AdaptiveMaxPool1d(1)
    
    
        # TODO : Implement point-wise mlp model based on PointNet Architecture.

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3] ## 여기서 N은 포인트클라우드 개수를 의미
        Output:
            - Global feature: [B,1024]
            - ...
        """
        
        ## 현재 pointcloud는 [B,N,3] 이니까, reshape
        
        B = pointcloud.shape[0]
        x = pointcloud.permute(0,2,1)
        transform1 = None
        transform2 = None
    
        if self.input_transform:
            transform1 = self.stn3(x)
            x = torch.bmm(pointcloud,transform1).permute(0,2,1)
            ## 이렇게 하고나면 차원은 B,3,N
            
        x = self.pointwise_mlp_1(x)
        ## x 차원은 B,64,N
        
        if self.feature_transform:            
            transform2 = self.stn64(x).permute(0,2,1)
            ## transform2 의 차원은 B 64,64
            x = torch.bmm(x.permute(0,2,1),transform2).permute(0,2,1)
            ## 이렇게 하고나면 차원은 B,64,N
        temp = x
        
        x = self.pointwise_mlp_2(x)
        x = self.pointwise_mlp_3(x)
        
        #print(x.shape)
        # 아마 shape은 B,1028,N 일 것임
        
        x = self.pooling(x).reshape(B,-1)

        ## [B,1028]
        
        if self.return_mid :
            return x, temp, transform1, transform2
        
        else :
            if self.input_transform or self.feature_transform :
                return x, transform1, transform2
            else :
                return x 

class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes
        
        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        
        # returns the final logits from the max-pooled features.
        
        self.mlp = nn.Sequential(nn.Linear(1024,512),nn.ReLU(inplace=True),nn.Linear(512,256),nn.ReLU(inplace=True),nn.Linear(256,self.num_classes)) 
        # TODO : Implement MLP that takes global feature as an input and return logits.

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - logits [B,num_classes]
            - ...
        """
        
        x,stn3,stn64= self.pointnet_feat(pointcloud)
        x = self.mlp(x)

        return x,stn3,stn64

class PointNetPartSeg(nn.Module):
    def __init__(self, m=50):
        super().__init__()

        # returns the logits for m part labels each point (m = # of parts = 50).
        # TODO: Implement part segmentation model based on PointNet Architecture.
        self.pointnet_feat = PointNetFeat(input_transform=True, feature_transform=True, return_mid=True)
        self.mlp = nn.Sequential(nn.Linear(1088,512),nn.ReLU(inplace=True),nn.Linear(512,256),nn.ReLU(inplace=True),nn.Linear(256,128),nn.ReLU(inplace=True),nn.Linear(128,m)) 

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - ...
        """
        
        x,temp,stn3,stn64= self.pointnet_feat(pointcloud)
        
        N = temp.shape[-1]
        # 지금 temp는 B,64,N
        
        
        # 현재 x 차원은 [B,1024]
        # 이걸 [B, N, 1024] 로 확장하고 temp [B, N, 64] 랑 더해서 B, N, 1088을 만들기
        
        x = x.unsqueeze(1).repeat(1,N,1)
        x = torch.cat([temp.permute(0,2,1),x],dim=-1)
        x = self.mlp(x)        
        
        return x,stn3,stn64

class PointNetAutoEncoder(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.num_points = num_points
        self.pointnet_feat = PointNetFeat(False,False,False)
        
        self.mlp1 = nn.Sequential(nn.Linear(1024, num_points//4), nn.BatchNorm1d(num_points//4), nn.ReLU(inplace=True))
        self.mlp2 = nn.Sequential(nn.Linear(num_points//4,num_points//2), nn.BatchNorm1d(num_points//2),nn.ReLU(inplace=True))
        self.mlp3 = nn.Sequential(nn.Linear(num_points//2,num_points),nn.BatchNorm1d(num_points),nn.Dropout(0.3), nn.ReLU(inplace=True))
        self.mlp4 = nn.Linear(num_points,num_points*3)        

        # Decoder is just a simple MLP that outputs N x 3 (x,y,z) coordinates.
        # TODO : Implement decoder.

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - pointcloud [B,N,3]
            - ...
        """
        B = pointcloud.shape[0]
        
        x= self.pointnet_feat(pointcloud)  # B 1024 
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        # B N*3
        
        x = x.reshape(B,-1,3)
        
        return x


def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()

"""

## test ## 

hola1 = PointNetCls(3,True,True)

hola2 = PointNetPartSeg()

hola3 = PointNetAutoEncoder(16)

dump = torch.randn(3,8,3)

print(dump.shape)

hola3(dump)

"""
