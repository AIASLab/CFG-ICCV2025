import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG

from Model.PointNetPP_ssg import PointNet2ClassificationSSG


class PointNet2ClassificationMSG(PointNet2ClassificationSSG):
    def _build_model(self):
        super()._build_model()
        if self.use_normal:
            ori_channel = 3
        else:
            ori_channel = 0
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.1, 0.2, 0.4],
                nsamples=[16, 32, 128],
                mlps=[[ori_channel, 32, 32, 64], [ori_channel, 64, 64, 128], [ori_channel, 64, 96, 128]],
                use_xyz=self.use_xyz,
            )
        )

        input_channels = 64 + 128 + 128
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[
                    [input_channels, 64, 64, 128],
                    [input_channels, 128, 128, 256],
                    [input_channels, 128, 128, 256],
                ],
                use_xyz=self.use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[128 + 256 + 256, 256, 512, 1024],
                use_xyz=self.use_xyz,
            )
        )

    def features_grad(self, pointcloud):
        
        if self.use_normal:
            ori_channel = 3
        else:
            ori_channel = 0
        

        sa1 = PointnetSAModuleMSG(
                npoint=512,
                radii=[0.1, 0.2, 0.4],
                nsamples=[16, 32, 128],
                mlps=[[ori_channel, 32, 32, 64], [ori_channel, 64, 64, 128], [ori_channel, 64, 96, 128]],
                use_xyz=self.use_xyz,
            )
        sa1 = sa1.cuda()

        input_channels = 64 + 128 + 128
        sa2 = PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[
                    [input_channels, 64, 64, 128],
                    [input_channels, 128, 128, 256],
                    [input_channels, 128, 128, 256],
                ],
                use_xyz=self.use_xyz,
            )
        sa2 = sa2.cuda()
        sa3 = PointnetSAModule(
                mlp=[128 + 256 + 256, 256, 512, 1024],
                use_xyz=self.use_xyz,
            )
        sa3 = sa3.cuda()
        
        fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),         
        )
        fc_layer = fc_layer.cuda()

        pointcloud = pointcloud.transpose(2, 1)
        xyz, features = self._break_up_pc(pointcloud)
        xyz, y = sa1(xyz, features)
        
        y.retain_grad()

        xyz, z = sa2(xyz, y)                                  
        xyz, z = sa3(xyz, z)
        z = self.fc_layer(z.squeeze(-1))
        return z,y
    
    def layer2_features(self, pointcloud):

        if self.use_normal:
            ori_channel = 3
        else:
            ori_channel = 0
        

        sa1 = PointnetSAModuleMSG(
                npoint=512,
                radii=[0.1, 0.2, 0.4],
                nsamples=[16, 32, 128],
                mlps=[[ori_channel, 32, 32, 64], [ori_channel, 64, 64, 128], [ori_channel, 64, 96, 128]],
                use_xyz=self.use_xyz,
            )
        sa1 = sa1.cuda()

        input_channels = 64 + 128 + 128
        sa2 = PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[
                    [input_channels, 64, 64, 128],
                    [input_channels, 128, 128, 256],
                    [input_channels, 128, 128, 256],
                ],
                use_xyz=self.use_xyz,
            )
        sa2 = sa2.cuda()
        sa3 = PointnetSAModule(
                mlp=[128 + 256 + 256, 256, 512, 1024],
                use_xyz=self.use_xyz,
            )
        sa3 = sa3.cuda()
        
        fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),         
        )
        fc_layer = fc_layer.cuda()

        pointcloud = pointcloud.transpose(2, 1)
        xyz, features = self._break_up_pc(pointcloud)
        xyz, y = sa1(xyz, features)
        
        return y

