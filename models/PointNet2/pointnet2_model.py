import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet_util import PointNetSetAbstraction, PointNetFeaturePropagation

class PointNet2(nn.Module):
    def __init__(self, cfg, num_classes):
        super(PointNet2, self).__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 6 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.decoders = nn.ModuleList()
        for num_class in self.num_classes:
            self.decoders.append(PointNet2Decoder(num_class, self.cfg))
            
    def forward(self, xyz):
        l0_points = torch.cat([xyz['points'], xyz['feats']], dim=-2)
        l0_xyz = xyz['points']
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        input_list = [l0_xyz, l0_points,
                      l1_xyz, l1_points,
                      l2_xyz, l2_points,
                      l3_xyz, l3_points,
                      l4_xyz, l4_points]
        preds = [decoder(*input_list) for decoder in self.decoders]
        return {'logits': [pred.permute(0, 2, 1) for pred in preds], 'sparse_logits':None}
            
class PointNet2Decoder(nn.Module):
    def __init__(self, num_class, cfg):
        super(PointNet2Decoder, self).__init__()
        self.cfg = cfg
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(cfg.TRAIN.DROPOUT_RATE)
        self.conv_out = nn.Conv1d(128, num_class, 1)

    def forward(self,   l0_xyz, l0_points,
                        l1_xyz, l1_points,
                        l2_xyz, l2_points,
                        l3_xyz, l3_points,
                        l4_xyz, l4_points):
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv_out(x)
        return x  
    
