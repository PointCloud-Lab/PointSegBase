import torch
import torch.nn as nn
from .pointcnn_util import RandPointCNN, knn_indices_func_gpu

# C_in, C_out, D, N_neighbors, dilution, N_rep, r_indices_func, C_lifted = None, mlp_width = 2
# (a, b, c, d, e) == (C_in, C_out, N_neighbors, dilution, N_rep)
# Abbreviated PointCNN constructor.
AbbPointCNN = lambda a, b, c, d, e: RandPointCNN(a, b, 6, c, d, e, knn_indices_func_gpu)

class PointCNN(nn.Module):
    def __init__(self, cfg):
        super(PointCNN, self).__init__()
        self.cfg = cfg
        self.num_class = cfg.DATASET.DATA.LABEL_NUMBER
        self.pcnn1 = AbbPointCNN(0, 32, 8, 1, -1)
        self.pcnn2 = nn.Sequential(
            AbbPointCNN(32, 64, 8, 2, 2048),
            AbbPointCNN(64, 96, 8, 4, 512),
            AbbPointCNN(96, 128, 12, 4, 128)
        )


        self.dcnn1 = nn.Sequential(
            AbbPointCNN(128, 96, 12, 4, 256),
            AbbPointCNN(96, 64, 8, 4, 512),
            AbbPointCNN(64, 32, 8, 2, 2048)
        )
        
        self.bn = nn.BatchNorm1d(32)
        self.dp = nn.Dropout(p=self.cfg.TRAIN.DROPOUT_RATE)
        self.dcnn2_0 = nn.Conv1d(32, self.num_class[0], kernel_size=1, bias=False)
        self.dcnn2_1 = nn.Conv1d(32, self.num_class[1], kernel_size=1, bias=False)
        self.dcnn2_2 = nn.Conv1d(32, self.num_class[2], kernel_size=1, bias=False)
        self.dcnn2_3 = nn.Conv1d(32, self.num_class[3], kernel_size=1, bias=False)
        self.dcnn2_4 = nn.Conv1d(32, self.num_class[4], kernel_size=1, bias=False)
        

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pcnn1(x)
        x = self.pcnn2(x)
        x = self.dcnn1(x)
        x = x[1].permute(0, 2, 1)
        x = self.bn(x)
        x = self.dp(x)
        x0 = self.dcnn2_0(x)
        x1 = self.dcnn2_1(x)
        x2 = self.dcnn2_2(x)
        x3 = self.dcnn2_3(x)
        x4 = self.dcnn2_4(x)
        return [x0, x1, x2, x3, x4]
