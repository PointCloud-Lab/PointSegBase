from .loss import MultiHeirarchicalCrossEntropyLoss, ConsistencyLoss, HeirarchicalCrossEntropyLoss
from models.DGCNN.dgcnn_model import DGCNN
from models.PointNet2.pointnet2_model import PointNet2
from models.PointCNN.pointcnn_model import PointCNN
from models.ResUnet.backbones import SparseMultiResUNet42 as UNet
from models.RandLANet.randlanet_model import RandLANet as RandLA
import torch
from torch import nn
import os 

def build_loss(cfg, h_matrices):
    if cfg.TRAIN.LOSS_FUNCTION == 'HeirarchicalCrossEntropyLoss':
        losses= HeirarchicalCrossEntropyLoss(h_matrices=h_matrices, 
                                            ignore_label=cfg.TRAIN.TRAIN_IGNORE_LABEL, 
                                            layer_weights=cfg.TRAIN.LAYER_WEIGHTS)
        def loss_func(pred, target, label_weights, epoch, **kwargs):
            return losses(pred, target)
        return loss_func

    if cfg.TRAIN.LOSS_FUNCTION == 'ConsistencyLoss':
        if len(cfg.TRAIN.CONSISTENCY_LOSS_WEIGHT) != 2:
            raise ValueError("Consistency loss requires two weights.")
        else:
            if cfg.TRAIN.REG_FUNC == 'abs':
                reg_func = torch.abs
            elif cfg.TRAIN.REG_FUNC == 'relu':
                reg_func = torch.relu
            elif cfg.TRAIN.REG_FUNC == 'relu2':
                reg_func = lambda x: torch.relu(x)**2
            losses = {'MHCE': MultiHeirarchicalCrossEntropyLoss(h_matrices=h_matrices,
                                                              ignore_label=cfg.TRAIN.TRAIN_IGNORE_LABEL,
                                                              layer_weights=cfg.TRAIN.LAYER_WEIGHTS),
                    'CL': ConsistencyLoss(h_matrices=h_matrices,
                                          ignore_label=cfg.TRAIN.TRAIN_IGNORE_LABEL,
                                          layer_weights=cfg.TRAIN.LAYER_WEIGHTS,
                                          reg_func=reg_func)}
            def epoch_loss_func(pred, target, label_weights, epoch):
                if epoch < cfg.TRAIN.CONSISTENCY_LOSS_EPOCH:
                    loss = losses['MHCE'](pred, target, label_weights)
                else:
                    loss1 = losses['MHCE'](pred, target, label_weights)
                    loss2 = losses['CL'](pred)
                    print(loss1, loss2)
                    loss = loss1 * cfg.TRAIN.CONSISTENCY_LOSS_WEIGHT[0]  + loss2*cfg.TRAIN.CONSISTENCY_LOSS_WEIGHT[1]
                return loss
        return epoch_loss_func
    raise ValueError("Unknown loss function: {}".format(cfg.TRAIN.LOSS_FUNCTION))

def build_model(cfg, h_matrices, device):
    class_num = [h.shape[0] for h in h_matrices]
    if cfg.TRAIN.MODEL_NAME == 'DGCNN':
        model = DGCNN(cfg)
    elif cfg.TRAIN.MODEL_NAME == 'PointNet2':
        model = PointNet2(cfg, class_num)
    elif cfg.TRAIN.MODEL_NAME == 'PointCNN':
        model = PointCNN(cfg)
    elif cfg.TRAIN.MODEL_NAME == 'UNet':
        model = UNet(in_channels=3,
                     width_multiplier=1,
                     num_classes=class_num,)
    elif cfg.TRAIN.MODEL_NAME == 'RandLA':
        model = RandLA(d_in=6,
                       num_classes=class_num,
                       num_neighbors=16,
                       decimation=4,
                       device=device)
    else:
        raise ValueError("Unknown model name: {}".format(cfg.MODEL.NAME))

    if len(cfg.DEVICES.GPU_ID) > 1:
        model = nn.DataParallel(model, device_ids=cfg.DEVICES.GPU_ID)
    else:
        model = model.to(device)
    if len(cfg.TRAIN.PRETRAINED_MODEL_PATH)>0:
        assert os.path.isfile(cfg.TRAIN.PRETRAINED_MODEL_PATH), 'Not a label file'
        model.load_state_dict(torch.load(cfg.TRAIN.PRETRAINED_MODEL_PATH))
    return model

def build_opt(cfg, model):
    if cfg.TRAIN.OPTIMIZER == 'SGD':
        return torch.optim.SGD(model.parameters(), 
                               lr=cfg.TRAIN.LEARNING_RATE, 
                               momentum=cfg.TRAIN.MOMENTUM, 
                               weight_decay=cfg.TRAIN.WEIGHT_DECAY, 
                               nesterov=cfg.TRAIN.NESTEROV)
    elif cfg.TRAIN.OPTIMIZER == 'Adam':
        return torch.optim.Adam(model.parameters(), 
                                lr=cfg.TRAIN.LEARNING_RATE,
                                betas=(0.9, 0.999),
                                eps=1e-08,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unknown optimizer: {}".format(cfg.TRAIN.OPTIMIZER))

def build_scheduler(cfg, optimizer):
    if cfg.TRAIN.SCHEDULER == 'cos':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                          T_max=cfg.TRAIN.MAX_EPOCH,
                                                          eta_min=1e-4)
    elif cfg.TRAIN.SCHEDULER == 'step':
        clip_ = cfg.TRAIN.LEARNING_RATE_CLIP
        step_size = cfg.TRAIN.STEP_SIZE
        init_lr = cfg.TRAIN.LEARNING_RATE
        lr_decay = cfg.TRAIN.LEARNING_RATE_DECAY
        if clip_ is None:
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size, lr_decay)
        else:
            decay_lr_func = lambda epoch: max(init_lr * (lr_decay ** (epoch // step_size)), lr_decay)
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=decay_lr_func)
            
    elif cfg.TRAIN.SCHEDULER == 'exponential':
        lr_decay = cfg.TRAIN.LEARNING_RATE_DECAY
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)
    else:
        raise ValueError("Unknown scheduler: {}".format(cfg.TRAIN.SCHEDULER))

def build_label_weights(cfg):
    if cfg.TRAIN.DATASET == 'Campus3D':
        return [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    elif cfg.TRAIN.DATASET == 'Urban3D':
        return [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    elif cfg.TRAIN.DATASET == 'Partnet':
        return [
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    