from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def logits_process(logits:list):
    if not isinstance(logits, (list, tuple)):
        logits = [logits]
    if not isinstance(logits[0],torch.Tensor):
        return [logit.F for logit in logits]
    else:
        return logits

def target_process(target):
    if not isinstance(target, torch.Tensor):
        return target.F
    else:
        return target

class MultiHeirarchicalCrossEntropyLoss(object):
    def __init__(self, h_matrices, ignore_label=-100, layer_weights=None, device="cuda"):
        super(MultiHeirarchicalCrossEntropyLoss, self).__init__()
        self.h_matrices = [torch.from_numpy(m).float().to(device) for m in h_matrices]
        self.gather_id = [torch.from_numpy(np.argmax(m, axis=0)).to(device) for m in h_matrices]
        self.ignore_label = ignore_label
        self.device = device
        if layer_weights is None:
            self.layer_weights = [1.0] * len(self.h_matrices)
        else:
            assert len(layer_weights) == len(self.h_matrices), "Number of layer weights must match number of matrices"
            self.layer_weights = layer_weights

    def __call__(self, pred, target, weight=None):
        total_loss = torch.Tensor([0.0]).to(self.device).float()

        pred = logits_process(pred)
        target = target_process(target)

        if weight is not None and not isinstance(weight, (list, tuple)):
            weight = [weight]
        
        target_list = [g[target] for g in self.gather_id]

        for i, (single_pred, single_target) in enumerate(zip(pred, target_list)):
            if len(single_pred.shape) == 3:
                B, N, C = single_pred.shape
                single_pred = single_pred.reshape(B*N, C)
                single_target = single_target.reshape(B*N)

            current_weight = None 
            loss = nn.functional.cross_entropy(single_pred, single_target, weight=current_weight, ignore_index=self.ignore_label)
            
            total_loss += loss * self.layer_weights[i]

        return total_loss / len(self.h_matrices)


class ConsistencyLoss(object):
    def __init__(self, h_matrices, ignore_label=-100, layer_weights=None, device='cuda', reg_func=torch.abs, mode='inter'):
        super(ConsistencyLoss, self).__init__()
        self.ignore_label = ignore_label
        self.h_matrices = [torch.from_numpy(m).float().to(device) for m in h_matrices]
        
        self.inter_h_matrices = []
        for i in range(0, len(h_matrices)-1):
            nm = np.clip(h_matrices[i]@h_matrices[i+1].T, 0., 1.0)
            self.inter_h_matrices.append(torch.from_numpy(nm).float().to(device))
        
        if layer_weights is None:
            self.layer_weights = [1.0] * len(self.h_matrices)
        else:
            assert len(layer_weights) == len(self.h_matrices), "Number of layer weights must match number of matrices"
            self.layer_weights = layer_weights
        self.device = device
        self.reg_func = reg_func
        self.mode = mode
        
    def flatten(self, x):
        if len(x.shape) == 3:
            B, N, C = x.shape
            return x.reshape(B*N, C)
        else:
            return x

    def __call__(self, pred):
        pred = logits_process(pred)

        assert len(pred) == len(self.h_matrices), "Mismatch in number of prediction layers and hierarchical matrices."

        reshaped_sd = self.flatten(pred[-1])
        sd = F.softmax(reshaped_sd, dim=-1)
        total_loss = torch.Tensor([0.0]).to(self.device)
        probs = [F.softmax(self.flatten(p), dim=-1) for p in pred]
        
        if self.mode == 'inter':
            for i in range(len(pred) - 1):
                su, sd = probs[i], probs[i+1]
                mapped_sd = torch.matmul(sd, self.inter_h_matrices[i].T)
                diff = self.reg_func(su - mapped_sd)
                total_loss += self.layer_weights[i] * torch.mean(diff)
        else:
            sd = probs[-1]
            for i in range(len(pred) - 1):
                su = probs[i]
                mapped_sd = torch.matmul(sd, self.h_matrices[i].T)
                diff = self.reg_func(su - mapped_sd)
                total_loss += self.layer_weights[i] * torch.mean(diff)

        return total_loss / (len(self.h_matrices) - 1)
    
    
class HeirarchicalCrossEntropyLoss(object):
    def __init__(self, h_matrices, ignore_label=-100, layer_weights=None, device="cuda"):
        super(HeirarchicalCrossEntropyLoss, self).__init__()
        self.ignore_label = ignore_label
        self.aggr_m = [torch.from_numpy(m).float().to(device) for m in h_matrices]
        append100=lambda x: torch.cat([x, torch.full((200,),-100).to(x)])
        self.gather_id = [torch.from_numpy(np.argmax(m, axis=0)).to(device) for m in h_matrices]
        if layer_weights is None:
            self.layer_weights = [1.0] * len(self.aggr_m)
        else:
            assert len(layer_weights) == len(self.aggr_m), "Number of layer weights must match number of matrices"
            self.layer_weights = layer_weights
        
    def __call__(self, pred, target, weight=None):
        pred = logits_process(pred)
        target = target_process(target)
        if len(pred[-1].shape) == 3:
            B, N, C = pred[-1].shape
            reshaped_pred = pred[-1].reshape(B*N, C)
            reshaped_target = target.reshape(B*N)
        else:
            reshaped_pred = pred[-1]
            reshaped_target = target

        pred_max =  reshaped_pred.max(-1)[0].unsqueeze(-1)
        exp_preds = torch.exp(reshaped_pred - pred_max) + 1e-12
        log_sum_exp_preds = exp_preds.sum(-1).log().unsqueeze(-1)
        
        loss = torch.Tensor([0.0]).to(reshaped_pred).float()
        included_indices = list(range(len(self.aggr_m)))

        for i, (m, gid) in enumerate(zip(self.aggr_m, self.gather_id)): 
            if i not in included_indices:
                continue
            target_ = gid[reshaped_target]
            log_sum_exp_preds_d = torch.matmul(exp_preds, m.T).log()
            log_prob = log_sum_exp_preds_d - log_sum_exp_preds
            current_weight = weight[i] if weight is not None else None
            layer_loss = nn.functional.nll_loss(log_prob, target_, weight=current_weight, ignore_index=self.ignore_label)
            
            loss += layer_loss * self.layer_weights[i]

        return loss[0]/len(self.aggr_m)
