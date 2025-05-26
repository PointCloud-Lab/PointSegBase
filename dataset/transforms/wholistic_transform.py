from typing import Any
import numpy as np
from ..reader import HierarchicalMatrixReader
from .quantization import np_quantize


class __WholisticTransform(object):
    def __init__(self, **kwargs):
        pass 

    def __call__(self, point, feats, labels,**kwargs):
        return point,  feats, labels

    def __repr__(self):
        return self.__class__.__name__

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data_dict, **kwargs):
        for t in self.transforms:
            data_dict = t( **data_dict, **kwargs)
        return data_dict


class DropPaddingShuffle(__WholisticTransform):
    def __init__(self, num_points:int, seed:int, **kwargs) -> None:
        self.ramdom_machine = np.random.default_rng(seed=seed)
        self.num_points = num_points
        
    def __call__(self, points, feats, labels, extra_label=None,  **kwds: Any) -> Any:
        num_points = points.shape[0]
        indices = self.ramdom_machine.permutation(num_points)
        if num_points >= self.num_points:
            indices = indices[:self.num_points]
        else:
            offset = self.num_points - num_points
            indices = np.concatenate([indices, self.ramdom_machine.choice(indices, size=offset)])
        output = {"points": points[indices],"feats":  feats[indices], "labels":labels[indices], "extra_label": extra_label}
        return output
    
class Voxelization(__WholisticTransform):
    def __init__(self, voxel_size, feature_method: str, label_method:str,  num_classes=None, **kwargs) -> None:
        self.voxel_size = voxel_size
        self.feature_method = feature_method
        self.label_method = label_method
        self.num_classes = num_classes
        
    def __call__(self, points, feats, labels, extra_label=None,  **kwds: Any) -> Any:
        pts, un_feats, un_labels, inverse_inds = np_quantize(coords=points,
                    features=feats,
                    labels=labels,
                    ignore_label=-100,
                    quantize_size=self.voxel_size,
                    feature_method=self.feature_method,
                    label_method=self.label_method,
                    num_classes=self.num_classes,
                    )
        return   {
            "points":points, "feats":  feats, "labels":labels,
            "sparse_points": pts, "sparse_feats": un_feats, "sparse_labels": un_labels, "extra_label": extra_label, "inverse": inverse_inds, "ori_labels": labels}
    
    
class LimitFilter(__WholisticTransform):
    def __init__(self, limit_point_num:int, seed:int, **kwargs) -> None:
        self.limit_point_num = limit_point_num
        self.ramdom_machine = np.random.default_rng(seed=seed)
        
    def __call__(self, points, feats, labels, extra_label=None,  **kwds: Any) -> Any:
        num_points = points.shape[0]
        indices = self.ramdom_machine.permutation(num_points)
        if num_points >= self.limit_point_num:
            indices = indices[:self.limit_point_num]
        output = {"points": points[indices],"feats":  feats[indices], "labels":labels[indices], "extra_label": extra_label}
        return output
    
    
class SwapDim(__WholisticTransform):
    def __init__(self, **kwargs) -> None:
        pass
        
    def __call__(self, points, feats, labels, extra_label=None,  **kwds: Any) -> Any:
        return   {"points": points.transpose(), 
                  "feats": feats.transpose(), "labels": labels, "extra_label": extra_label}