from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation

class __PointTransform(object):
    def __init__(self):
        pass 

    def __call__(self, point, **kwargs):
        return point 

    def __repr__(self):
        return self.__class__.__name__

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, points, **kwargs):
        for t in self.transforms:
            points = t(points, **kwargs)
        return points
    
class BlockCenteralization(__PointTransform):
    def __init__(self, block_size_x, block_size_y, **kwargs) -> None:
        self.block_size_x = block_size_x
        self.block_size_y = block_size_y
        
    def __call__(self, points,  **kwds: Any) -> Any:
        box_min = np.min(points, axis=0)
        shift = np.array([box_min[0] + self.block_size_x/ 2,
                            box_min[1] + self.block_size_y / 2,
                            box_min[2]])
        return points - shift
    
class GlobalNormalization(__PointTransform):
    def __init__(self, max_bounds, min_bounds, **kwargs) -> None:
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
    
    def __call__(self, points,  **kwds: Any) -> Any:
        bounds = self.max_bounds - self.min_bounds
        return (points - self.min_bounds) / bounds

class SphericalNormalization(__PointTransform):
    def __init__(self,  **kwargs) -> None:
        pass
    
    def __call__(self, points,  **kwds: Any) -> Any:
        center = kwds.get('center', np.mean(points, axis=0))
        bounds = np.max(np.linalg.norm(points - center, axis=1))
        return (points - center) / bounds
    

    
class LocalNormalization(__PointTransform):
    def __init__(self, scale_strategy="postive", **kwargs) -> None:
        self.scale_strategy = scale_strategy
    
    def __call__(self, points,  **kwds: Any) -> Any:
        max_bounds = kwds.get('max_bounds', np.max(points, axis=0))
        min_bounds =  kwds.get('min_bounds',np.min(points, axis=0))
        if self.scale_strategy == 'positive':
            bounds = max_bounds - min_bounds
            return (points - self.min_bounds) / bounds
        elif self.scale_strategy == "centered":
            center = (max_bounds + min_bounds) / 2
            bounds = max_bounds - min_bounds
            return (points - center) / bounds
    
class Centeralization(__PointTransform):
    def __init__(self, dim=[0 ,1], **kwargs) -> None:
        self.dim = dim 
        
    def __call__(self, points,  **kwds: Any) -> Any:
        center = np.zeros(3, dtype=points.dtype)
        ind_in_scene = kwds.get('ind_in_scene', None)
        scene_points = kwds.get('scene_points', None)
        if ind_in_scene is None:
            center[self.dim] = np.mean(points[:, self.dim], axis=0)
        else:
            center[self.dim] = scene_points[ind_in_scene, self.dim]
        return points - center

class  Scaling(__PointTransform):
    """Scale augmentation for pointcloud.

    If `scale_anisotropic` is True, each point is scaled differently.
    else, same scale from range ['min_s', 'max_s') is applied to each point.

    Args:
        pc: Pointcloud to scale.
        cfg: configuration dict.

    """

    def __init__(self, scale, **kwargs) -> None:
        self.scale = scale
        
    def __call__(self, points, **kwargs):
        return points * self.scale
    

class  RandomScaling(__PointTransform):
    """Scale augmentation for pointcloud.

    If `scale_anisotropic` is True, each point is scaled differently.
    else, same scale from range ['min_s', 'max_s') is applied to each point.

    Args:
        pc: Pointcloud to scale.
        cfg: configuration dict.

    """
    def __init__(self, min_scale, max_scale, **kwargs) -> None:
        self.min_s = min_scale
        self.max_s = max_scale
        
    def __call__(self, points):
        scale = ((self.max_s / self.min_s)**self.rng.random()) * self.min_s
        return points * scale
    
class RandomRotation(__PointTransform):
    def __init__(self, max_angle=np.pi, **kwargs) -> None:
        self.max_angle = max_angle
        
    def __call__(self, points,**kwargs):
        
        """Rotate the pointcloud.

        Two methods are supported. `vertical` rotates the pointcloud
        along yaw. `all` randomly rotates the pointcloud in all directions.

        Args:
            pc: Pointcloud to augment.
            cfg: configuration dictionary.

        """
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.rand() * self.max_angle
        r = Rotation.from_rotvec(angle * axis)
        R = r.as_matrix().astype(points.dtype)
        
        center = kwargs.get('center', None)
        if center is None:
            center = np.mean(points, axis=0)
        return np.dot(points - center, R.T) + center
    
class VerticalRotation(__PointTransform):
    def __init__(self, max_angle=2*np.pi, **kwargs) -> None:
        self.max_angle = max_angle
        
    def __call__(self, points, **kwargs):
        
        """Rotate the pointcloud.

        Two methods are supported. `vertical` rotates the pointcloud
        along yaw. `all` randomly rotates the pointcloud in all directions.

        Args:
            pc: Pointcloud to augment.
            cfg: configuration dictionary.

        """
        
        theta = self.rng.random() * self.max_angle
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=points.dtype)
        
        center = kwargs.get('center', None)
        if center is None:
            center = np.mean(points, axis=0)
        return np.matmul(points - center, R) + center

Raw = __PointTransform 
