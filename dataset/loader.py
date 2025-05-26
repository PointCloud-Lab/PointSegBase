from __future__ import division
from __future__ import print_function
from .reader import read_file_list
from .dataset import SceneDataset
from torch.utils import data as D
import torch
from torchsparse.utils.collate import sparse_collate 
from torchsparse import SparseTensor
from sampling import point_samplers, scene_samplers

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return {key: torch.stack([torch.from_numpy(d[key]) for d in batch]) for key in batch[0] if not (batch[0][key] is None)}

def sparse_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    result = {}
    for key in batch[0]:
        result[key] = [torch.from_numpy(d[key]) for d in batch if not (d[key] is None)]  # 对于 'inverse'，只返回列表
    sparse_points = []
    sparse_labels = []
    for coords, feats, labels in zip(result['sparse_points'],result['sparse_feats'],result['sparse_labels']):
        sparse_points.append(SparseTensor(coords=coords, feats=feats))
        sparse_labels.append(SparseTensor(coords=coords, feats=labels))
    result['sparse_points'] = sparse_collate(sparse_points)
    result['sparse_labels'] = sparse_collate(sparse_labels)
    return result

def numpy_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return {key: [d[key] for d in batch] for key in batch[0]}

    
def create_dataloader(dataset, batch_size, num_workers, **kwargs):
    shuffle = dataset.is_training
    if not dataset.voxelized:
        TorchDataLoader = D.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=collate_fn, **kwargs)
    else:
        TorchDataLoader = D.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=sparse_collate_fn,**kwargs)
    return TorchDataLoader

def create_dataset(cfg, set="TRAIN"):
    data_list = read_file_list(cfg.data.data_list_file, 
                       sets=set, 
                       use_color=cfg.dataset.use_color,
                       color_channel=(3, 4, 5)) 
    sample_cfg = cfg.dataset.sample
    if set in ['TRAIN']:
        istraining = True
        if sample_cfg.point_sampler_type == 'BlockSampler':
            point_sampler_type = 'RandomBlockSampler'
        elif sample_cfg.point_sampler_type == 'KnnSampler':
            point_sampler_type = 'RandomKnnSampler'
        else:
            flag = getattr(point_samplers,sample_cfg.point_sampler_type, None)
            if flag is None:
                raise NotImplementedError('Unknown scene sampler type: {}'.format(sample_cfg.point_sampler_type))
            point_sampler_type = sample_cfg.point_sampler_type 
        
        if sample_cfg.scene_sampler_type == 'SceneSampler':
            scene_sampler_type = 'RandomSceneSampler'
        else:
            flag = getattr(scene_samplers,sample_cfg.scene_sampler_type, None)
            if flag is None:
                raise NotImplementedError('Unknown point sampler type: {}'.format(sample_cfg.scene_sampler_type))
            scene_sampler_type = sample_cfg.scene_sampler_type 
    elif set in ['TEST', 'VALIDATION']:
        istraining = False
        if sample_cfg.point_sampler_type == 'BlockSampler':
            point_sampler_type = 'SlidingBlockSampler'
        elif sample_cfg.point_sampler_type == 'KnnSampler':
            point_sampler_type = 'TraversalKnnSampler'
        else:
            flag = getattr(point_samplers,sample_cfg.point_sampler_type, None)
            if flag is None:
                raise NotImplementedError('Unknown point sampler type: {}'.format(sample_cfg.point_sampler_type))
            point_sampler_type = sample_cfg.point_sampler_type 
        if sample_cfg.scene_sampler_type == 'SceneSampler':
            scene_sampler_type = 'SequentialSceneSampler'
        else:
            flag = getattr(scene_samplers,sample_cfg.scene_sampler_type, None)
            if flag is None:
                raise NotImplementedError('Unknown scene sampler type: {}'.format(sample_cfg.scene_sampler_type))
            scene_sampler_type = sample_cfg.scene_sampler_type 
    else:
        raise NotImplementedError('Unknown set: {}'.format(set))
    
    transform_cfg = cfg.dataset.transforms
    dataset = SceneDataset(data_list, 
                           point_sampler_type=point_sampler_type, 
                           scene_sampler_type=scene_sampler_type,
                           sampler_settings=sample_cfg.setting,
                           point_transform_type=transform_cfg.point_transforms, 
                           data_transform_type=transform_cfg.data_transforms,
                           transform_settings=transform_cfg.setting,
                           seed=cfg.dataset.random_seed_basis,
                           istraining=istraining
                           )
    dataset.voxelized = "Voxelization" in transform_cfg.data_transforms
    return dataset

