from .sampling import point_samplers
from .sampling  import scene_samplers
from .transforms import point_transform
from .transforms import wholistic_transform
from torch.utils.data import Dataset
import numpy as np

def build_transforms(data_transform, transform_types, settings,seed):
    if data_transform:
        transform = wholistic_transform
    else:
        transform = point_transform
    settings.update({"SEED": seed})
    types = transform_types
    if len(types) == 0:
        return transform.Compose([])
    else:
        transforms = []
        for i, type in enumerate(types):
            transform_gen = getattr(transform, type, None)
            if transform_gen is None:
                raise NotImplementedError('Transform {} not implemented.'.format(type))
            transforms.append(transform_gen(**{k.lower():v for k, v in settings.items()}))
        return transform.Compose(transforms)  
    

def build_scene_sampler(point_sampler_list,
                        scene_sampler_type,
                        setting, 
                        seed):
    sampler_gen = getattr(scene_samplers, scene_sampler_type, None)
    print(scene_samplers)
    if sampler_gen is None:
        raise NotImplementedError('Scene sampler {} not implemented.'.format(scene_sampler_type))
    subsampler_lengths = [len(sampler) for sampler in point_sampler_list]
    return  sampler_gen(
            subsampler_lengths=subsampler_lengths,
            seed=seed,
            **setting.lower_case_copy())
  
def build_point_samplers(data_list, point_sampler_type, setting, seeds):
    def gen_sampler(gen_func, file_data, setting, seed):
        return gen_func(
            reader=file_data,
            seed=seed,
            params = setting)
    sampler_gen = getattr(point_samplers, point_sampler_type, None)
    if sampler_gen is None:
        raise NotImplementedError('Point sampler {} not implemented.'.format(point_sampler_type))
    list_sampler = [gen_sampler(sampler_gen, file_data, setting, seeds[i])
                    for i, file_data in enumerate(data_list)]
    print("len:", [len(l) for l in list_sampler])
    return list_sampler


class SceneDataset(Dataset):
    '''
    Sampler for data of scenes
    dataset sampler ->  [sampler_1(scene_2) ... sampler_n(scene_n)]-> randomly select a sampler -> sample

    class Type
        set_name = str
        list_sampler = [Sampler, ]
        is_training = bool
        num_classes = [int, ]
        label_distribution = [np.ndarray, ]
        label_weights = [np.ndarray, ]
        random_machine = data_utils.random_machine.RandomMachine

    __init__(
            set_name: str,
            params: AttrDict,
            is_training: bool {default: True},
            )

    __getitem__(index: int)
        Returns:
            points_centered, labels, colors, raw_points:
            np.ndarray, np.ndarray, np.ndarray, np.ndarray

    __getattr__(attr: str)
        Returns:
            list of attribute: [sampler.attr, ]

    '''
    def __init__(self, 
                 reader_list, 
                 point_sampler_type, 
                 scene_sampler_type,
                sampler_settings,
                point_transform_type, 
                data_transform_type,
                transform_settings,
                 seed=0,
                 istraining=True
                 ):

        # Dataset parameters
        self.is_training = istraining
        self.data_list = reader_list
        self.point_sampler_type = point_sampler_type 
        self.scene_sampler_type = scene_sampler_type 

        random_machine = np.random.RandomState(seed)
        seeds = [random_machine.randint(1e6) for _ in range(len(self.data_list)+1)]

        self.point_samplers = build_point_samplers(self.data_list, 
                                                   self.point_sampler_type, sampler_settings, 
                                                   seeds[:-1])
        
        self.scene_sampler = build_scene_sampler(self.point_samplers, 
                                                 self.scene_sampler_type, 
                                                 sampler_settings, 
                                                 seeds[-1])
        
        self.__length = len(self.scene_sampler)

        self.point_transforms = build_transforms(False,
                                                 point_transform_type,
                                                 transform_settings,
                                                 seeds[-1]) 
                                                   
        self.transforms = build_transforms(True,
                                    data_transform_type,
                                    transform_settings,
                                    seeds[-1]) 
        self.voxelized = False

    def __len__(self):
        return self.__length

    def __getitem__(self, ind):
        assert isinstance(ind, int) or isinstance(ind, np.int), \
            'index must be int.'
        assert ind < len(self), 'index out of bound {}.'.format(len(self))
        
        scene_index, ind_in_scene = self.scene_sampler[ind]
        scene_points = self.data_list[scene_index].points
        point_indices = self.point_samplers[scene_index].get_indices(ind_in_scene, scene_points)
        if point_indices is None:
            return None
        else:
            points = self.data_list[scene_index].points[point_indices]         
            if not(self.point_transforms is None):
                points = self.point_transforms(points, 
                                               ind_in_scene=ind_in_scene, 
                                               scene_points=scene_points)
                
            
            if self.data_list[0].colors is not None:
                feats = self.data_list[scene_index].colors[point_indices]
            else:
                feats = points.copy()
                
            extra_label= getattr(self.data_list[scene_index], 'etra_label', None)
            labels = self.data_list[scene_index].labels[point_indices]

            self.transforms = self.transforms if self.transforms else lambda x: x
            return self.transforms({"points": points, 
                                    "feats": feats, 
                                    "labels": labels, 
                                    "extra_label": extra_label,
                                    })
                
