import numpy as np
from .qurey import BlockQuery, KnnQuery
import torch.multiprocessing as mp


class BasicSampler(object):
    """
    Sampler template
    One can create new sampler based on this template

    class Type:
        reader = FileDataReader
        is_training = Bool
        modify_type = [str,]
        modify_dunc = PointModifier

    __init__(
            reader: .reader.FileDataReader
            params: AttrDict
            is_training: bool {default: True}
            *args,
            **kwrgs
            )

    sample(index: int, set_random_machine: np.random.RandomState {default: None}, *args, **kwargs)
        Return:
            points_modified, points, labels, colors:
            np.ndarray, np.ndarray, np.ndarray, np.ndarray

    cal_length()
        Return:
            length of sampler: int

    """

    def __init__(self, reader, params, is_training=True, seed=0, *args, **kwargs):
        _update_from_config(self, params)
        self.is_training = is_training
        self.random_machine = np.random.RandomState(seed)
        self._length = 1
 
    def get_indices(self, ind, points=None, *args, **kwargs):
        return slice(None, None)
    
    def __len__(self):
        return self._length


class ObjectSampler(object):
    def __init__(self, reader, params, is_training=True, seed=0, *args, **kwargs):
        _update_from_config(self, params)
        self.is_training = is_training
        self._length = len(reader)
 
    def get_indices(self, ind, points=None, *args, **kwargs):
        return ind
    
    def __len__(self):
        return self._length



def _update_from_config(obj, cfg):
    for k in obj.__dict__.keys():
        try:
            obj.__dict__[k] = cfg[k.upper()]
        except KeyError:
            raise KeyError("\'{}\' has not been defined in config file".format(k.upper()))
        except Exception as e:
            raise Exception(e)
        

class RandomBlockSampler(BasicSampler):
    def __init__(self, reader, params, is_training=True, seed=0, **kwargs):
        self.block_size = (12,12)
        self.sparse_thresh_num_points = 2000
        self.search_module = None
        self.rebuild_tree = False
        super().__init__(*[reader, params, is_training, seed])

        self.random_machine = np.random.RandomState(seed)
        self._build_query(reader)
        self._length = len(reader)

    def _build_query(self, reader):
        if reader.search_tree_2d is None or  self.rebuild_tree:
            self.search_module = reader.build_2d_tree(grid_size=self.block_size)
        else:
            self.search_module = reader.search_tree_2d
        self.q = BlockQuery(self.search_module,
                        self.block_size)
        
    def get_indices(self, ind, points, *args, **kwargs):
        assert ind < len(points), "Index out of range"
        center_point = points[ind]
        return self._extract_block(center_point, points)

    def _extract_block(self, center_point, points):
        indices = self.q.search(center_point,  points)
        if len(indices) <= self.sparse_thresh_num_points:
            return None
        return indices

class SlidingBlockSampler(BasicSampler):
    def __init__(self, reader, params, is_training=True, seed=0, sliding_ratio=0.5,**kwargs):
        self.block_size = (12, 12)
        self.sparse_thresh_num_points = 2000
        self.sliding_ratio = sliding_ratio
        self.rebuild_tree = False
        super().__init__(*[reader, params, is_training, seed])
        self._build_query(reader)
        self.center_points = self._generate_center_points(reader.min_bounds, reader.max_bounds, reader.points)
        self._length = len(self.center_points)

    def _build_query(self, reader):
        if reader.search_tree_2d is None or  self.rebuild_tree:
            print('Rebuliding search tree.....')
            self.search_module = reader.build_2d_tree(grid_size=self.block_size)
        else:
            self.search_module = reader.search_tree_2d
        self.q = BlockQuery(self.search_module,
                        self.block_size)
    
    def _generate_center_points(self, min_bounds, max_bounds, points):
        x_min, y_min, _ = min_bounds
        x_max, y_max, _ = max_bounds
        x_size, y_size = self.block_size
        slide_x, slide_y = self.block_size[0] * self.sliding_ratio, self.block_size[1] * self.sliding_ratio
        
        centers = []
        x_center = x_min + x_size / 2
        while x_center - x_size / 2 <= x_max:
            y_center = y_min + y_size / 2
            while y_center - y_size / 2 <= y_max:
                z_center = (min_bounds[2] + max_bounds[2]) / 2
                center = np.asarray([x_center, y_center, z_center])

                indices = self.q.search(center, points)
                if len(indices) >  self.sparse_thresh_num_points:
                    centers.append(center)
                
                y_center += slide_y
            x_center += slide_x
        
        return centers
        
    def get_indices(self, ind, points, *args, **kwargs):
        c_center = ind % len(self.center_points)
        center_point = self.center_points[c_center]
        return self._extract_block(center_point, points)

    def _extract_block(self, center_point, points):
        indices = self.q.search(center_point,  points)
        if len(indices) <= self.sparse_thresh_num_points:
            return None
        return indices


class RandomKnnSampler(BasicSampler):
    def __init__(self, reader, params, is_training=True, seed=0, **kwargs):
        self.num_points = 2048
        self.search_module = None
        self.rebuild_tree = False
        super().__init__(*[reader, params, is_training, seed])

        self.random_machine = np.random.RandomState(seed)
        self._build_query(reader)
        self._length = len(reader)

    def _build_query(self, reader):
        if reader.search_tree_3d is None or  self.rebuild_tree:
            print('Rebuliding search tree.....')
            self.search_module = reader.build_3d_tree()
        else:
            self.search_module = reader.search_tree_3d
        self.q = KnnQuery(self.search_module)
        
    def get_indices(self, ind, points, *args, **kwargs):
        assert ind < len(points), "Index out of range"
        center_point = points[ind]
        return self.q.search(center_point, self.num_points)
    
class TraversalKnnSampler(BasicSampler):
    def __init__(self, reader, params, is_training=True, seed=0, **kwargs):
        self.num_points = 2048
        self.traverse_time = 1.0
        self.search_module = None
        self.rebuild_tree = False
        super().__init__(*[reader, params, is_training, seed])

        self.random_machine = np.random.RandomState(seed)
        self._build_query(reader)
        self._length = int(len(reader) / self.num_points * self.traverse_time)
        self.possibilities = mp.Array('d', (self.random_machine.rand(len(reader)) * 1e-3).tolist())
        self.lock = mp.Lock()

    def _build_query(self, reader):
        if reader.search_tree_3d is None or  self.rebuild_tree:
            print('Rebuliding search tree.....')
            self.search_module = reader.build_3d_tree()
        else:
            self.search_module = reader.search_tree_3d
        self.q = KnnQuery(self.search_module)
        
    def get_indices(self, ind, points, *args, **kwargs):
        ind = np.argmin(self.possibilities)
        center_point = points[ind]
        searched_indices = self.q.search(center_point, self.num_points)
        self.random_machine.shuffle(searched_indices)
        pc = points[searched_indices]
        dists = np.sum(np.square((pc - center_point)),
                    axis=1)
        delta = np.square(1 - dists / np.max(dists))
        with self.lock:

            for i, ind in enumerate(searched_indices):
                self.possibilities[ind] += delta[i]
            return searched_indices
            
        
    '''
    def get_indices(self, ind, set_random_machine=None, *args, **kwargs):
        random_machine = self.random_machine if set_random_machine is None else set_random_machine
        ind, center_point = self._sample_index(ind, random_machine)
        if not self.return_index:
            points, colors, labels = self.reader[ind]
            points_centered = self.modify_points(points, center=center_point)
            #print(points_centered.shape)
            return points_centered, points, labels, colors
        else:
            return None
            
    def _sample_center_index(self, ind, random_machine):
        return random_machine.randint(0, len(self.reader))

    def _sample_index(self, ind, random_machine):
        if self.is_training:
            center_index = self._sample_center_index(ind, random_machine)
            center_point = self.reader.points[center_index]
        else:
            center_point = self.center_list[ind]
        _, neighbour_ind = self.q.search(np.expand_dims(center_point, axis=0),
                                        self.num_points_per_sample)
        neighbour_ind = neighbour_ind[0]
        random_machine.shuffle(neighbour_ind)
        return neighbour_ind, center_point

    def _gen_center_list(self):
        centers = o3d.voxel_sampling(self.reader.points, voxel_size=7)
        self.random_machine.shuffle(centers)
        if len(centers) >= len(self):
            return centers[:len(self)]
        else:
            res_len = len(self)-len(centers)
            random_centers_index = self.random_machine.randint(0, len(self.reader), res_len)
            return np.concatenate([centers,  self.reader.points[random_centers_index]])
    '''
    
class RandomSampler(BasicSampler):
    def __init__(self, reader, params, is_training=True, seed=0, return_index=False):
        self.num_points_per_sample = 0
        self.modify_type = None
        super(RandomSampler, self).__init__(*[reader, params, is_training])
        self.center = np.array([(self.reader.max_bounds[0] - self.reader.min_bounds[0]) / 2,
                                (self.reader.max_bounds[1] - self.reader.min_bounds[1]) / 2,
                                self.reader.min_bounds[2]])
        self.random_machine = np.random.RandomState(seed)
        self.return_index=return_index

        self._infer_seq, self._res_num = self._gen_random_seq()

    def modify_points(self, points, *args, **kwargs):
        return self.modify_func(points, center=self.center)

    def cal_length(self):
        return int(len(self.reader) / self.num_points_per_sample) + 1

    def get_indices(self, ind, set_random_machine=None, *args, **kwargs):
        ind = self._sample_index(ind, set_random_machine)
        if True:
            points, colors, labels = self.reader[ind]
            points_centered = self.modify_points(points)
            return points_centered, points, labels, colors

    def _gen_random_seq(self):
        seq = np.random.permutation(len(self.reader))
        res = len(self) * self.num_points_per_sample - len(self.reader)
        seq = np.concatenate([seq, seq[:res]])
        return seq, res

    def _get_train_index(self, set_random_machine=None):
        random_machine = self.random_machine \
            if set_random_machine is not None else set_random_machine
        return random_machine.permutation(len(self.reader.points))[:self.num_points_per_sample]

    def _get_infer_index(self, ind):
        seq_ind = np.arange(ind * self.num_points_per_sample,
                            (ind + 1) * self.num_points_per_sample)
        return self._infer_seq[seq_ind]

    def _sample_index(self, ind, set_random_machine=None):
        if self.is_training:
            ind = self._get_train_index(set_random_machine)
        else:
            ind = self._get_infer_index(ind)
        return ind