#from .o3d import kdtree as o3d_kdtree
from sklearn.neighbors import KDTree as sk_kdtree
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pickle
import os
DTYPE = np.float16
'''
FAISS_INSTALLED = False
try:
    faiss = import_module('faiss')
    FAISS_INSTALLED = True
except Exception as e:
    print(e)
    print('Cannot import faiss for GPU nearest neighbout search, use Open3d instead.')
    
SKLEARN_INSTALLED = False
try:
    _neighbours = import_module("sklearn.neighbors")
    sk_kdtree = getattr(_neighbours, 'KDTree')
    SKLEARN_INSTALLED = True
except Exception as e:
    print(e)
    print('Cannot import sklearn for nearest neighbout search, use Open3d instead.')

'''

class _NearestNeighbors(object):
    def __init__(self, set_k=None, **kwargs):
        self.model = None
        self.set_k = set_k

    def train(self, data):
        pass
    
    @staticmethod
    def save(filename):
        return
    
    @staticmethod
    def load(filename):
        return 

    def search(self, data, k, return_distance=True):
        if self.set_k is not None:
            assert self.set_k == k, \
                'K not match to setting {}'.format(self.set_k)
        D, I = None, None
        return D, I


class Open3dNN(_NearestNeighbors):
    def __init__(self, set_k=None, **kwargs):
        super(Open3dNN, self).__init__(set_k, **kwargs)
        self.model = None
    def train(self, data):
        assert data.shape[1] == 3, 'Must be shape [?, 3] for point data'
        self.model = o3d_kdtree(data)

    def search(self, data, k, return_distance=False):
        assert self.model is not None, "Model have not been trained"
        if data.shape[0] == 1:
            [__, I, _] = self.model.search_knn_vector_3d(data[0], k)
        else:
            I = np.zeros((data.shape[0], k), dtype=np.int)
            with ThreadPoolExecutor(256) as executor:
                for i in range(I.shape[0]):
                    executor.submit(self._search_multiple, (self.model, I, data, k, i,))
        return None, I
    def search_radius(self, data, radius, return_distance=False):
        assert self.model is not None, "Model have not been trained"
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=0)
        if data.shape[0]== 1:
            return self.model.query_radius(data, r=radius)[0]
        else:
            return self.model.query_radius(data, r=radius)
        
    def save(self, filename):
        assert self.model is not None, "Model have not been trained"
        pickle.dump(self.model, open(filename, 'wb'))
    
    @staticmethod
    def load(filename):
        assert os.path.isfile(filename), '{} does not exist'.format(filename)
        init_nn = SkNN()
        init_nn.model = pickle.load(open(filename, 'rb')) 
        return init_nn

    @staticmethod
    def _search_multiple(knn_searcher, I, data, k, i):
            [__, I_, _] = knn_searcher.search_knn_vector_3d(data[i, :], k)
            I[i, :] = np.asarray(I_)


class SkNN(_NearestNeighbors):
    def __init__(self, set_k=None, **kwargs):
        super(SkNN, self).__init__(set_k, **kwargs)
        self.model = None
        if 'leaf_size' in kwargs.keys():
            self.leaf_size = kwargs['leaf_size']
        else:
            self.leaf_size = 20
            
    def save(self, filename):
        assert self.model is not None, "Model have not been trained"
        pickle.dump(self.model, open(filename, 'wb'))
    
    @staticmethod
    def load(filename):
        assert os.path.isfile(filename), '{} does not exist'.format(filename)
        init_nn = SkNN()
        with open(filename, 'rb') as f:
            init_nn.model = pickle.load(f) 
        return init_nn

    def train(self, data):
        self.model = sk_kdtree(data.astype(DTYPE), leaf_size=self.leaf_size)

    def search(self, data, k, return_distance=False):
        assert self.model is not None, "Model have not been trained"
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=0)
        if data.shape[0]== 1:
            return self.model.query(data.astype(DTYPE), k)[1][0]
        else:
            return self.model.query(data.astype(DTYPE), k)[1]
            
    def search_radius(self, data, radius, return_distance=False):
        assert self.model is not None, "Model have not been trained"
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=0)
        if data.shape[0]== 1:
            return self.model.query_radius(data.astype(DTYPE), r=radius)[0]
        else:
            return self.model.query_radius(data.astype(DTYPE), r=radius)
        

import numpy as np
from numba import njit, typed
from numba.types import int64, UniTuple

@njit
def gather_indices(keys, numba_dict, sorted_indices):
    total_len = 0
    new_keys = typed.List()
    for key in keys:
        if key in numba_dict:
            total_len += numba_dict[key][1]- numba_dict[key][0]  + 1
            new_keys.append(key)
    gathered_indices = np.empty(total_len, dtype=np.int64)
    
    insert_pos = 0
    for key in new_keys:
        first_pos, last_pos = numba_dict.get(key)
        indices = sorted_indices[first_pos:last_pos+1]
        len_indices = last_pos+1 - first_pos
        gathered_indices[insert_pos:insert_pos+len_indices] = indices
        insert_pos += len_indices
    return gathered_indices

class GridNN(object):
    def __init__(self, grid_size, **kwargs):
        size_x, size_y=grid_size[0]/2, grid_size[1]/2
        self.size_x = size_x
        self.size_y = size_y
        self.min_vals = None
        self.bins_x = None
        self.bins_y = None
        self.sorted_indices = None 
        self.numba_dict = None 
        
    def save(self, filename):
        assert self.numba_dict is not None, "Model have not been trained"
        unique_tuples,first_pos, last_pos = self.__resolve_dict()
        np.savez(filename, unique_tuples=unique_tuples, 
                 first_pos= first_pos,
                 last_pos = last_pos,
                 sorted_indices=self.sorted_indices,
                 size_x=self.size_x,
                 size_y=self.size_y,
                 min_vals=self.min_vals,
                 bins_x=self.bins_x,
                 bins_y=self.bins_y)
    
    @staticmethod
    def load(filename):
        assert os.path.isfile(filename), '{} does not exist'.format(filename)
        init_nn = GridNN((0.0, 0.0))
        dic = np.load(filename)
        init_nn.__build_dict(dic['unique_tuples'], dic['first_pos'], dic['last_pos'])
        init_nn.sorted_indices = dic['sorted_indices']
        init_nn.size_x = dic['size_x']
        init_nn.size_y = dic['size_y']
        init_nn.min_vals = dic['min_vals']
        init_nn.bins_x = dic['bins_x']
        init_nn.bins_y = dic['bins_y']
        return init_nn
    
    def __resolve_dict(self):
        unique_tuples,first_pos, last_pos = [], [], []
        for key, value in self.numba_dict.items():
            unique_tuples.append(key)
            first_pos.append(value[0])
            last_pos.append(value[1])
        return np.asarray(unique_tuples), np.asarray(first_pos), np.asarray(last_pos)
    
    def train(self, data):
        unique_tuples, first_pos, last_pos = self.quantize_and_sort(data)
        self.__build_dict(unique_tuples, first_pos, last_pos)
        
    def quantize(self,data):
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=0)
        x = data[:, 0] - self.min_vals[0]
        y = data[:, 1] - self.min_vals[1]
        quantized_x = np.digitize(x, self.bins_x) + 1
        quantized_y = np.digitize(y, self.bins_y) + 1
        quantized_data = np.column_stack((quantized_x, quantized_y))
        return quantized_data
        
    def quantize_all(self, data, size_x, size_y):
        self.min_vals = np.min(data, axis=0)
        range_x = np.max(data[:, 0]) - self.min_vals[0]
        range_y = np.max(data[:, 1]) - self.min_vals[1]
        bin_num_x = np.ceil(range_x/size_x)+1
        bin_num_y = np.ceil(range_y/size_y)+1
        self.bins_x = np.linspace(0, bin_num_x * size_x, bin_num_x.astype(int) + 1).astype(np.float32)
        self.bins_y = np.linspace(0, bin_num_y * size_y, bin_num_y.astype(int) +1).astype(np.float32)
        
        return self.quantize(data)
        
    def quantize_and_sort(self, data):
        quantized_data = self.quantize_all(data, self.size_x, self.size_y)
        self.sorted_indices = np.lexsort((quantized_data[:, 1], quantized_data[:, 0])).astype(np.uint32)
        sorted_data = quantized_data[self.sorted_indices]
        unique_tuples, first_pos, counts = np.unique(sorted_data, axis=0, return_index=True, return_counts=True)
        last_pos = first_pos + counts - 1
        return unique_tuples, first_pos, last_pos

    def __build_dict(self, unique_tuples, first_pos, last_pos):
        self.numba_dict = typed.Dict.empty(key_type=UniTuple(int64, 2), value_type=UniTuple(int64, 2))
        for key, first_pos, last_pos in zip(unique_tuples.tolist(), first_pos, last_pos):
            self.numba_dict[tuple(key)] = (first_pos, last_pos)

    def search_grid(self, data, grid_size=None, **kwargs):
        k = self.quantize(data)[0]
        ks = typed.List()
        if grid_size is None:
            extend_x, extend_y = 1, 1
        else:
            extend_x = int(np.floor(grid_size[0] / self.size_x)) - 1
            extend_y = int(np.floor(grid_size[1] / self.size_y)) - 1
        for i in range(-extend_x, extend_x+1):
            for j in range(-extend_y, extend_y+1):
                ks.append((k[0]+i, k[1]+j))
        return gather_indices(ks, self.numba_dict, self.sorted_indices)
