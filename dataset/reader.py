import os
import numpy as np
import yaml
from .data_utils.o3d import read_point_cloud
from .data_utils import cal_knn
from os.path import basename, splitext
import h5py
LAZY_HDF5 = True

def read_file_list(list_file,
                   sets, 
                   use_color, 
                   color_channel=(), 
                   search_tree_3d_postfix='_3d.pickle',
                   search_tree_2d_postfix='_2d.npz',
                   return_file_list=False):
    with open(list_file, 'r') as yf:
        all_list = yaml.load(yf, Loader=yaml.FullLoader )

    main_path = all_list['MAIN_DATA_PATH']
    data_file_list = [d['DATA_PATH']['DATA'] for d in all_list[sets]]
    data_file_list = [os.path.join(main_path, f) for f in data_file_list]
    prefixes = [splitext(basename(p))[0] for p in data_file_list]

    main_path = all_list['MAIN_DATA_PATH']
    label_file_list = [d['DATA_PATH']['LABEL'] for d in all_list[sets]]
    label_file_list = [os.path.join(main_path, fl) for fl in label_file_list]
    
    try:
        main_path = all_list['MAIN_DATA_PATH']
        elabel_file_list = [d['DATA_PATH']['EXTRA_LABEL'] for d in all_list[sets]]
        elabel_file_list = [os.path.join(main_path, fl) for fl in elabel_file_list]
    except KeyError:
        elabel_file_list = [None] * len(label_file_list)
    
    search_tree_3d_file_list = [splitext(f)[0] + search_tree_3d_postfix for f in data_file_list]
    search_tree_2d_file_list = [splitext(f)[0] + search_tree_2d_postfix for f in data_file_list]

    reader_list = []
    for data_f, label_ps, ext_label_ps, prefix, tree_f_3d, tree_f_2d in zip(data_file_list, label_file_list, elabel_file_list,prefixes, search_tree_3d_file_list,search_tree_2d_file_list):
        reader_list.append(FileDataReader(data_f, label_ps,
                                          prefix, use_color,
                                          color_channel, 
                                          tree_f_3d,tree_f_2d,
                                          extra_label_path=ext_label_ps))
    
    if return_file_list:
        return reader_list, data_file_list
    else:
        return reader_list


def read_h_matrix_file_list(list_file):
    with open(list_file, 'r') as yf:
        file_list = yaml.load(yf, Loader=yaml.FullLoader)['file_list']
    return HierarchicalMatrixReader(files=file_list)


class FileDataReader(object):
    def __init__(
            self,
            data_path,
            label_path,
            prefix,
            use_color,
            color_channel=(),
            search_tree_3d_path=None,
            search_tree_2d_path=None,
            extra_label_path=None
    ):
        """
        Read file format data
        data should be save in such format:
            data_path + prefix + data_ext (ext should be in (.pcd, .ply, .npy))
        label should be saved in such format:
            label_path + prefix + postfix_1 + label_ext (ext should be in (.npy, .txt, .pts))
                                            :
            label_path + prefix + postfix_n + label_ext (ext should be in (.npy, .txt, .pts))

        class Type:
            region = str
            points = np.ndarray
            colors = np.ndarray
            labels = np.ndarray
            min_bounds = np.ndarray
            max_bounds = np.ndarray
            scene_z_size = np.float64

        __init__(
                data_path: str
                label_path: str
                prefix: str
                data_ext: str
                label_ext: str
                label_file_postfixes: [str, ]
                use_color: bool
                color_channel: tuple {default: ()}
                search_tree_file: str
                )
        """

        self.region = prefix
        self.points = None
        self.colors = None
        self.labels = None
        self.min_bounds = None
        self.max_bounds = None
        self.scene_z_size = None
        self.search_tree_3d_path = search_tree_3d_path
        self.search_tree_2d_path = search_tree_2d_path

        self.search_tree_3d = None
        self.search_tree_2d = None
        self.__valid_file_types = ['.pcd', '.ply', '.pts', '.npy', '.h5']
        self.__valid_label_types = ['.npy', '.txt', '.labels', '.h5']
        self.__color_channel = color_channel
        self.__use_color = use_color
        self.__h5_file = None 

        # Load points
        self.check_valid(data_path, label_path, extra_label_path)
        print('Reading data from prefix \'{}\''.format(prefix))
        self.read_point_cloud(data_path)
        print('Reading label from prefix \'{}\''.format(prefix))
        self.labels = self.read_label(label_path).astype(np.int64)
        self.check_label_valid(self.labels)
        
        if not (extra_label_path is None):
            self.extra_labels = self.read_label(extra_label_path)
            self.check_label_valid(self.extra_labels)
        else:
            self.extra_labels = None
       
        
        print('Prefix \'{}\' point number: {}, label categories: {}'. \
                format(prefix, len(self.points), (not self.extra_labels is None)+1 ))
        self.__load_trees()

    def check_valid(self, data_path, label_path, extra_label_path=None):
        data_ext = splitext(basename(data_path))[-1]
        if data_ext not in self.__valid_file_types:
            IOError('{} is not valid data file format'.format(data_path))

        label_ext = splitext(basename(label_path))[-1]
        if label_ext not in self.__valid_label_types:
            raise IOError('{} is not valid label file format'.format(label_path))
        
        if not (extra_label_path is None):
            extra_label_ext = splitext(basename(extra_label_path))[-1]
            if extra_label_ext not in self.__valid_label_types:
                raise IOError('{} is not valid label file format'.format(extra_label_path))

    def read_point_cloud(self, data_path):
        data_ext = splitext(basename(data_path))[-1]
        if data_ext in ['.pcd', '.ply', 'pts']:
            return self.__read_point_cloud_open3d(data_path)
        elif data_ext == ".npy":
            return self.__read_point_cloud_numpy(data_path, self.__color_channel)
        elif data_ext == '.h5':
            return self.__read_point_cloud_hdf5(data_path)

    def read_label(self, label_path, **load_args):
        label_ext = splitext(basename(label_path))[-1]
        if label_ext == '.npy':
            return  np.load(label_path)
        elif label_ext in ["labels", ".txt"]:
            return np.loadtxt(label_path, **load_args)
        elif label_ext == '.h5':
            return self.__h5_file['label_seg'][:]
                
    def check_label_valid(self, labels):
        assert len(labels) == self.points.shape[0], \
            "Labels dimension and points dimension are not matched."

    def __read_point_cloud_open3d(self, file_):
        points, colors, self.min_bounds, self.max_bounds = \
            read_point_cloud(file_, use_color=self.__use_color, return_bounds=True)
        self.points = points 
        self.colors = colors
        self.scene_z_size = self.max_bounds[2] - self.min_bounds[2]

        

    def __read_point_cloud_hdf5(self, file_,):
        self.__h5_file = h5py.File(file_, 'r')
        self.points = self.__h5_file['data'][:].astype(np.float32)
        if self.__use_color: self.colors = self.__h5_file['colors'][:]
        
        
    def __read_point_cloud_numpy(self, file_, color_channel):
        pts_array = np.load(file_)
        points = pts_array[:, :3]
        if self.__use_color:
            colors = pts_array[:, np.asarray(color_channel)]
        else:
            colors = np.zeros_like(self.points)
        self.points = points 
        self.colors = colors 
        self.min_bounds = np.amin(self.points, axis=0)
        self.max_bounds = np.amax(self.points, axis=0)
        self.scene_z_size = self.max_bounds[2] - self.min_bounds[2]


    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, item):
        if not (self.__h5_file is None):
            if isinstance(item, np.ndarray):
                if item.dtype == bool:
                    item = np.where(item)[0]
            sort_inds = np.argsort(item)
            item = item[sort_inds]
            sorted_sort_inds = np.argsort(sort_inds)
        pts, clrs, lbls = self.points[item, ...], self.colors[item, ...], self.labels[item, ...]
        if not (self.__h5_file is None):
            return pts[sorted_sort_inds], clrs[sorted_sort_inds], lbls[sorted_sort_inds]
        else:
            return pts, clrs, lbls
    
    def __load_trees(self):
        try:
            self.search_tree_3d = cal_knn.SkNN.load(self.search_tree_3d_path)
            print('Search tree 3d loaded.')
        except AssertionError:
            self.search_tree_3d = None 
            print('No search tree 3d at {}.'.format(self.search_tree_3d_path))
        except Exception as e:
            raise Exception(e)
        
        try:
            self.search_tree_2d = cal_knn.GridNN.load(self.search_tree_2d_path)
            print('Search tree 2d loaded.')
        except AssertionError:
            self.search_tree_2d = None 
            print('No search tree 2d at {}.'.format(self.search_tree_2d_path))
        except Exception as e:
            raise Exception(e)

    def build_3d_tree(self, *args, **kwargs):
        print('Building search tree 3d ...')
        search_tree = cal_knn.SkNN(*args, **kwargs)
        search_tree.train(self.points)
        search_tree.save(self.search_tree_3d_path)
        self.search_tree_3d = search_tree
        return self.search_tree_3d
        
    def build_2d_tree(self, *args, **kwargs):
        search_tree = cal_knn.GridNN(*args, **kwargs)
        search_tree.train(self.points)
        search_tree.save(self.search_tree_2d_path)
        self.search_tree_2d = search_tree
        return self.search_tree_2d

            
        
class HierarchicalMatrixReader(object):
    def __init__(self, matrices=None, files=None):
        self.hierarchical_matrices = []
        self.project_matrices = []
        if matrices is not None:
            self.hierarchical_matrices = np.asarray(matrices,
                                                    dtype=np.object)
        elif files is not None:
            self.load_files(files)
        else:
            raise TypeError('Missing required positional argument: \'matirces\' or \'files\'.')
       
        self.layer_num = len(self.hierarchical_matrices)
        self.classes_num = np.array([arr.shape[0] for arr in self.hierarchical_matrices])
        self.sort_matrices()
        self.all_valid_h_label = self._cal_valid_path()
        self.projet_labels = [m.argmax(axis=0) for m in self.hierarchical_matrices]

    def load_files(self, files):
        self.hierarchical_matrices = np.empty((len(files),), dtype=np.object)
        for i, f in enumerate(files):
            m = np.loadtxt(f, delimiter=',')
            self.hierarchical_matrices[i] = m

    def sort_matrices(self,):
        sort_ind = np.argsort(self.classes_num)
        self.hierarchical_matrices = self.hierarchical_matrices[sort_ind]
        self.classes_num = self.classes_num[sort_ind]
        self._all_label_project()

    def _project_matrix(self, matrix1, matrix2):
        return np.clip(np.dot(matrix1, matrix2.transpose()).transpose(), 0, 1)

    def _cal_valid_path(self):
        leaf_length = self.classes_num[-1]
        layer_num = self.layer_num
        all_valid_path = np.zeros((leaf_length, layer_num), dtype=np.int)
        for i in range(leaf_length):
            leaf_label = np.array([i], dtype=np.int)
            for j in range(layer_num):
                all_valid_path[i, j] = self.projet_label(leaf_label, -1, j)
        return all_valid_path
    
    def _all_label_project(self):
        s = len(self.hierarchical_matrices)
        self.project_matrices = np.empty((s, s, ), dtype=np.object)
        for i in range(self.project_matrices.shape[0]):
            for j in range(self.project_matrices.shape[1]):
                self.project_matrices[i, j] = \
                    self._project_matrix(self.hierarchical_matrices[i],
                                         self.hierarchical_matrices[j])

    def projet_label(self, labels, input_layer, output_layer, mode='num'):
        if mode == 'one_hot':
            return np.dot(labels, self[input_layer, output_layer])
        else:
            return np.argmax(self[output_layer, input_layer][labels], axis=1)

    def __getitem__(self, item):
        return self.project_matrices[item]

    