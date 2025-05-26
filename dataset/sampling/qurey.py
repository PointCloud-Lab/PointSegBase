import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np


class KnnQuery(object):
    def __init__(self,  search_tree_3d):
        self.knn_model = search_tree_3d

    def search(self, point, k):
        if len(point.shape) < 2:
            point = np.expand_dims(point, axis=0)
        return self.knn_model.search(point, k)

class BlockQuery(object):
    def __init__(self, search_tree_2d, block_size):
        self.knn_model = search_tree_2d
        self.block_size = np.array(block_size)

    def search_candidates(self, center_pt, *args):
        center_pt = center_pt[:-1] if len(center_pt.shape) == 1 else center_pt[:, :-1]
        return self.knn_model.search_grid(center_pt, grid_size=self.block_size)

    def search(self, center, points):
        assert len(points.shape) == 2, 'Points must be in shape N x dim'
        points = points[:, :-1]
        candidate_index = self.search_candidates(center)
        candidate_points = points[candidate_index]
        return    candidate_index[filter_indices(candidate_points, center[0] - self.block_size[0] / 2, center[0] + self.block_size[0] / 2,
                                            center[1] - self.block_size[1] / 2, center[1] + self.block_size[1] / 2)]
            
        
from numba import njit

@njit
def filter_indices(array, x_lower_bound, x_upper_bound, y_lower_bound, y_upper_bound):
    rows, cols = array.shape
    valid_indices = []

    for i in range(rows):
        x, y = array[i][0], array[i][1]

        if x_lower_bound <= x <= x_upper_bound and y_lower_bound <= y <= y_upper_bound:
            valid_indices.append(i)

    return np.array(valid_indices)
