import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../..")))
from dataset.data_utils import cal_knn
def interpolate(sparse_points, dense_points, nn_num=1, knn_module='SkNN'):#GPU_id=None):
    '''
    sparse_points: predicted points
    '''
    knn_module = getattr(cal_knn, knn_module)
    knn = knn_module()
    knn.train(sparse_points)
    return knn.search(dense_points, nn_num)

if __name__ == '__main__':
    import numpy as np
    import time
    d_pts = np.random.rand(200000, 3)
    s_pts = np.random.rand(100000, 3)
    tic = time.time()
    _, ind = interpolate(s_pts, d_pts, nn_num=1, GPU_id=None)
    print(time.time() - tic)
    print(ind.shape)
    print(s_pts[ind].shape)