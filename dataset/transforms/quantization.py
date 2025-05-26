import numpy as np
from itertools import repeat
from .groupby import aggregate

def np_quantize(coords,
                features,
                labels=None,
                ignore_label=-100,
                quantize_size=[1.,1.,1],
                *,
                feature_method='first',
                label_method='distinct_row',
                num_classes=None,
                ):
    assert feature_method in ['mean', 'first', ], \
        "\'{}\' is not a feature quantization method.".format(feature_method)
    assert label_method in ['distinct_row', 'distinct_element', 'first', 'mean', 'mode'], \
        "\'{}\' is not a label quantization method.".format(label_method)
    
    pts = np.copy(coords)
    pts = np.floor(pts / quantize_size).astype(np.int32)
    
    b_pts = np.ascontiguousarray(pts).view(
        np.dtype((np.void, pts.dtype.itemsize * pts.shape[1])))

    _, first_show_ind, group_inds, counts = np.unique(b_pts, return_inverse=True, return_index=True, return_counts=True )
    discrete_pts = pts[first_show_ind]

    if feature_method != 'first':
        un_feats = aggregate(group_inds, features, func=feature_method, axis=0)
    else:
        un_feats = features[first_show_ind]

    if not (labels is None):
        if label_method.startswith('distinct'):
            un_labels = __distinct_label(group_inds, labels, first_show_ind, ignore_label, use_row=label_method.endswith('row'))
        elif label_method == 'mean':
            assert not (num_classes is None ), "num_classes is needed."
            one_hot_labels = __one_hot(labels, num_classes).astype(float)
            un_labels = aggregate(group_inds, one_hot_labels, func="mean", axis=0)
        elif label_method == 'mode':
            num_classes = np.max(labels) + 1
            one_hot_labels = __one_hot(labels, num_classes).astype(float)
            un_labels = aggregate(group_inds, one_hot_labels, func="sum", axis=0)
            un_labels = np.argmax(un_labels, axis=-1)
        else:
            un_labels = labels[first_show_ind]

    return discrete_pts, un_feats, un_labels, group_inds


def __distinct_label(group_inds, labels, first_show_ind, ignore_label, use_row=True):
    un_label_std = aggregate(group_inds, labels, func='std', axis=0)
    un_labels = labels[first_show_ind]
    if use_row and labels.ndim > 1:
        ignore_row_labels = np.full((labels.shape[1],), ignore_label).astype(labels.dtype)
        nonzero_row_inds = un_label_std.sum(1).nonzero()
        un_labels[nonzero_row_inds] = ignore_row_labels
    else:
        un_labels[un_label_std.nonzero()] = np.array([ignore_label]).astype(labels.dtype)
    return un_labels

def __one_hot(ori_label, num_classes):
    EYE =  np.eye(num_classes)
    one_hot_label = EYE[ori_label]
    return one_hot_label