import numpy as np
import traceback
import torch 
import numpy as np

class MetricRecorder(object):
    def __init__(self, h_matrices, record_all=True, ignore_label=None) -> None:
        self.h_matrices = h_matrices
        self.gather_id = [np.argmax(m, axis=0) for m in h_matrices]
        self.record_all = record_all
        self.record_pred_probs = [[] for _ in range(len(h_matrices))]
        self.record_targets = []
        self.cur_batch_size = 0
        self.ignore_label = ignore_label
    
    def renew(self):
        self.record_pred_probs = [[] for _ in range(len(self.h_matrices))]
        self.record_targets = []

    def inverse_label(self, pred_probs, inverse_list, len_list):
        split_indices = [sum(len_list[:i+1]) for i in range(len(len_list)-1)] 
        subarrays = np.split(pred_probs, split_indices)
        new_pred_probs = []
        for inv, subarray in zip(inverse_list, subarrays):
            new_pred_probs.append(subarray[inv])
        return np.concatenate(new_pred_probs, axis=0)

    def update(self, batch_pred, batch_data, *args, **kwargs):
        sparse_flag = batch_pred['logits'] is None
        for i, hm in enumerate(self.h_matrices):
            pred_logits = batch_pred['sparse_logits'][i].F if sparse_flag else batch_pred['logits'][i]
            with torch.no_grad():
                pred_probs = torch.softmax(pred_logits, dim=-1)
            pred_probs = pred_probs.cpu().numpy()
            if sparse_flag:
                pred_probs = self.inverse_label(pred_probs, 
                    [inv.cpu().numpy()for inv in batch_data['inverse']],
                    [lb.shape[0] for lb in batch_data['sparse_feats']]
                    )
            self.record_pred_probs[i].append(pred_probs)
        if sparse_flag:
            targets = np.concatenate([t.cpu().numpy() for t in batch_data['labels']])
        else:
            targets = batch_data['labels'].cpu().numpy()
        self.record_targets.append(targets)
        self.cur_batch_size = len(batch_data['labels'])
        
    def get_dict(self):
        target = np.concatenate(self.record_targets)
        save_dict = {}
        for i, gid in enumerate( self.gather_id): 
            target_ = gid[target]
            pred = np.concatenate(self.record_pred_probs[i])
            save_dict.update({'pred_%d'%i:pred, 'target_%d'%i:target_})
        return save_dict
        
        
    @property
    def consistent_score(self):
        c_score = None
        for i, gid in enumerate( self.gather_id): 
            target_ = gid[target]
            pred = np.concatenate(self.record_pred_probs[i])
            pred_label = np.argmax(pred, axis=-1)
            mious.append(self.cal_mean_iou(pred_label, target_, pred.shape[-1]))
        return self.get_result()[0]
    
    @property
    def cur_consistent_score(self):
        return self.get_result()[0]
    
    @property
    def miou(self):
        target = np.concatenate(self.record_targets)
        mious= []
        for i, gid in enumerate( self.gather_id): 
            target_ = gid[target]
            pred = np.concatenate(self.record_pred_probs[i])
            pred_label = np.argmax(pred, axis=-1)
            mious.append(self.cal_mean_iou(pred_label, target_, pred.shape[-1]))
        return mious

    @property
    def cur_miou(self):
        target = self.record_targets[-1]
        mious= []
        for i, gid in enumerate( self.gather_id): 
            target_ = gid[target]
            pred = self.record_pred_probs[i][-1]
            pred_label = np.argmax(pred, axis=-1)
            mious.append(self.cal_mean_iou(pred_label, target_, pred.shape[-1]))
        return mious
    
    @property
    def mean_acc(self):
        target = np.concatenate(self.record_targets)
        acc_s = []
        for i, gid in enumerate( self.gather_id): 
            target_ = gid[target]
            pred = np.concatenate(self.record_pred_probs[i])
            pred_label = np.argmax(pred, axis=-1)
            acc_s.append(self.cal_mean_acc(pred_label, target_, pred.shape[-1]))
        return acc_s
    
    @property
    def cur_mean_acc(self):
        target = self.record_targets[-1]
        acc_s = []
        for i, gid in enumerate( self.gather_id): 
            target_ = gid[target]
            pred = self.record_pred_probs[i][-1]
            pred_label = np.argmax(pred, axis=-1)
            acc_s.append(self.cal_mean_acc(pred_label, target_, pred.shape[-1]))
        return acc_s
    
    @property
    def overall_acc(self):
        target = np.concatenate(self.record_targets)
        acc_s = []
        for i, gid in enumerate( self.gather_id): 
            target_ = gid[target]
            pred = np.concatenate(self.record_pred_probs[i])
            pred_label = np.argmax(pred, axis=-1)
            acc_s.append(self.cal_overall_acc(pred_label, target_))
        return acc_s
    
    @property
    def cur_overall_acc(self):
        target =self.record_targets[-1]
        acc_s = []
        for i, gid in enumerate( self.gather_id): 
            target_ = gid[target]
            pred = self.record_pred_probs[i][-1]
            pred_label = np.argmax(pred, axis=-1)
            acc_s.append(self.cal_overall_acc(pred_label, target_))
        return acc_s
    
    def cal_mean_iou(self, pred_labels, target_labels, num_class):
        classes = np.arange(num_class)
        pred_labels = pred_labels.reshape(-1)
        target_labels = target_labels.reshape(-1)
        if not( self.ignore_label is None):
            mask = np.logical_and(pred_labels != self.ignore_label, target_labels != self.ignore_label)
            pred_labels = pred_labels[mask]
            target_labels = target_labels[mask]
        pred_per_class = [pred_labels == c for c in classes]
        target_per_class = [target_labels == c for c in classes]
        IoU_per_class = []
        for pred, target in zip(pred_per_class, target_per_class):
            intersection = np.logical_and(pred, target).sum()
            union = np.logical_or(pred, target).sum()
            if union == 0:
                IoU_per_class.append(1)
            else:
                IoU_per_class.append(intersection / union)
        return np.mean(IoU_per_class)
    
    def cal_mean_acc(self,  pred_label, target_label, num_class):
        pred_label = pred_label.astype(np.int32)
        target_label = target_label.astype(np.int32)
        if not( self.ignore_label is None):
            mask = np.logical_and(pred_label != self.ignore_label, target_label != self.ignore_label)
            pred_label = pred_label[mask]
            target_label = target_label[mask]
        accs = []
        for i in range(num_class):
            pred_i = pred_label == i
            target_i = target_label == i
            accs.append((pred_i & target_i).sum() / target_i.sum())
        return np.mean(accs)
    
    def cal_overall_acc(self,  pred_label, target_label):
        pred_labels = pred_label.astype(np.int32)
        target_labels = target_label.astype(np.int32)
        if not( self.ignore_label is None):
            mask = np.logical_and(pred_labels != self.ignore_label, target_labels != self.ignore_label)
            pred_labels = pred_labels[mask]
            target_labels = target_labels[mask]
        assert pred_labels.shape == target_labels.shape, "pred_labels.shape != target_labels.shape"
        acc = np.sum(pred_labels == target_labels) / np.prod(np.array(target_labels.shape))
        return acc
    
class ConfusionMatrix(object):
    def __init__(self, label_range):
        self.confusion_matrix = np.zeros((len(label_range), len(label_range)),
                                         dtype=np.int)
        self.label_range = label_range

    def update(self, pred, target):
        update_m = self.cal_iou_matrix(pred, target, label_range=self.label_range)
        self.confusion_matrix += update_m

    @staticmethod
    def cal_iou_matrix(pred, target, label_range=[]):
        '''
        :param pred:
        :param target:
        :param label_range:
        :return: matrix: label size x label size
                            target
                            l1 l2 l3 l4 l5
                        l1
                result  l2
                        l3
                        l4
                        l5
        '''
        IouMetric._check_shape(pred, target)
        if len(pred.shape) >= 2:
            pred = pred.flatten()
            target = target.flatten()
        label_size = len(label_range)
        point_size = pred.shape[0]
        matrix = np.zeros((label_size, label_size), dtype=np.int64)
        label_repeat = np.tile(np.array(label_range), (point_size, 1)).astype(np.int64).transpose()
        pred_i = np.tile(pred , (label_size, 1)) == label_repeat
        target_i = np.tile(target, (label_size, 1)) == label_repeat
        for i, l in enumerate(label_range):
            new_target_i = np.tile(target_i[i, :], (label_size, 1))
            matrix[:, i] = np.sum(pred_i * new_target_i, axis=1)
        return matrix


class HierarchicalConsistency(object):
    @staticmethod
    def cal_consistency_proportion_per_point(h_matrix, labels):
        num_pts = labels.shape[0]
        leaf_length = h_matrix.classes_num[-1]
        all_score = np.zeros((num_pts, leaf_length), dtype=np.int)
        for i in range(leaf_length):
            one_path = h_matrix.all_valid_h_label[i]
            all_score[:, i] = np.sum(labels == one_path, axis=1)
        return np.max(all_score, axis=1) / float(h_matrix.layer_num)

    @staticmethod
    def cal_consistency_rate(h_matrix, labels, cp_thresh=1.0):
        cp = HierarchicalConsistency.cal_consistency_proportion_per_point(h_matrix, labels)
        return np.sum(cp >= cp_thresh) / float(labels.shape[0])

class AccuracyMetric:
    def __init__(self, label_range):
        self.confusion_matrix = ConfusionMatrix(label_range)
        self.label_range = label_range

    def __repr__(self):
        return 'Confusion Matrix \n :{}\n'.format(str(self.confusion_matrix))

    def overall_accuracy(self):
        return self.matrix2OA(self.confusion_matrix)

    def avg_accuracy(self):
        return self.matrix2AA(self.confusion_matrix)

    def update(self, pred, target):
        self.confusion_matrix.update(pred, target)

    @staticmethod
    def cal_oa(pred, target):
        IouMetric._check_shape(pred, target)
        return np.sum(pred==target) / float(pred.shape[0])

    @staticmethod
    def matrix2OA(matrix):
        total_num = np.sum(matrix)
        return np.trace(matrix) / float(total_num)

    @staticmethod
    def matrix2AA(matrix):
        total_num = np.sum(matrix, axis=0)
        per_class_acc = np.diagonal / total_num
        return np.mean(per_class_acc), per_class_acc

    @staticmethod
    def _check_shape(pred, target):
        try:
            assert pred.shape == target.shape
        except AssertionError:
            raise ValueError('Shapes of {} and {} are not matched'.format(pred.shape, target.shape))
        except Exception as e:
            traceback.print_exc()


class IouMetric:
    def __init__(self, label_range):
        self.confusion_matrix = ConfusionMatrix(label_range)
        self.label_range = label_range

    def __repr__(self):
        return 'Confusion Matrix \n :{}\n'.format(str(self.confusion_matrix))

    def iou(self):
        return self.matrix2iou(self.confusion_matrix.confusion_matrix)

    def avg_iou(self):
        return self.matrix2avg_iou(self.confusion_matrix.confusion_matrix)

    def update(self, pred, target):
        self.confusion_matrix.update(pred, target)

    @staticmethod
    def _check_shape(pred, target):
        try:
            assert pred.shape == target.shape
        except AssertionError:
            raise ValueError('Shapes of {} and {} are not matched'.format(pred.shape, target.shape))
        except Exception as e:
            traceback.print_exc()

    @staticmethod
    def cal_iou(pred, target, label_range=[]):
        IouMetric._check_shape(pred, target)
        iou = []
        for l in label_range:
            pi = (pred==l)
            ti = (target==l)
            i = np.sum(pi*ti)
            u = (np.sum((pi + ti) != 0))
            iou_l = float(i) / float(u) if u != 0 else -1.0
            iou.append(iou_l)
        return np.asarray(iou)

    @staticmethod
    def cal_avg_iou(pred, target, label_range=[]):
        IouMetric._check_shape(pred, target)
        iou = IouMetric.cal_iou(pred, target, label_range)
        return IouMetric._average_iou(iou)


    @staticmethod
    def matrix2iou(matrix):
        size = matrix.shape[0]
        iou = []
        for j in range(size):
             i = matrix[j, j]
             u = np.sum(matrix[j, :]) + np.sum(matrix[:, j]) - i
             iou_one = float(i) / float(u) if u != 0 else -1.0
             iou.append(iou_one)
        return np.asarray(iou)

    @staticmethod
    def matrix2avg_iou(matrix):
        iou = IouMetric.matrix2iou(matrix)
        return IouMetric._average_iou(iou)

    @staticmethod
    def _average_iou(iou):
        mask = iou != -1
        if np.sum(mask) == 0:
            return np.sum(mask).astype(np.float)
        return np.sum(iou[mask]) / np.sum(mask).astype(np.float)

def iou_compare(best_mean_iou, best_class_iou, mean_iou, class_iou):
    if len(best_mean_iou) == 0:
        return True
    if len(mean_iou) <= 1:
        return mean_iou[0] > best_mean_iou
    ratio = 0.67
    bmi = np.asarray(best_mean_iou)
    mi = np.asarray(mean_iou)
    mi_ratio = np.sum(bmi <= mi).astype(float) / float(len(bmi))
    mean_iou_flag = mi_ratio > ratio
    all_iou_flag = []
    for i in range(len(class_iou)):
        bai = np.asarray(best_class_iou[i])
        ai = np.asarray(class_iou[i])
        ai_ratio = np.sum(bai <= ai).astype(float) / float(len(bai))
        iou_flag = ai_ratio > ratio
        all_iou_flag.append(int(iou_flag))
    all_iou_flag = float(sum(all_iou_flag)) / float(len(all_iou_flag))
    if all_iou_flag or mean_iou_flag:
        return True
    else:
        return False
