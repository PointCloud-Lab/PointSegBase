import numpy as np
import os
class __BaseSceneSampler(object):
    def __init__(self, subsampler_lengths,  **kwarg) -> None:
        self.subsampler_lengths = np.asarray(subsampler_lengths)
        self.total_length = np.sum(np.asarray(subsampler_lengths))
    
    def __getitem__(self, index):
        pass
    
    def __len__(self):
        return self.total_length 
    

class RandomSceneSampler(__BaseSceneSampler):
    def __init__(self, subsampler_lengths, **kwarg) -> None:
        super().__init__(subsampler_lengths, **kwarg)
        self.equal_flag = kwarg.get( 'equal_scene', False)
        max_sample_num = kwarg.get( 'max_samples_per_epcoh', 500)
        if max_sample_num is None:
            self.total_length = np.sum(subsampler_lengths)
        else:
            self.total_length = max_sample_num
        self.bins = self.partition()
        
        seed = kwarg.get( 'seed', 0)
        seed_seq = np.random.RandomState(seed)
        self.random_states = [np.random.RandomState(s) for s in seed_seq.randint(1e6, size=len(self))]
           
    def __getitem__(self, index):
        scene_ind =  np.digitize(index, self.bins)
        cnts = self.subsampler_lengths[scene_ind]
        ind_in_scene = self.random_states[index].randint(0, cnts)
        return scene_ind, ind_in_scene
    
    def partition(self):
        if self.equal_flag:
            p = np.full(len(self.subsampler_lengths), 1/len(self.subsampler_lengths))
        else:
            p = self.subsampler_lengths / np.sum(self.subsampler_lengths)
        bins = np.add.accumulate(p) * self.total_length
        bins = np.round(bins).astype(int)
        bins[-1] = self.total_length
        return bins


class SequentialSceneSampler(__BaseSceneSampler):
    def __init__(self, subsampler_lengths, **kwarg) -> None:
        super().__init__(subsampler_lengths, **kwarg)
        self.bins = self.partition()
    
    def __getitem__(self, index):
        scene_ind =  np.digitize(index, self.bins)
        ind_in_scene = index - self.bins[scene_ind-1] if scene_ind > 0 else index
        return scene_ind, ind_in_scene
    
    def partition(self):
        bins = np.add.accumulate(self.subsampler_lengths)
        bins[-1] = self.total_length
        return bins 
    