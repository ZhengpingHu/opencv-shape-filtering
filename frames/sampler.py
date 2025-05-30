# sampler.py
import numpy as np
from torch.utils.data import Sampler

class DiversitySampler(Sampler):
    def __init__(self, data_source, batch_size, num_segments=10):
        self.N = len(data_source)
        self.batch_size = batch_size
        self.num_segments = min(num_segments, batch_size)
        self.segment_size = int(np.ceil(self.N / self.num_segments))

    def __iter__(self):
        all_indices = []
        per_seg = self.batch_size // self.num_segments
        for seg in range(self.num_segments):
            start = seg * self.segment_size
            end = min((seg+1) * self.segment_size, self.N)
            if start >= end: break
            choices = np.random.choice(np.arange(start, end), per_seg, replace=False)
            all_indices.extend(choices.tolist())
        while len(all_indices) < self.batch_size:
            all_indices.append(np.random.randint(0, self.N))
        np.random.shuffle(all_indices)
        return iter(all_indices)

    def __len__(self):
        return self.N
