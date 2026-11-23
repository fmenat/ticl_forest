import numpy as np
import math
from torch.utils.data import BatchSampler


class YearGroupedBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, sorted_years = True):
        self.batch_size = batch_size
        self.year_to_indices = dataset.year_to_indices
        self.validation_flag = dataset.validation
        self.sorted_years = sorted_years
        
    def __iter__(self):
        all_batches = []
        for year in sorted(self.year_to_indices.keys()):
            indices = self.year_to_indices[year]
            if not self.validation_flag:
                np.random.shuffle(indices)
            
            # Create batches for that year
            year_batches = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]
            all_batches.extend(year_batches)
            
        if not self.sorted_years and not self.validation_flag:
            np.random.shuffle(all_batches)
        
        for batch in all_batches:
            yield batch

    def __len__(self):
        return sum(math.ceil(len(indices) / self.batch_size) for indices in self.year_to_indices.values())



class BalancedYearGroupedBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, sorted_years = True):
        self.batch_size = batch_size
        self.year_to_indices = dataset.year_to_indices
        self.validation_flag = dataset.validation
        self.dist_region_samples = dataset.dist_region_samples
        self.sorted_years = sorted_years
        
    def __iter__(self):
        all_batches = []
        for year in sorted(self.year_to_indices.keys()):
            indices = self.year_to_indices[year]
            
            if self.validation_flag:
                year_indices = indices
            else:
                weights = np.array([self.dist_region_samples[i] for i in indices], dtype=np.float64) + 1e-3  #** self.power
                weights /= weights.sum()
                year_indices = np.random.choice(indices, size=len(indices),  replace=True,  p=weights)

            # create batches for that year
            year_batches = [year_indices[i : i + self.batch_size]  for i in range(0, len(year_indices), self.batch_size)]
            all_batches.extend(year_batches)
            
        if not self.sorted_years and not self.validation_flag:
            np.random.shuffle(all_batches)
        
        for batch in all_batches:
            yield batch

    def __len__(self):
        return sum(math.ceil(len(indices) / self.batch_size) for indices in self.year_to_indices.values())
