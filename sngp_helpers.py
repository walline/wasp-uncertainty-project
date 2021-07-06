import torch
import numpy as np
from sklearn.model_selection import ShuffleSplit

class InternalSplit:
    def __init__(self, train_size=0.8, random_state=None):
        self.train_size = train_size
        self.random_state = random_state
    
    def __call__(self, dataset, _):
        splitter = ShuffleSplit(n_splits=1, train_size=self.train_size, random_state=self.random_state)
        args = (np.arange(dataset.__len__()),)
        ii_train, ii_valid = next(splitter.split(*args))
        dataset_train = torch.utils.data.Subset(dataset, ii_train)
        dataset_valid = torch.utils.data.Subset(dataset, ii_valid)
        return dataset_train, dataset_valid
