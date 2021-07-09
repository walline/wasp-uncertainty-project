import torch
import pickle
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from netcal.metrics import ECE

class InternalSplit:
    def __init__(self, train_size=0.8, random_state=None):
        self.train_size = train_size
        # We should be able to reproduce the internal split in __call__
        self.random_state = random_state if random_state is not None else np.random.randint(0, 101)
    
    def __call__(self, dataset, _):
        splitter = ShuffleSplit(n_splits=1, train_size=self.train_size, random_state=self.random_state)
        args = (np.arange(dataset.__len__()),)
        ii_train, ii_valid = next(splitter.split(*args))
        dataset_train = torch.utils.data.Subset(dataset, ii_train)
        dataset_valid = torch.utils.data.Subset(dataset, ii_valid)
        return dataset_train, dataset_valid

def sce(y_probas, y_one_hot, n_bins=10):
    ece = ECE(n_bins)
    sce = 0
    for yp, y in zip(y_probas.T, y_one_hot.T):
        sce += ece.measure(yp, y)
    return sce / y_probas.shape[1]

def evaluate(y, y_probas):
    y_preds = y_probas.argmax(axis=1)

    y = list(map(int, y))
    n_classes = y_probas.shape[1]
    y_one_hot = np.eye(n_classes)[y]

    ece = ECE(bins=10)

    metrics = {}
    metrics['accuracy'] = accuracy_score(y, y_preds)
    metrics['log_loss'] = log_loss(y, y_probas)
    metrics['ece'] = ece.measure(y_probas, y_one_hot)
    metrics['sce'] = sce(y_probas, y_one_hot)

    return metrics

def save_data(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)
