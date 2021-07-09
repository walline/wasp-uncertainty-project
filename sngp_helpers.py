import torch
import pickle
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, log_loss

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
    sce = 0
    for yp, y in zip(y_probas.T, y_one_hot.T):
        sce += ece(yp, y, n_bins)
    return sce / y_probas.shape[1]

def evaluate(y, y_probas):
    y_preds = y_probas.argmax(axis=1)

    y = list(map(int, y))
    n_classes = y_probas.shape[1]
    y_one_hot = np.eye(n_classes)[y]

    metrics = {}
    metrics['accuracy'] = accuracy_score(y, y_preds)
    metrics['log_loss'] = log_loss(y, y_probas)
    metrics['ece'] = ece(y_probas, y_one_hot)
    metrics['sce'] = sce(y_probas, y_one_hot)

    return metrics

def save_data(data, file):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def ece(y_probas, y, n_bins=10):  # Added this function for computing SCE
    '''
    Parameters
    ----------
    y_probas : NumPy array of shape (n_samples, n_classes) or (n_samples,)
        Predicted probabilities.
    y : NumPy array of shape (n_samples, n_classes) or (n_samples,)
        True labels. Either as label vector (1-D) or as one-hot encoded ground truth array (2-D).
    '''    
    if y_probas.ndim == 2:
        jj = y_probas.argmax(axis=-1)
        if y.ndim == 2:
            y = np.take_along_axis(y, np.expand_dims(jj, axis=-1), axis=-1).squeeze(axis=-1)
        elif not np.array_equal(y, y.astype(bool)):
            assert set(y) == set(jj), 'The labels must range from 0 to n_classes-1.' 
            y = np.array([yi == j for yi, j in zip(y, jj)], dtype=bool)
        y_probas = np.take_along_axis(y_probas, np.expand_dims(jj, axis=-1), axis=-1).squeeze(axis=-1)
    else:
        assert y.ndim == 1 and np.array_equal(y, y.astype(bool))
        
    bins = np.linspace(0, 1, n_bins+1)
    bin_indices = np.digitize(y_probas, bins, right=False)  # Assume no probabilities = 1

    ece = 0
    for i in range(1, n_bins+1):
        bin_mask = bin_indices == i
        if bin_mask.sum() == 0:
            continue
        y_probas_bin = y_probas[bin_mask]
        conf = np.mean(y_probas_bin)
        y_bin = y[bin_mask]     
        acc = np.mean(y_bin)
        ece += len(y_bin)*np.abs(acc-conf)

    return ece/len(y_probas)

def compute_accuracy_ece(predictions, labels, n_bins):
    # args:
    #
    # predictions: 'n_data x n_classes' array
    # labels: 'n_data' array of integers
    # n_bins: number of bins for ece evaluation
    
    n_data = labels.size

    y = np.argmax(predictions, axis=-1)
    p = np.amax(predictions, axis=-1)

    accuracy = (y == labels).sum(dtype=np.float32) / n_data

    ece = 0

    bin_edges = np.linspace(0, 1, num=n_bins+1, endpoint=True)

    for bin_ in range(n_bins):

        l_bound, h_bound = bin_edges[bin_], bin_edges[bin_+1]
        mask = (p > l_bound) & (p <= h_bound)

        if mask.sum() == 0:
            continue

        bin_confidence = np.mean(p[mask])
        bin_accuracy = (labels[mask] == y[mask]).sum(dtype=np.float32) / mask.sum()

        ece += np.abs(bin_confidence-bin_accuracy)*mask.sum()

    ece = ece/n_data

    return accuracy, ece
