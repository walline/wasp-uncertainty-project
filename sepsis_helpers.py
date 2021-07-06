import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy.stats import rankdata
from torch.utils.data import Dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from torch.nn.utils import spectral_norm
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

C_IMPORTANT_FEATURES = [
    'HR', 'SysBP', 'MeanBP', 'DiaBP', 'Shock_Index',  # circulation status
    'Hb',  # volume depletion
    'BUN', 'Creatinine', 'output_4hourly',  # kidney perfusion
    'Arterial_pH', 'Arterial_BE', 'HCO3', 'Arterial_lactate',  # global perfusion
    'PaO2_FiO2',  # fluid tolerance
    'age', 'elixhauser', 'SOFA'
]
C_TREATMENTS = ['input_4hourly', 'max_dose_vaso']
C_GROUP = 'icustayid'

C_SHIFT = ['gender', 'mechvent', 're_admission']
C_LOG_SCALE = [
    'SpO2', 'BUN', 'Creatinine', 'SGOT', 'SGPT', 'Total_bili', 'INR',
    'input_total', 'input_4hourly', 'output_total', 'output_4hourly',
    'max_dose_vaso', 'input_4hourly_prev', 'max_dose_vaso_prev'
]
C_SCALE = [
    'age', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 
    'Temp_C', 'FiO2_1', 'Potassium', 'Sodium', 'Chloride', 'Glucose', 
    'Magnesium', 'Calcium', 'Hb', 'WBC_count', 'Platelets_count', 'PTT', 
    'PT', 'Arterial_pH', 'paO2', 'paCO2', 'Arterial_BE', 'HCO3', 'Arterial_lactate', 
    'SOFA', 'SIRS', 'Shock_Index', 'PaO2_FiO2', 'cumulated_balance'
]
C_SCALE += ['elixhauser']

scale_tf = Pipeline(
        steps=[('scale', StandardScaler())]
    )
log_scale_tf = Pipeline(
    steps=[
        ('logaritmize', FunctionTransformer(func=lambda x: np.log(0.1+x))),
        ('scale', StandardScaler())
    ]
)

def discretize_treatments(treatments):
    discrete_treatments = np.zeros(treatments.size)  # 0 is default (zero dose)
    is_nonzero = treatments > 0
    ranked_nonzero_treatments = rankdata(treatments[is_nonzero]) / np.sum(is_nonzero)
    discrete_nonzero_treatments = np.digitize(ranked_nonzero_treatments, bins=[0., 0.25, 0.5, 0.75, 1.], right=True)
    discrete_treatments[is_nonzero] = discrete_nonzero_treatments
    return discrete_treatments

class SequentialDataset(Dataset):
    def __init__(self, X, y=None):
        '''
        Parameters
        ----------
        X : NumPy Array of shape (n_samples, n_features)
            Input data. The last column of X should represent the groups.
        y : NumPy Array of shape (n_samples,)
            Output targets.
        '''
        super(Dataset, self).__init__()
        sequences = []
        sequence_targets = [] if y is not None else y
        Xg = pd.DataFrame(X)
        c_groups = Xg.columns[-1]
        for _, sequence in Xg.groupby(by=c_groups):
            sequences += [sequence.drop(c_groups, axis=1)]  # Exclude the groups
            ii_sequence = list(sequence.index)
            if y is not None: sequence_targets += [y[ii_sequence]]
        self.sequences = sequences
        self.sequence_targets = sequence_targets

    def _transform(self, X, y):
        X = torch.Tensor(X.values)
        if y is None:
            y = torch.Tensor([0])
        else:
            assert isinstance(y,  np.ndarray)
            y = torch.Tensor(y).type(torch.LongTensor)
        return X, y

    def __getitem__(self, i):
        X, y = self.sequences, self.sequence_targets
        Xi = X[i]
        yi = y[i] if y is not None else y
        return self._transform(Xi, yi)

    def __len__(self):
        return len(self.sequences)

def pad_pack_sequences(batch):
    sequences, targets = zip(*batch)
    lengths = [sequence.shape[0] for sequence in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True)
    packed_padded_sequences = pack_padded_sequence(
        padded_sequences,
        batch_first=True,
        lengths=lengths,
        enforce_sorted=False
    )
    targets = torch.cat(targets, dim=0)
    return packed_padded_sequences, targets

def flatten(t):
    return [item for sublist in t for item in sublist]

class RNNMapping(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, use_residual_blocks=False, use_spectral_norm=False):
        super(RNNMapping, self).__init__()
        if not use_residual_blocks:
            self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=True)
            if use_spectral_norm:
                weights_biases = flatten(self.rnn._all_weights)
                weights = [w for w in weights_biases if w.startswith('weight')]
                for w in weights:
                    self.rnn = spectral_norm(self.rnn, w)
        else:
            raise NotImplementedError

    def forward(self, inputs):
        encodings, _ = self.rnn(inputs)
        encodings, lengths = pad_packed_sequence(encodings, batch_first=True)
        lengths = lengths.cpu().tolist()
        max_length = max(lengths)
        ii = np.concatenate([i*max_length+np.arange(l) for i, l in enumerate(lengths)])
        hidden_size = encodings.shape[-1]
        encodings = encodings.view(-1, hidden_size)
        return encodings[ii]
