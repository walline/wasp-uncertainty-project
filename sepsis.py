import torch
import pandas as pd
import numpy as np
from sepsis_helpers import *
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sngp import SNGP, SNGPClassifier
from sngp_helpers import InternalSplit

TRAIN_SIZE = 0.8
RANDOM_STATE = 0
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
N_EPOCHS = 10
BATCH_SIZE = 64

def main():
    sepsis_data = pd.read_csv('sepsis_data.csv')

    groups = sepsis_data[C_GROUP]

    Y = sepsis_data[C_TREATMENTS]
    Y_discrete = Y.apply(discretize_treatments, raw=True)
    # _, y = np.unique(Y_discrete, axis=0, return_inverse=True)
    y = Y_discrete['max_dose_vaso'].to_numpy()
    n_classes = len(set(y))

    Yg = pd.concat([Y, groups], axis=1)
    previous_doses = Yg.groupby(by=C_GROUP).apply(func=lambda df: df.shift(periods=1).fillna(0))
    previous_doses = previous_doses.drop(C_GROUP, axis=1)
    previous_doses = previous_doses.rename({'input_4hourly': 'input_4hourly_prev', 'max_dose_vaso': 'max_dose_vaso_prev'}, axis=1)

    X = sepsis_data[C_IMPORTANT_FEATURES]
    X = pd.concat([X, previous_doses], axis=1)
    Xg = pd.concat([X, groups], axis=1)
    n_features = X.shape[1]

    gss = GroupShuffleSplit(n_splits=1, train_size=TRAIN_SIZE, random_state=RANDOM_STATE)
    ii_train, ii_test = next(gss.split(X, y, groups))
    Xg_train, y_train = Xg.iloc[ii_train], y[ii_train]
    Xg_test, y_test = Xg.iloc[ii_test], y[ii_test]

    preprocessor = ColumnTransformer(
        transformers=[
            ('scale', scale_tf, list(set(X).intersection(C_SCALE))),
            ('log_scale', log_scale_tf, list(set(X).intersection(C_LOG_SCALE)))
        ],
        remainder='passthrough'
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sngp = SNGPClassifier(
        module=SNGP,
        module__hidden_map=RNNMapping,  # Sepsis
        module__gp_input_size=16,
        module__gp_hidden_size=32,
        module__gp_output_size=n_classes,
        module__use_spectral_norm=True,
        module__use_gp_layer=True,
        module__input_size=n_features,  # Sepsis
        module__hidden_size=16,  # Sepsis
        module__n_layers=1,  # Sepsis
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=LEARNING_RATE,
        optimizer__weight_decay=WEIGHT_DECAY,
        max_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        iterator_train__collate_fn=pad_pack_sequences,  # Sepsis
        iterator_valid__collate_fn=pad_pack_sequences,  # Sepsis
        dataset=SequentialDataset,  # Sepsis
        train_split=InternalSplit(),
        device=device
    )

    sngp_clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', sngp)])
    sngp_clf.fit(Xg_train, y_train)
    print(sngp_clf.score(Xg_test, y_test))

if __name__ == '__main__':
    main()
