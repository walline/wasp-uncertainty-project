import torch
import pandas as pd
import numpy as np
from sepsis_helpers import *
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sngp import SNGP, SNGPClassifier
from sngp_helpers import InternalSplit, evaluate, save_data

C_TREATMENTS = ['max_dose_vaso']  # ['input_4hourly', 'max_dose_vaso']

USE_SPECTRAL_NORM = True
USE_GP_LAYER = True

PERFORM_OOD_EVALUATION = False

TRAIN_SIZE = 0.8  # Ignored if PERFORM_OOD_EVALUATION = True
RANDOM_STATE = 0
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
N_EPOCHS = 5
BATCH_SIZE = 64
ENCODER_HIDDEN_SIZE = 32
ENCODER_N_LAYERS = 2
GP_HIDDEN_SIZE = 64

def main():
    sepsis_data = pd.read_csv('sepsis_data.csv')

    groups = sepsis_data[C_GROUP]

    Y = sepsis_data[C_TREATMENTS]
    Y_discrete = Y.apply(discretize_treatments, raw=True)
    _, y = np.unique(Y_discrete, axis=0, return_inverse=True)
    n_classes = len(set(y))

    Yg = pd.concat([Y, groups], axis=1)
    previous_doses = Yg.groupby(by=C_GROUP).apply(func=lambda df: df.shift(periods=1).fillna(0))
    previous_doses = previous_doses.drop(C_GROUP, axis=1)
    rename = [(c, c + '_prev') for c in previous_doses.columns]
    previous_doses = previous_doses.rename(dict(rename), axis=1)

    X = sepsis_data[C_IMPORTANT_FEATURES]
    X = pd.concat([X, previous_doses], axis=1)
    Xg = pd.concat([X, groups], axis=1)
    n_features = X.shape[1]

    if PERFORM_OOD_EVALUATION:
        y_df = pd.DataFrame(y, columns=['y'])
        yg_df = pd.concat([y_df, groups], axis=1)
        ii_test = []
        for _, data in yg_df.groupby(by=C_GROUP):
            targets = data.y.values
            if np.max(targets) == n_classes-1:
                ii_test += list(data.index)
        ii_train = list(set(y_df.index)-set(ii_test))
    else:
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
        module__hidden_map=RNNMapping,
        module__gp_input_size=ENCODER_HIDDEN_SIZE,
        module__gp_hidden_size=GP_HIDDEN_SIZE,
        module__gp_output_size=n_classes,
        module__use_spectral_norm=USE_SPECTRAL_NORM,
        module__use_gp_layer=USE_GP_LAYER,
        module__input_size=n_features,
        module__hidden_size=ENCODER_HIDDEN_SIZE,
        module__n_layers=ENCODER_N_LAYERS,
        criterion=nn.CrossEntropyLoss,
        classes=np.arange(n_classes),
        optimizer=torch.optim.Adam,
        optimizer__lr=LEARNING_RATE,
        optimizer__weight_decay=WEIGHT_DECAY,
        max_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        iterator_train__collate_fn=pad_pack_sequences,
        iterator_valid__collate_fn=pad_pack_sequences,
        dataset=SequentialDataset,
        train_split=InternalSplit(),
        device=device
    )

    sngp_clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', sngp)])
    sngp_clf.fit(Xg_train, y_train)
    save_data(sngp_clf, 'model.pickle')

if __name__ == '__main__':
    main()
