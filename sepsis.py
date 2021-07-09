import torch
import pickle
import pandas as pd
import numpy as np
from sepsis_helpers import *
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
    train_data, test_data = get_train_test_data(
        C_TREATMENTS,
        C_GROUP,
        C_IMPORTANT_FEATURES,
        PERFORM_OOD_EVALUATION,
        TRAIN_SIZE,
        RANDOM_STATE
    )
    Xg_train, y_train = train_data
    Xg_test, y_test = test_data

    n_features = Xg_train.shape[1]-1
    n_classes = len(set(y_train))

    preprocessor = ColumnTransformer(
        transformers=[
            ('scale', scale_tf, list(set(Xg_train).intersection(C_SCALE))),
            ('log_scale', log_scale_tf, list(set(Xg_train).intersection(C_LOG_SCALE)))
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

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', sngp)])
    pipeline.fit(Xg_train, y_train)
    
    # Save the classifier
    net = pipeline[-1]
    with open('net.pkl', 'wb') as f:
        pickle.dump(net, f)

    if not PERFORM_OOD_EVALUATION:
        y_test_probas = pipeline.predict_proba(Xg_test)
        print(evaluate(y_test, y_test_probas))

if __name__ == '__main__':
    main()
