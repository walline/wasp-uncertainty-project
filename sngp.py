import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from skorch.dataset import uses_placeholder_y, unpack_data
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring
from sklearn.metrics import accuracy_score
from skorch.utils import to_numpy

class GP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, scale=2.0, s=0.001, m=0.999):
        super(GP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.scale = scale
        self.s = s
        self.m = m

        self.W = torch.normal(mean=0.0, std=1.0, size=(hidden_size, input_size))
        self.b = np.pi*torch.rand(hidden_size)
        
        beta = np.random.multivariate_normal(np.zeros(hidden_size), np.eye(hidden_size), output_size).astype(np.float32)
        self.beta = torch.tensor(beta, requires_grad=True)
        
        self.C = np.sqrt(hidden_size/scale)
        
        I = np.eye(hidden_size).astype(np.float32)
        self.precisions = torch.tensor(s*I).repeat(output_size, 1, 1)

    def propagate_gp_layer(self, inputs):
        return self.C*torch.cos(F.linear(inputs, -self.W, self.b))

    def forward(self, inputs):
        Phi = self.propagate_gp_layer(inputs)
        logits = F.linear(Phi, self.beta, bias=None)
        return logits, Phi

    def mc_averaging(self, logits, variances, n_samples):
        probas = []
        for l, v in zip(logits, variances):
            samples = torch.normal(l, torch.sqrt(v), size=(n_samples, 1))
            probas += [samples]
        probas = torch.cat(probas, dim=1)
        probas = F.softmax(probas, dim=1)
        probas = torch.mean(probas, dim=0)
        return probas

    def predict_proba(self, inputs, n_mc_samples=10):
        logits, Phi = self.forward(inputs)
        if hasattr(self, 'covariances'):  # Assuming model is fully trained
            batch_size = Phi.shape[0]
            Phi = Phi.unsqueeze(1)  # (batch_size, 1, hidden_size)
            PhiT = PhiT = Phi.permute(0, 2, 1)  # (batch_size, hidden_size, 1)
            variances = []
            for cov in self.covariances:
                cov_batch = cov.repeat(batch_size, 1, 1)
                temp = torch.bmm(Phi, cov_batch)
                var_k = torch.bmm(temp, PhiT)
                var_k = var_k.flatten()
                variances += [var_k]
            variances = torch.stack(variances, dim=1)
            probas = []
            for l, v in zip(logits, variances):
                probas += [self.mc_averaging(l, v, n_mc_samples)]
            probas = torch.stack(probas, dim=0)
            return probas
        else:
            return F.softmax(logits, dim=1)

    def update_precisions(self, inputs):
        logits, Phi = self.forward(inputs)
        preds = F.softmax(logits, dim=1)
        Phi = Phi.view(-1, self.hidden_size, 1)  # (batch_size, hidden_size, 1)
        PhiT = Phi.permute(0, 2, 1)  # (batch_size, 1, hidden_size)
        kernels = torch.matmul(Phi, PhiT)
        for k in range(self.output_size):
            temp = preds[:, k]*(1-preds[:, k])
            temp = temp.unsqueeze(1)
            update = (temp[:, :, None]*kernels).sum(dim=0)  # Sum over batch
            self.precisions[k] = (1-self.m)*self.precisions[k] + self.m*update

    def compute_covariances(self):
        self.covariances = torch.inverse(self.precisions)

class SNGP(nn.Module):
    def __init__(self, hidden_map, gp_input_size=128, gp_hidden_size=1024, gp_output_size=10, use_spectral_norm=False, use_gp_layer=True, **kwargs):
        super(SNGP, self).__init__()
        
        self.hidden_map = hidden_map(use_spectral_norm=use_spectral_norm, **kwargs)

        self.use_gp_layer = use_gp_layer
        if use_gp_layer:
            self.output_layer = GP(gp_input_size, gp_hidden_size, gp_output_size)  # Should maybe pass scale, s and m here
        else:
            self.output_layer = nn.Linear(gp_input_size, gp_output_size)

    def forward(self, inputs):
        encodings = self.hidden_map(inputs)
        return self.output_layer(encodings)

    def update_precisions(self, inputs):
        if self.use_gp_layer:
            with torch.no_grad():
                self.eval()
                encodings = self.hidden_map(inputs)
                self.output_layer.update_precisions(encodings)

    def compute_covariances(self):
        if self.use_gp_layer:
            self.output_layer.compute_covariances()

    def predict_proba(self, inputs):
        with torch.no_grad():
            self.eval()
            encodings = self.hidden_map(inputs)
            if self.use_gp_layer:
                return self.output_layer.predict_proba(encodings)
            else:
                logits = self.output_layer(encodings)
                return F.softmax(logits, dim=1)

class SNGPClassifier(NeuralNetClassifier):
    def __init__(self, **kwargs):
        super(SNGPClassifier, self).__init__(**kwargs)

    def get_loss(self, yp, y, X=None, training=False):
        yp = yp[0] if isinstance(yp, tuple) else yp
        return super(SNGPClassifier, self).get_loss(yp, y)

    @property
    def _default_callbacks(self):
        return [
            ('train_acc', EpochScoring(scoring='accuracy', lower_is_better=False, on_train=True, name='train_acc')),
            ('valid_acc', EpochScoring(scoring='accuracy', lower_is_better=False, name='valid_acc'))
        ]

    def on_batch_begin(self, net, Xi=None, yi=None, training=False, **kwargs):
        pass
    
    def on_batch_end(self, net, Xi=None, yi=None, training=False, **kwargs):
        pass

    def on_epoch_begin(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        self.history.new_epoch()
        self.history[-1].pop('batches')  # No batch information is stored
        self.history.record('epoch', len(self.history))

    def on_epoch_end(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        def _compute_loss_on_dataset(net, dataset, training):
            losses = []
            for data in net.get_iterator(dataset, training):
                Xi, yi = unpack_data(data)
                with torch.no_grad():
                    net.module_.eval()
                    yp = net.infer(Xi)
                loss = self.get_loss(yp, yi)
                losses += [loss.item()]
            return np.mean(losses)
        train_loss = _compute_loss_on_dataset(net, dataset_train, training=True)
        net.history.record('train_loss', train_loss)
        if dataset_valid:
            valid_loss = _compute_loss_on_dataset(net, dataset_valid, training=False)
            net.history.record('valid_loss', valid_loss)

    def on_train_begin(self, net, X=None, y=None, **kwargs):
        pass

    def on_train_end(self, net, X=None, y=None):
        dataset_train, _ = self.get_split_datasets(X, y)
        for data in net.get_iterator(dataset_train, training=True):
            Xi, _ = unpack_data(data)
            net.module_.update_precisions(Xi)
        net.module_.compute_covariances()

    def run_single_epoch(self, dataset, training, prefix, step_fn, **fit_params):
        is_placeholder_y = uses_placeholder_y(dataset)
        batch_count = 0
        for data in self.get_iterator(dataset, training=training):
            Xi, yi = unpack_data(data)
            yi_res = yi if not is_placeholder_y else None
            self.notify('on_batch_begin', X=Xi, y=yi_res, training=training)
            step = step_fn(Xi, yi, **fit_params)
            self.notify('on_batch_end', X=Xi, y=yi_res, training=training, **step)
            batch_count += 1
        self.history.record(prefix + '_batch_count', batch_count)

    def predict_proba(self, X):
        y_probas = []
        dataset = self.get_dataset(X)
        for data in self.get_iterator(dataset, training=False):
            Xi = unpack_data(data)[0]
            yp = self.module_.predict_proba(Xi)
            y_probas.append(to_numpy(yp))
        y_probas = np.concatenate(y_probas, 0)
        return y_probas

    def predict(self, X):
        yp = self.predict_proba(X)
        return yp.argmax(axis=1)

    def score(self, X, y):
        yp = self.predict(X)
        return accuracy_score(y, yp)
