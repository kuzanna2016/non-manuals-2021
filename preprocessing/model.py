import json
import os
from tqdm import tqdm
tqdm.pandas()
import pickle
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import ShuffleSplit

from forecasting_metrics import evaluate


class Model:

    def __init__(self, model, name=None, save_to=None):
        self.model = model
        if name is not None:
            self.name = name
        else:
            self.name = self._make_name()
        self.fp = save_to
        self.logs = {}
        self.weights = None
        self.features = None
        self.scaler = StandardScaler()

    @classmethod
    def get_model(self, kind='mlp', params=None, name=None, save_to=None):
        if kind == 'mlp':
            m = self.mlp(params or {}, name=name, save_to=save_to)
        elif kind == 'lasso':
            m = self.lr(params, regularizer='lasso', name=name, save_to=save_to)
        elif kind == 'ridge':
            m = self.lr(params, regularizer='ridge', name=name, save_to=save_to)
        else:
            raise ValueError('no such kind of model')
        return m

    @classmethod
    def lr(cls, params, regularizer='lasso', name=None, save_to=None):
        if regularizer == 'lasso':
            model = Lasso(**params)
        elif regularizer == 'ridge':
            model = Ridge(**params)
        else:
            raise ValueError('no such regularizer')
        return cls(model, name=name, save_to=save_to)

    @classmethod
    def mlp(cls, params, name=None, save_to=None):
        model = MLPRegressor(**params)
        return cls(model, name=name, save_to=save_to)

    def _make_name(self):
        return str(self.model.__class__.__name__)

    def train(self, X, y, X_val=None, y_val=None, metrics=None):
        self.features = X.columns
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        if metrics is not None:
            self.logs['train'] = evaluate(y, y_pred, metrics)
            self.logs['train']['std_true'] = y.std().tolist()
            self.logs['train']['std_pred'] = y_pred.std().tolist()
            self.logs['train']['mean_true'] = y.mean().tolist()
            self.logs['train']['mean_pred'] = y_pred.mean().tolist()
        if X_val is not None and y_val is not None:
            y_pred = self.predict(X_val)
            self.logs['val'] = evaluate(y_val, y_pred, metrics)
            self.logs['val']['std_true'] = y_val.std().tolist()
            self.logs['val']['std_pred'] = y_pred.std().tolist()
            self.logs['val']['mean_true'] = y_val.mean().tolist()
            self.logs['val']['mean_pred'] = y_pred.mean().tolist()

    def save_model(self):
        if self.fp is not None:
            path = self.fp + 'model_' + self.name
            if not os.path.isdir(path):
                os.mkdir(path)
            pickle.dump(self.model, open(path + '/model.pkl', 'wb'))

    def load_model(self):
        self.model = pickle.load(open(self.fp + 'model_' + self.name + '/model.pkl', 'rb'))

    def save_logs(self, logs, is_cv=False):
        if is_cv:
            name = 'cv_logs'
        else:
            name = 'logs'
        with open(self.fp + 'model_' + self.name + f'/{name}.json', 'w') as f:
            json.dump(logs, f)

    @staticmethod
    def load_logs(fp, is_cv=False):
        if is_cv:
            name = 'cv_logs'
        else:
            name = 'logs'
        with open(fp + f'/{name}.json', 'r') as f:
            logs = json.load(f)
        return logs

    def predict(self, X):
        X = self.scaler.transform(X)
        y_pred = self.model.predict(X)
        return y_pred

    def plot_weights(self, w_intercept=False):
        if w_intercept:
            w = self.weights.reshape(-1, 1).T[1:]
        else:
            w = self.weights.reshape(-1, 1).T
        plt.matshow(w, cmap=plt.cm.bwr)
        plt.xticks(np.arange(0, w.shape[0]), self.features, rotation=90)
        plt.yticks(())
        plt.colorbar(orientation='horizontal')
        plt.show()

    @staticmethod
    def _leave_n_out(features, n, features_subset):
        static_features = [f for f in features if f not in features_subset]
        for comb in combinations(features_subset, len(features_subset) - n):
            n_out = [f for f in features_subset if f not in comb]
            yield n_out, comb + static_features

    def cross_validate_features(self, X, y, features_subset, n_out=1, n_splits=5, test_size=0.25, random_state=0):
        ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        cv_logs = []
        features = X.columns()
        for n_out, features_set in self._leave_n_out(features, n_out, features_subset):
            logs = {'n_out': n_out,
                    'train': {'folds': [], 'mean_score': {}},
                    'val': {'folds': [], 'mean_score': {}}}
            for train_index, val_index in ss.split(X):
                X = X[features_set]
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                self.train(X_train, y_train, X_val, y_val)
                logs['train']['folds'].append(self.logs['train'])
                logs['val']['folds'].append(self.logs['val'])

            for subset in ['train', 'val']:
                mse = 0
                mae = 0
                for score in logs[subset]['folds']:
                    mse += score['mse']
                    mae += score['mae']
                logs[subset]['mean_score']['mse'] = mse / n_splits
                logs[subset]['mean_score']['mae'] = mae / n_splits

            cv_logs.append(logs)
        self.save_logs(cv_logs)
        return cv_logs
