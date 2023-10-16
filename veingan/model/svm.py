import os.path
from typing import Any, AnyStr, Dict

import joblib
from sklearn.svm import OneClassSVM


class OneSVM:

    def __init__(self, saved_model: Any = None, param: Dict = None):
        self.model = saved_model
        self.param = param or {'gamma': 'scale', 'kernel': 'poly', 'nu': 0.01}

        if not self.model:
            self.model = OneClassSVM(**self.param)

    @classmethod
    def load_pretrained(cls, pretrained_file: AnyStr):
        if not os.path.exists(pretrained_file):
            raise FileNotFoundError(
                f'{cls.__name__} Failed Retrieving Pretrained File with Filename {pretrained_file}.')
        return cls(saved_model=joblib.load(pretrained_file))

    def fit(self, X: Any):
        self.model.fit(X)
        return

    def predict(self, X: Any):
        return self.model.predict(X)

    def score_samples(self, X: Any):
        return self.model.score_samples(X)
