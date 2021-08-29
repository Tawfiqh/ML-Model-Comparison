from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import numpy as np


class BaseClassificationModel:
    def fit(self, dataset):
        raise Exception(
            "To be implemented -- this should be implemented in the subclass."
        )

    def score_all(self, train, test, val):
        train_score = self.model.score(*train)
        test_score = self.model.score(*test)
        val_score = self.model.score(*val)

        return train_score, test_score, val_score

    def _f1_score(self, dataset):
        X_true = dataset[0]
        y_true = dataset[1]

        y_pred = self.model.predict(X_true)

        return f1_score(y_true, y_pred)

    def score_precision(self, dataset):
        X_true = dataset[0]
        y_true = dataset[1]

        y_pred = self.model.predict(X_true)

        return precision_score(y_true, y_pred)

    def score_recall(self, dataset):
        X_true = dataset[0]
        y_true = dataset[1]

        y_pred = self.model.predict(X_true)

        return recall_score(y_true, y_pred)

    def score_all_f1(self, train, test, val):
        train_score = self._f1_score(train)
        test_score = self._f1_score(test)
        val_score = self._f1_score(val)
        return train_score, test_score, val_score
