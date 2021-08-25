from sklearn.metrics import mean_squared_error
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

    def _mse(self, dataset):
        X_true = dataset[0]
        y_true = dataset[1]

        y_pred = self.model.predict(X_true)

        return mean_squared_error(y_true, y_pred)

    def score_all_mse(self, train, test, val):
        train_score = self._mse(train)
        test_score = self._mse(test)
        val_score = self._mse(val)
        return train_score, test_score, val_score

    def mae(self, data_set):
        X_df = pd.DataFrame(data_set[0])
        y = pd.DataFrame(data_set[1])

        y_hat = self.model.predict(data_set[0])
        y_hat = pd.DataFrame(y_hat)

        X_df["y"] = y
        X_df["y_hat"] = y_hat

        X_df["error"] = X_df["y"] - X_df["y_hat"]
        X_df["absolute_error"] = X_df["error"].abs()
        mean_absolute_error = np.mean(X_df["absolute_error"])
        return mean_absolute_error
