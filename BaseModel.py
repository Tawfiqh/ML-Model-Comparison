from sklearn.metrics import mean_squared_error


class BaseModel:
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

