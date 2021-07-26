from sklearn.ensemble import RandomForestRegressor

class RandomForest_Regressor:
    def __init__(self) -> None:
        self.rf_reg = RandomForest_Regressor()

    def fit(self, dataset):
        X = dataset[0]
        y = dataset[1]
        self.rf_reg.fit(X, y)

    def score_all(self, train, test, val):
        return (self.rf_reg.score(train[0], train[1]),
                self.rf_reg.score(test[0], test[1]),
                self.rf_reg.score(val[0], val[1]))