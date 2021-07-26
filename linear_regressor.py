from sklearn import linear_model
class LinearRegressor:
    def __init__(self) -> None:
        self.model = linear_model.LinearRegression(fit_intercept=))

    def fit(self, dataset):
        X = dataset[0]
        y = dataset[1]
        self.model.fit(X, y)

    def score_all(self, train, test, val):
        score_train = self.model.score(*train)
        score_test = self.model.score(*test])
        score_val = self.model.score(*val)
        return score_train, score_test, score_val

# {'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'normalize': False, 'positive': False}