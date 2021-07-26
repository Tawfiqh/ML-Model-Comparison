from sklearn import linear_model
class LinearRegressor:
    def __init__(self) -> None:
        self.model = linear_model.LinearRegression()

    def fit(self, dataset):
        X = dataset[0]
        y = dataset[1]
        self.model.fit(X, y)

    def score_all(self, train, test, val):
        score_train = self.model.score(train[0], train[1])
        score_test = self.model.score(train[0], train[1])
        score_val = self.model.score(train[0], train[1])
        return score_train, score_test, score_val

# {'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'normalize': False, 'positive': False}