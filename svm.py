from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV


class Svm_svr:
    def __init__(self) -> None:

        self.param_grid = {
            #'C': np.logspace(-3, 2, 6),
            "kernel": ["linear"],
            #'degree': np.logspace(2, 3, 4),
            #'gamma' : np.logspace(-3, 2, 6)
            "epsilon": [0.5],  # , 0.1, 1.5]
        }
        self.grid_search = GridSearchCV(
            SVR(), self.param_grid, cv=5, scoring="accuracy"
        )
        self.final_model = 0

    def fit(self, dataset):
        X = dataset[0]
        y = dataset[1]
        self.grid_search.fit(X, y)
        self.final_model = self.grid_search.best_estimator_

    def score_all(self, train, test, val):
        return (
            self.final_model.score(*train),
            self.final_model.score(*test),
            self.final_model.score(*val),
        )
