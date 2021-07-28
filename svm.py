from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from BaseModel import BaseModel


class Svm_svr(BaseModel):
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
        self.model = 0

    def fit(self, dataset, dataset_train):
        X = dataset[0]
        y = dataset[1]
        self.grid_search.fit(X, y)
        self.model = self.grid_search.best_estimator_
