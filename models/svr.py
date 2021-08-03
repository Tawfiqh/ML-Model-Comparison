from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from .BaseModel import BaseModel
import numpy as np


class Svr(BaseModel):
    def __init__(self) -> None:

        self.param_grid = {
            #'C': np.logspace(-3, 2, 6),
            "kernel": ["linear"],
            #'degree': np.logspace(2, 3, 4),
            #'gamma' : np.logspace(-3, 2, 6)
            "epsilon": [0.5],  # , 0.1, 1.5]
        }
        self.grid_search = GridSearchCV(SVR(), self.param_grid, cv=5)
        self.model = 0

    def fit(self, dataset, dataset_train):
        X = np.concatenate((dataset[0], dataset_train[0]), axis=0)
        y = np.concatenate((dataset[1], dataset_train[1]), axis=0)

        self.grid_search.fit(X, y)
        self.model = self.grid_search.best_estimator_
