# now, each person pick a model we’ve learnt about  ✅
#   - create a new file for that model in the repo ✅
#   - create a function to fit the model to some data, and takes in several different hyperparameter values
#   - create another function which generates random hyperparameter values
#   - create a final function which calls both of the above functions and returns a list of dictionaries of the train, val and test scores

from scipy.sparse import data
from sklearn.neighbors import KNeighborsRegressor
from BaseModel import BaseModel


class KNearest(BaseModel):
    def __init__(self) -> None:
        self.algorithm = "auto"
        self.n_neighbors = 5
        self.leaf_size = 30

    def find_hyper_paramters(self, dataset):
        X = dataset[0]
        y = dataset[1]

        all_results = []

        for algorithm in ["auto", "ball_tree", "kd_tree", "brute"]:
            for n_neighbors in range(20):
                for leaf_size in range(0, 100, 10):
                    score = self._fit_hyperparameters(
                        X, y, n_neighbors, algorithm, leaf_size
                    )
                    results = [algorithm, n_neighbors, leaf_size, score]
                    all_results.append(results)

    def fit(self, dataset):
        X = dataset[0]
        y = dataset[1]

        # self.find_hyper_paramters(dataset)

        algorithm = self.algorithm
        n_neighbors = self.n_neighbors
        leaf_size = self.leaf_size
        score = self._fit_hyperparameters(X, y, n_neighbors, algorithm, leaf_size)

        print(
            f"Trained model with algorithm-{algorithm}  n_neighbors-{n_neighbors}  leaf_size-{leaf_size}  and score:{score}"
        )
        return

    def _fit_hyperparameters(self, X, y, n_neighbors, algorithm, leaf_size):
        self.model = KNeighborsRegressor(
            n_neighbors=n_neighbors, algorithm=algorithm, leaf_size=leaf_size
        )
        self.model.fit(X, y)
        train_score = self.model.score(X, y)
        return train_score
