from sklearn.neighbors import KNeighborsRegressor
from BaseModel import BaseModel
import pandas as pd


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
            for n_neighbors in range(1, 21):
                for leaf_size in range(10, 100, 10):
                    score = self._fit_hyperparameters(
                        X, y, n_neighbors, algorithm, leaf_size
                    )
                    results = [algorithm, n_neighbors, leaf_size, score]

                    print(
                        f"Trained model with algorithm-{algorithm}  n_neighbors-{n_neighbors}  leaf_size-{leaf_size}  and score:{score}"
                    )
                    all_results.append(results)

        df = pd.DataFrame(
            all_results, columns=["algorithm", "n_neighbors", "leaf_size", "score"],
        )
        pd.options.display.float_format = "{:,.4f}".format
        print("Hypertuning k-nearest - results:")
        print(df)
        print()

        best_result = df[df["score"] == df["score"].max()]
        print("Best model result:")
        print(best_result)

        self.algorithm = best_result["algorithm"].head(1).item()
        self.n_neighbors = best_result["n_neighbors"].head(1).item()
        self.leaf_size = best_result["leaf_size"].head(1).item()

    def fit(self, dataset):
        X = dataset[0]
        y = dataset[1]

        self.find_hyper_paramters(dataset)

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
