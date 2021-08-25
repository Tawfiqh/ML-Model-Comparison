from sklearn.tree import DecisionTreeClassifier

from .BaseClassificationModel import BaseClassificationModel
import pandas as pd


class DecisionTreeClassification(BaseClassificationModel):
    def __init__(self) -> None:
        self.n_neighbors = 5
        self.weights = "uniform"
        self.algorithm = "auto"
        self.leaf_size = 30
        self.p = 2

    def find_hyper_paramters(self, dataset, test_dataset):

        X = dataset[0]
        y = dataset[1]

        all_results = []
        for weight in ["uniform", "distance"]:
            for p in [1, 2]:
                for algorithm in ["auto", "ball_tree", "kd_tree", "brute"]:
                    for n_neighbors in range(2, 90, 10):
                        for leaf_size in range(10, 100, 10):
                            score = self._fit_hyperparameters(
                                X,
                                y,
                                test_dataset,
                                n_neighbors,
                                weight,
                                algorithm,
                                leaf_size,
                                p,
                            )
                            results = [
                                n_neighbors,
                                weight,
                                algorithm,
                                leaf_size,
                                p,
                                score,
                            ]

                            # print(
                            #     f"Trained model with algorithm-{algorithm}  n_neighbors-{n_neighbors}  leaf_size-{leaf_size}  and score:{score}"
                            # )
                            all_results.append(results)

        df = pd.DataFrame(
            all_results,
            columns=["n_neighbors", "weight", "algorithm", "leaf_size", "p", "score"],
        )
        pd.options.display.float_format = "{:,.4f}".format
        # print("Hypertuning k-nearest - results:")
        # print(df)
        # print()

        best_result = df[df["score"] == df["score"].max()]
        # print("Best model result:")
        # print(best_result)

        self.n_neighbors = best_result["n_neighbors"].head(1).item()
        self.weight = best_result["weight"].head(1).item()
        self.algorithm = best_result["algorithm"].head(1).item()
        self.leaf_size = best_result["leaf_size"].head(1).item()
        self.p = best_result["p"].head(1).item()

    def fit(self, dataset, dataset_train):
        X = dataset[0]
        y = dataset[1]

        self._fit_hyperparameters(
            X,
            y,
            dataset_train,
            self.n_neighbors,
            self.weights,
            self.algorithm,
            self.leaf_size,
            self.p,
        )

    def _fit_hyperparameters(
        self,
        X,
        y,
        test_dataset,
        n_neighbors,
        weights,
        algorithm,
        leaf_size,
        p,
    ):
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
        )
        # KNeighborsClassifier(n_neighbors=5,
        #  weights='uniform',
        #  algorithm='auto',
        #  leaf_size=30,
        #  p=2,
        #  metric='minkowski',
        #  metric_params=None,
        #  n_jobs=None)s

        self.model.fit(X, y)
        if test_dataset:
            train_score = self.model.score(*test_dataset)
            return train_score
        return None
