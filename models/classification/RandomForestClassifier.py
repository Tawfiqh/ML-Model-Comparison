from sklearn.ensemble import RandomForestClassifier

from .BaseClassificationModel import BaseClassificationModel
import pandas as pd


class RandomForestClassification(BaseClassificationModel):
    def __init__(self) -> None:
        self.n_estimators = 100
        self.criterion = "gini"  # {“gini”, “entropy”}
        self.max_depth = None
        self.max_leaf_nodes = None
        self.warm_start = False

    def find_hyper_paramters(self, dataset, test_dataset):

        X = dataset[0]
        y = dataset[1]

        all_results = []

        max_depth = self.max_depth
        warm_start = self.warm_start
        criterion = "entropy"  # self.criterion
        for n_estimators in range(100, 1000, 100):
            for criterion in ["gini", "entropy"]:
                for max_leaf_nodes in [None, *range(2, 100, 10)]:

                    score = self._fit_hyperparameters(
                        X,
                        y,
                        test_dataset,
                        n_estimators,
                        criterion,
                        max_depth,
                        max_leaf_nodes,
                        warm_start,
                    )
                    results = [
                        n_estimators,
                        criterion,
                        max_depth,
                        max_leaf_nodes,
                        warm_start,
                        score,
                    ]
                    print(
                        f"n_estimators={n_estimators}  criterion={criterion}  max_depth={max_depth}  max_leaf_nodes={max_leaf_nodes}  warm_start={warm_start}  score={score}"
                    )
                    all_results.append(results)

        df = pd.DataFrame(
            all_results,
            columns=[
                "n_estimators",
                "criterion",
                "max_depth",
                "max_leaf_nodes",
                "warm_start",
                "score",
            ],
        )
        pd.options.display.float_format = "{:,.4f}".format

        best_result = df[df["score"] == df["score"].max()]
        print("Best model result:")
        print(best_result)

        self.n_estimators = best_result["n_estimators"].head(1).item()
        self.criterion = best_result["criterion"].head(1).item()
        self.max_depth = best_result["max_depth"].head(1).item()
        self.max_leaf_nodes = best_result["max_leaf_nodes"].head(1).item()
        self.warm_start = best_result["warm_start"].head(1).item()

    def fit(self, dataset, dataset_train):
        X = dataset[0]
        y = dataset[1]

        self._fit_hyperparameters(
            X,
            y,
            dataset_train,
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            max_leaf_nodes=self.max_leaf_nodes,
            warm_start=self.warm_start,
        )

    def _fit_hyperparameters(
        self,
        X,
        y,
        test_dataset,
        n_estimators,
        criterion,
        max_depth,
        max_leaf_nodes,
        warm_start,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes,
            warm_start=warm_start,
        )

        self.model.fit(X, y)
        if test_dataset:
            train_score = self.model.score(*test_dataset)
            return train_score
        return None
