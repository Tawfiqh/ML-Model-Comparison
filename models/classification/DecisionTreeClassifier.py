from sklearn.tree import DecisionTreeClassifier

from .BaseClassificationModel import BaseClassificationModel
import pandas as pd


class DecisionTreeClassification(BaseClassificationModel):
    def __init__(self) -> None:
        self.criterion = "gini"
        self.max_leaf_nodes = None
        self.splitter = "best"
        self.min_samples_leaf = 1
        self.max_depth = None
        self.min_samples_split = 2

    def find_hyper_paramters(self, dataset, test_dataset):

        X = dataset[0]
        y = dataset[1]

        all_results = []
        criterion = self.criterion
        splitter = self.splitter
        max_leaf_nodes = self.max_leaf_nodes
        # for criterion in ["entropy", "gini"]:
        # for splitter in ["best", "random"]:
        # for max_leaf_nodes in [None, *range(2, 200, 10)]:
        for min_samples_leaf in range(1, 20, 5):
            for max_depth in range(2, 5000, 100):
                for min_samples_split in range(2, 20, 2):
                    score = self._fit_hyperparameters(
                        X,
                        y,
                        test_dataset,
                        criterion,
                        max_leaf_nodes,
                        splitter,
                        min_samples_leaf,
                        max_depth,
                        min_samples_split,
                    )
                    results = [
                        criterion,
                        max_leaf_nodes,
                        splitter,
                        min_samples_leaf,
                        max_depth,
                        min_samples_split,
                        score,
                    ]
                    print(f"score: {score} --- {results}")
                    all_results.append(results)

        df = pd.DataFrame(
            all_results,
            columns=[
                "criterion",
                "max_leaf_nodes",
                "splitter",
                "min_samples_leaf",
                "max_depth",
                "min_samples_split",
                "score",
            ],
        )
        pd.options.display.float_format = "{:,.4f}".format
        # print("Hypertuning k-nearest - results:")
        # print(df)
        # print()

        best_result = df[df["score"] == df["score"].max()]
        print("Best model result:")
        print(best_result)

        self.criterion = best_result["criterion"].head(1).item()
        self.max_leaf_nodes = best_result["max_leaf_nodes"].head(1).item()
        self.splitter = best_result["splitter"].head(1).item()
        self.min_samples_leaf = best_result["min_samples_leaf"].head(1).item()
        self.max_depth = best_result["max_depth"].head(1).item()
        self.min_samples_split = best_result["min_samples_split"].head(1).item()

    def fit(self, dataset, dataset_train):
        X = dataset[0]
        y = dataset[1]

        self._fit_hyperparameters(
            X,
            y,
            dataset_train,
            self.criterion,
            self.max_leaf_nodes,
            self.splitter,
            self.min_samples_leaf,
            self.max_depth,
            self.min_samples_split,
        )

    def _fit_hyperparameters(
        self,
        X,
        y,
        test_dataset,
        criterion,
        max_leaf_nodes,
        splitter,
        min_samples_leaf,
        max_depth,
        min_samples_split,
    ):
        self.model = DecisionTreeClassifier(
            criterion=criterion,
            max_leaf_nodes=max_leaf_nodes,
            splitter=splitter,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )

        self.model.fit(X, y)
        if test_dataset:
            train_score = self.model.score(*test_dataset)
            return train_score
        return None
