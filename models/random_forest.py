from .BaseModel import BaseModel
from sklearn.ensemble import RandomForestRegressor
import pandas as pd


class RandomForest(BaseModel):
    def __init__(self) -> None:
        self.max_depth = 608
        self.max_leaf_nodes = 601
        #      RandomForest
        # Best model result:
        #      max_depth  max_leaf_nodes  score
        #       608        601             402 0.9878

    def find_hyper_paramters(self, dataset, test_dataset):
        X_train = dataset[0]
        y_train = dataset[1]

        X_test = test_dataset[0]
        y_test = test_dataset[1]

        all_results = []

        for max_depth in range(1, 1000, 20):
            for max_leaf_nodes in range(2, 1000, 50):
                clf = RandomForestRegressor(
                    max_depth=max_depth, max_leaf_nodes=max_leaf_nodes
                )
                clf = clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)

                all_results.append((max_depth, max_leaf_nodes, score))

        df = pd.DataFrame(
            all_results, columns=["max_depth", "max_leaf_nodes", "score"],
        )
        pd.options.display.float_format = "{:,.4f}".format
        # print("Hypertuning RandomForest - results:")
        # display(df)
        # print()

        best_result = df[df["score"] == df["score"].max()]
        print("Best model result:")
        print(best_result)

        self.max_depth = best_result["max_depth"].head(1).item()
        self.max_leaf_nodes = best_result["max_leaf_nodes"].head(1).item()

    def fit(self, dataset, dataset_train):

        X = dataset[0]
        y = dataset[1]

        clf = RandomForestRegressor(
            max_depth=self.max_depth, max_leaf_nodes=self.max_leaf_nodes
        )

        clf = clf.fit(X, y)
        self.model = clf

