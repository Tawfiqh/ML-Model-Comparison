from sklearn.gaussian_process import GaussianProcessClassifier
from .BaseClassificationModel import BaseClassificationModel
import pandas as pd


class GaussianProcessClassification(BaseClassificationModel):
    def __init__(self) -> None:
        self.max_iter_predict = 100
        self.warm_start = False
        self.multi_class = "one_vs_rest"

    def find_hyper_paramters(self, dataset, test_dataset):

        X = dataset[0]
        y = dataset[1]

        all_results = []

        for max_iter_predict in range(100, 1000, 100):
            for warm_start in [True, False]:
                for multi_class in ["one_vs_rest", "one_vs_one"]:

                    score = self._fit_hyperparameters(
                        X,
                        y,
                        test_dataset,
                        max_iter_predict,
                        warm_start,
                        multi_class,
                    )
                    results = [
                        max_iter_predict,
                        warm_start,
                        multi_class,
                        score,
                    ]

                    # print(
                    #     f"Trained model with algorithm-{algorithm}  n_neighbors-{n_neighbors}  leaf_size-{leaf_size}  and score:{score}"
                    # )
                    all_results.append(results)

        df = pd.DataFrame(
            all_results,
            columns=[
                "max_iter_predict",
                "warm_start",
                "multi_class",
                "score",
            ],
        )
        pd.options.display.float_format = "{:,.4f}".format
        # print("Hypertuning k-nearest - results:")
        # print(df)
        # print()

        best_result = df[df["score"] == df["score"].max()]
        # print("Best model result:")
        # print(best_result)

        self.max_iter_predict = best_result["max_iter_predict"].head(1).item()
        self.warm_start = best_result["warm_start"].head(1).item()
        self.multi_class = best_result["multi_class"].head(1).item()

    def fit(self, dataset, dataset_train):
        X = dataset[0]
        y = dataset[1]

        self._fit_hyperparameters(
            X,
            y,
            dataset_train,
            self.max_iter_predict,
            self.warm_start,
            self.multi_class,
        )

    def _fit_hyperparameters(
        self,
        X,
        y,
        test_dataset,
        max_iter_predict,
        warm_start,
        multi_class,
    ):
        self.model = GaussianProcessClassifier(
            max_iter_predict=max_iter_predict,
            warm_start=warm_start,
            multi_class=multi_class,
        )

        self.model.fit(X, y)
        if test_dataset:
            train_score = self.model.score(*test_dataset)
            return train_score
        return None
