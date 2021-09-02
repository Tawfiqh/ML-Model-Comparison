from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from .BaseClassificationModel import BaseClassificationModel
import pandas as pd


class LinearDiscriminantClassifier(BaseClassificationModel):
    def __init__(self) -> None:
        self.solver = "svd"

    def find_hyper_paramters(self, dataset, test_dataset):

        X = dataset[0]
        y = dataset[1]

        all_results = []
        for solver in ["svd", "lsqr", "eigen"]:
            score = self._fit_hyperparameters(
                X,
                y,
                test_dataset,
                solver,
            )
            results = [
                solver,
                score,
            ]

            # print(
            #     f"Trained model with algorithm-{algorithm}  n_neighbors-{n_neighbors}  leaf_size-{leaf_size}  and score:{score}"
            # )
            all_results.append(results)

        df = pd.DataFrame(
            all_results,
            columns=["solver", "score"],
        )
        pd.options.display.float_format = "{:,.4f}".format

        best_result = df[df["score"] == df["score"].max()]
        # print("Best model result:")
        # print(best_result)

        self.solver = best_result["solver"].head(1).item()

    def fit(self, dataset, dataset_train):
        X = dataset[0]
        y = dataset[1]

        self._fit_hyperparameters(
            X,
            y,
            dataset_train,
            self.solver,
        )

    def _fit_hyperparameters(
        self,
        X,
        y,
        test_dataset,
        solver,
    ):
        self.model = LinearDiscriminantAnalysis(
            solver=solver,
        )

        self.model.fit(X, y)
        if test_dataset:
            train_score = self.model.score(*test_dataset)
            return train_score
        return None
