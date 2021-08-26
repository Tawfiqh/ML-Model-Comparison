from sklearn.linear_model import LogisticRegression

from .BaseClassificationModel import BaseClassificationModel
import pandas as pd


class LogisticRegressionClassifier(BaseClassificationModel):
    def __init__(self) -> None:
        self.penalty = "l2"
        self.fit_intercept = True
        self.solver = "lbfgs"
        self.max_iter = 100

    def find_hyper_paramters(self, dataset, test_dataset):

        X = dataset[0]
        y = dataset[1]

        all_results = []
        penalty = self.penalty
        # for penalty in ["l1", "l2", "elasticnet", "none"]:
        for fit_intercept in [True, False]:
            for solver in ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]:
                for max_iter in range(1, 1000, 100):
                    score = self._fit_hyperparameters(
                        X,
                        y,
                        test_dataset,
                        penalty,
                        fit_intercept,
                        solver,
                        max_iter,
                    )
                    results = [
                        penalty,
                        fit_intercept,
                        solver,
                        max_iter,
                        score,
                    ]

                    # print(
                    #     f"Trained model with algorithm-{algorithm}  n_neighbors-{n_neighbors}  leaf_size-{leaf_size}  and score:{score}"
                    # )
                    all_results.append(results)

        df = pd.DataFrame(
            all_results,
            columns=["penalty", "fit_intercept", "solver", "max_iter", "score"],
        )
        pd.options.display.float_format = "{:,.4f}".format

        best_result = df[df["score"] == df["score"].max()]
        print("Best model result:")
        print(best_result)

        penalty = best_result["penalty"].head(1).item()
        fit_intercept = best_result["fit_intercept"].head(1).item()
        solver = best_result["solver"].head(1).item()
        max_iter = best_result["max_iter"].head(1).item()

    def fit(self, dataset, dataset_train):
        X = dataset[0]
        y = dataset[1]

        self._fit_hyperparameters(
            X,
            y,
            dataset_train,
            self.penalty,
            self.fit_intercept,
            self.solver,
            self.max_iter,
        )

    def _fit_hyperparameters(
        self,
        X,
        y,
        test_dataset,
        penalty,
        fit_intercept,
        solver,
        max_iter,
    ):
        self.model = LogisticRegression(
            penalty=penalty,
            fit_intercept=fit_intercept,
            solver=solver,
            max_iter=max_iter,
        )

        self.model.fit(X, y)
        if test_dataset:
            train_score = self.model.score(*test_dataset)
            return train_score
        return None
