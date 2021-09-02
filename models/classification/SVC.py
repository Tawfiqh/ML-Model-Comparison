from sklearn.svm import SVC


from .BaseClassificationModel import BaseClassificationModel
import pandas as pd


class SVClassifier(BaseClassificationModel):
    def __init__(self) -> None:
        self.break_ties = False

    def find_hyper_paramters(self, dataset, test_dataset):

        X = dataset[0]
        y = dataset[1]

        all_results = []
        for break_ties in [True, False]:
            score = self._fit_hyperparameters(
                X,
                y,
                test_dataset,
                break_ties,
            )

            results = [
                break_ties,
                score,
            ]

            all_results.append(results)

        df = pd.DataFrame(
            all_results,
            columns=["break_ties", "score"],
        )
        pd.options.display.float_format = "{:,.4f}".format

        best_result = df[df["score"] == df["score"].max()]
        # print("Best model result:")
        # print(best_result)

        self.break_ties = best_result["break_ties"].head(1).item()

    def fit(self, dataset, dataset_train):
        X = dataset[0]
        y = dataset[1]

        self._fit_hyperparameters(X, y, dataset_train, self.break_ties)

    def _fit_hyperparameters(self, X, y, test_dataset, break_ties):
        self.model = SVC(break_ties=break_ties)

        self.model.fit(X, y)
        if test_dataset:
            train_score = self.model.score(*test_dataset)
            return train_score
        return None
