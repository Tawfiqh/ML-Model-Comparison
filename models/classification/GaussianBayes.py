from sklearn.naive_bayes import GaussianNB

from .BaseClassificationModel import BaseClassificationModel


class GaussianNBClassifier(BaseClassificationModel):
    def find_hyper_paramters(self, dataset, test_dataset):
        pass

    def fit(self, dataset, dataset_train):
        X = dataset[0]
        y = dataset[1]

        self._fit_hyperparameters(X, y, dataset_train)

    def _fit_hyperparameters(self, X, y, test_dataset):
        self.model = GaussianNB()

        self.model.fit(X, y)
        if test_dataset:
            train_score = self.model.score(*test_dataset)
            return train_score
        return None
