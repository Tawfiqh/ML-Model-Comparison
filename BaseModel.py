class BaseModel:
    def fit(self, dataset):
        raise Exception(
            "To be implemented -- this should be implemented in the subclass."
        )

    def score_all(self, train, test, val):
        train_score = self.model.score(*train)
        test_score = self.model.score(*test)
        val_score = self.model.score(*val)

        return train_score, test_score, val_score

