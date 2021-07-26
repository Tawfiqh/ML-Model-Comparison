class BaseModel:
    def score_all(self, train, test, val):
        train_score = self.model.score(*train)
        test_score = self.model.score(*test)
        val_score = self.model.score(*val)

        return train_score, test_score, val_score

