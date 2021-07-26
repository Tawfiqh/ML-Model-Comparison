class Svc:
    from sklearn.svm import SVC
    def __init__(self) -> None:
        pass

    def fit(self, dataset):
        #C : float, default=1.0
        #kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
        X = dataset[0]
        y = dataset[1]
        all_scores = []
        for i in ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']:
                svc = SVC(kernel= i, gamma='auto').fit(X, y) # gamma='scale'
                svc.fit(X,y)
                all_scores.append(svc.score(X, y, sample_weight=None))


    def score_all(self, train, test, val):
        pass
        