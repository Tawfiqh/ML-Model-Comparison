#%%
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import datasets
import numpy as np
dataset = datasets.load_boston(return_X_y=True)
class Svc:
    
    def __init__(self) -> None:
        pass


    def fit(self, dataset):
        #C : float, default=1.0
        #kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
        X = dataset[0]
        y = dataset[1]
        param_grid = {
            'C': np.logspace(-3, 2, 6), 
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': np.logspace(-3, 2, 6)
            }
        clf = GridSearchCV(SVC(), param_grid,cv=5,scoring='accuracy')
        clf.fit(X, y)
        print(clf.best_params_)

        
    def score_all(self, train, test, val):
        pass

a = Svc()
a.fit(dataset)

# %%
