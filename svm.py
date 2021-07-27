#%%
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import datasets
import numpy as np
dataset = datasets.load_boston()
from sklearn import model_selection
from sklearn import preprocessing
X = dataset['data']
y = dataset['target']
#%%
class Svm_svr:
    
    def __init__(self) -> None:
       
        self.param_grid = {
                    #'C': np.logspace(-3, 2, 6), 
                    'kernel': ['linear'],
                    #'degree': np.logspace(2, 3, 4),
                    #'gamma' : np.logspace(-3, 2, 6)
                    'epsilon': [ 0.5, 0.1, 1.5]
                    } 
        self.clf = GridSearchCV(SVR(), self.param_grid ,cv=5, scoring='accuracy')
        

    def fit(self, dataset):
        X = dataset['data']
        y = dataset['target']
        self.clf.fit(X, y)
        best_parameters = self.clf.best_params_
        final_model = self.clf.best_estimator_
    
    def score_all():
        final_model.score(train), final_model.score(test), final_model.score(val)
       

#%%
model = Svm_svr()
final = model.fit_model(dataset)
# %%
def generate_random_seed():
    return 3
random_seed = generate_random_seed()
def get_boston_train_test_val_datasets():
    X, y = datasets.load_boston(return_X_y=True)

    # random-state fr test_split will default to using the global random state instance from numpy.random. Calling the function multiple times will reuse the same instance, and will produce different results.
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=random_seed
    )
    X_train, X_val, y_train, y_val = model_selection.train_test_split(
        X_train, y_train, test_size=0.25, random_state=random_seed
    )  # 0.25 x 0.8 = 0.2

    # Normalise data before returning it
    sc = preprocessing.StandardScaler()
    sc.fit(X_train)
    X_train_normalised = sc.transform(X_train)
    X_test_normalised = sc.transform(X_test)
    X_val_normalised = sc.transform(X_val)

    train = (X_train_normalised, y_train)
    test = (X_test_normalised, y_test)
    val = (X_val_normalised, y_val)

    return {"train": train, "test": test, "val": val}
data_set = get_boston_train_test_val_datasets()
# %%
X,y = data_set["train"]
final.score(X,y) #, model.score(data_set["test"]),model.score(data_set["val"])

# %%
