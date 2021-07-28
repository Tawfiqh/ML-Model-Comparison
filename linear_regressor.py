from BaseModel import BaseModel
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from BaseModel import BaseModel


class LinearRegressor(BaseModel):
    def __init__(self) -> None:
        self.model = linear_model.LinearRegression()

    def fit(self, dataset):

        X = dataset[0]
        y = dataset[1]

        # defining parameter range
        param_grid = {
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "copy_X": [True, False],
        }

        self.model = GridSearchCV(self.model, param_grid)

        self.model.fit(X, y)

        # # print best parameter after tuning
        # print(self.model.get_params().keys())

        # # print best parameter before tuning
        # # print(self.model.best_params_)

        # # print best parameter after tuning
        # print(self.grid_model.best_params_)

        # print (f'r2 / variance : {self.grid_model.best_score_}')

        # print('Residual sum of squares: %.2f'
        #       % np.mean((self.grid_model.predict(X) - y) ** 2))
