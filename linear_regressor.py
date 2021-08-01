from BaseModel import BaseModel
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from BaseModel import BaseModel


class LinearRegressor(BaseModel):
    def fit(self, dataset, dataset_train):

        X = dataset[0]
        y = dataset[1]

        self.model = linear_model.LinearRegression()
        self.model.fit(X, y)
