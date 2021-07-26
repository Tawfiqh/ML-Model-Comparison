from sklearn.ensemble import RandomForestRegressor

class RandomForest_Regressor:
    def __init__(self) -> None:
        self.rf_reg = RandomForest_Regressor()
    
    def _fit_hyperparameters():
        """
        Hyperparameters to tune:
        - n_estimatorsint, default=100
        - max_depthint, default=None
        - min_samples_splitint or float, default=2
        - min_samples_leafint or float, default=1
        - min_weight_fraction_leaffloat, default=0.0
        - max_features{“auto”, “sqrt”, “log2”}, int or float, default=”auto”
        - max_leaf_nodesint, default=None
        - min_impurity_decreasefloat, default=0.0
        - min_impurity_splitfloat, default=None
        - bootstrapbool, default=True
        - oob_scorebool, default=False
        - n_jobsint, default=None
        - random_stateint, RandomState instance or None, default=None
        - verboseint, default=0
        - warm_startbool, default=False
        - ccp_alphanon-negative float, default=0.0
        - max_samplesint or float, default=None
        """
        list_of_hyperparameters = ["n_estimators", "criterion", "max_depth",
            "min_samples_split", "min_samples_leaf", "min_weight_fraction_leaf"]
            
        self.rf_reg = Random(n_estimators=100, criterion='mse', max_depth=None,
            min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
            max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None,
            random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0,
            max_samples=None)

    def fit(self, dataset):
        X = dataset[0]
        y = dataset[1]
        self.rf_reg.fit(X, y)

    def score_all(self, train, test, val):
        return (self.rf_reg.score(train[0], train[1]),
                self.rf_reg.score(test[0], test[1]),
                self.rf_reg.score(val[0], val[1]))